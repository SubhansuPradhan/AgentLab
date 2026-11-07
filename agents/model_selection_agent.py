"""
Model Selection and Tuning Agent

Job: find the right model for the dataset, then decide whether
fine-tuning is worth doing, and say why either way.

Step 1, choose the model. Four candidates are compared with 5-fold
cross validation and scored once on a held-out test set. The agent
recommends the SIMPLEST model whose cross-validated score is within one
standard error of the best (the "one standard error rule"). If a simple
model is as good as a complex one within the noise, there is no evidence
the complex one is better, so it is not recommended.

Step 2, decide on tuning. Rules, not the LLM, decide whether tuning is
required. It is NOT required when the model has no settings to tune, when
the features carry almost no signal, when the score is so perfect it
suggests a leak, or when there are too few rows for a reliable search. When
it IS required, the agent works out which settings to search and why, runs
Optuna, and compares tuned against default on data the search never saw.
The tuned model is adopted only if it clearly wins there.

The LLM only explains the decisions in plain language.

Inputs (from state):
    - feature_df: pandas DataFrame, output of the Feature Engineering Agent
    - target_column: str, name of the original target column
    - run_id: str, used to namespace where plots get saved
    - n_trials: int, optional, Optuna trials, defaults to 20
    - timeout_seconds: int, optional, tuning time limit, defaults to 300

Outputs (added to state):
    - task_type: "classification" or "regression"
    - model_name: str, the recommended model
    - trained_model: the recommended model with default settings, fitted
    - tuned_model: the tuned model if it was adopted, otherwise None
    - final_model: the model to use from here on (tuned or default)
    - final_model_choice: "tuned" or "default"
    - model_recommendation: dict, the pick and the reasons behind it
    - tuning_plan: dict, whether tuning is required, why, and which settings
    - tuning_ran: bool
    - best_params, tuning_history, tuning_comparison, param_importance,
      tuning_plot_path: results of the search (empty or None if not run)
    - model_metrics: dict, cross-validated scores for every model
    - stats_table / stats_table_str: full test metrics for every model
    - comparison_table / comparison_table_str: default versus tuned
    - confusion_matrix_path: str or None (classification only)
    - label_mapping: dict or None, encoded integer labels to text labels
    - train_test_split_info: dict, split sizes and random seed
    - model_selection_summary: str, LLM-written explanation
"""
import os
import re
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb

from agents.base_agent import BaseAgent
from prompts.model_selection_prompt import build_model_selection_prompt
from config.settings import settings

optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelSelectionAgent(BaseAgent):
    name = "model_selection_agent"

    CANDIDATES = {
        "classification": ["logistic_regression", "random_forest", "xgboost", "lightgbm"],
        "regression": ["linear_regression", "random_forest", "xgboost", "lightgbm"],
    }
    DEFAULT_PARAMS = {"random_forest": {"n_estimators": 200}}

    # Used to break near-ties: prefer the simpler, more explainable model.
    COMPLEXITY = {
        "logistic_regression": 0, "linear_regression": 0,
        "random_forest": 1, "lightgbm": 2, "xgboost": 3,
    }
    LINEAR_MODELS = {"logistic_regression", "linear_regression"}

    PARAM_MEANING = {
        "C": "regularization strength, smaller means a simpler decision boundary",
        "n_estimators": "number of trees",
        "max_depth": "how detailed each tree can get",
        "min_samples_split": "how many rows a branch needs before it may split",
        "min_samples_leaf": "minimum rows in each leaf, larger means more cautious splits",
        "learning_rate": "size of each boosting step, smaller means slower, more careful fitting",
        "subsample": "share of rows each tree sees, below 1.0 adds protective randomness",
        "colsample_bytree": "share of features each tree sees, below 1.0 adds protective randomness",
        "num_leaves": "maximum leaves per tree, the main size control in LightGBM",
        "min_child_samples": "minimum rows in each leaf, larger means more cautious splits",
    }

    SELECTION_CV_FOLDS = 5
    TUNING_CV_FOLDS = 3              # 3 folds keeps 20 trials fast on a laptop
    MIN_TRAIN_ROWS_FOR_TUNING = 300
    OVERFIT_GAP = {"classification": 0.15, "regression": 0.25}
    NEAR_CHANCE = {"classification": 0.6, "regression": 0.05}
    PERFECT = 0.98
    RANK_SIGNAL = 0.2                # Spearman, see _diagnostics

    # ------------------------------------------------------------------
    # main flow
    # ------------------------------------------------------------------
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        feature_df: pd.DataFrame = state["feature_df"]
        target_column: str = state["target_column"]
        run_id = state.get("run_id", "default_run")
        n_trials = state.get("n_trials", 20)
        timeout = state.get("timeout_seconds", 300)

        X, y, resolved_target = self._split_features_target(feature_df, target_column)
        task_type = self._detect_task_type(y)

        label_mapping = None
        if task_type == "classification" and not pd.api.types.is_numeric_dtype(y):
            encoder = LabelEncoder()
            y = pd.Series(encoder.fit_transform(y), index=y.index, name=y.name)
            label_mapping = {int(i): str(label) for i, label in enumerate(encoder.classes_)}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if task_type == "classification" else None,
        )
        n_classes = y_train.nunique() if task_type == "classification" else None
        scoring = self._scoring(task_type)
        names = self.CANDIDATES[task_type]

        # step 1: compare the candidates and recommend one
        metrics = self._cross_validate_all(names, X_train, y_train, task_type, scoring)
        fitted, stats_table, train_scores = self._fit_and_score_all(
            names, X_train, y_train, X_test, y_test, task_type, n_classes, metrics
        )
        recommendation = self._recommend_model(metrics)
        chosen = recommendation["model"]
        default_model = fitted[chosen]

        # step 2: decide whether tuning is required, and run it if so
        diagnostics = self._diagnostics(
            default_model, X_test, y_test, stats_table[chosen], train_scores[chosen],
            task_type, len(X_train),
        )
        self._add_signal_reasons(recommendation, diagnostics)
        plan = self._plan_tuning(chosen, task_type, diagnostics)

        tuning = None
        if plan["required"]:
            tuning = self._run_tuning(
                chosen, task_type, plan, X_train, y_train, X_test, y_test,
                scoring, n_classes, n_trials, timeout, run_id,
            )
            if tuning is None:
                plan["reasons"].append("The search could not finish inside the time limit, so the default settings are kept.")

        adopted = bool(tuning and tuning["adopted"])
        final_model = tuning["model"] if adopted else default_model

        confusion_matrix_path = None
        if task_type == "classification":
            confusion_matrix_path = self._plot_confusion_matrix(final_model, X_test, y_test, label_mapping, run_id)

        default_metrics = stats_table[chosen]
        comparison_table = {"default": default_metrics}
        if tuning:
            comparison_table["tuned"] = tuning["metrics"]

        self._log_to_mlflow(chosen, metrics[chosen], tuning)
        summary = self._summarize(
            task_type, self._target_description(y, task_type), metrics, stats_table,
            recommendation, diagnostics, plan, tuning,
        )

        self.log(
            f"task={task_type}, recommended={chosen}, tuning "
            f"{'run' if tuning else 'not run'}, using {'tuned' if adopted else 'default'} model"
        )

        return {
            "task_type": task_type,
            "model_name": chosen,
            "trained_model": default_model,
            "tuned_model": tuning["model"] if adopted else None,
            "final_model": final_model,
            "final_model_choice": "tuned" if adopted else "default",
            "model_recommendation": recommendation,
            "tuning_plan": plan,
            "tuning_ran": tuning is not None,
            "best_params": tuning["best_params"] if tuning else {},
            "tuning_history": tuning["history"] if tuning else [],
            "tuning_comparison": tuning["comparison"] if tuning else None,
            "param_importance": tuning["param_importance"] if tuning else {},
            "tuning_plot_path": tuning["plot_path"] if tuning else None,
            "model_metrics": metrics,
            "stats_table": stats_table,
            "stats_table_str": self._format_table(stats_table, task_type),
            "comparison_table": comparison_table,
            "comparison_table_str": self._format_table(comparison_table, task_type),
            "confusion_matrix_path": confusion_matrix_path,
            "label_mapping": label_mapping,
            "train_test_split_info": {
                "train_size": len(X_train), "test_size": len(X_test),
                "random_state": 42, "target_column": resolved_target,
            },
            "model_selection_summary": summary,
        }

    # ------------------------------------------------------------------
    # data helpers
    # ------------------------------------------------------------------
    def _split_features_target(self, feature_df: pd.DataFrame, target_column: str):
        """
        The Feature Engineering Agent renames passthrough columns with a
        'remainder__' prefix, so look for the target under either name.
        """
        if target_column in feature_df.columns:
            resolved = target_column
        else:
            candidates = [c for c in feature_df.columns if c.endswith(f"__{target_column}")]
            if not candidates:
                raise KeyError(
                    f"Target column '{target_column}' not found in feature_df. "
                    f"Available columns: {list(feature_df.columns)}"
                )
            resolved = candidates[0]
        return feature_df.drop(columns=[resolved]), feature_df[resolved], resolved

    def _detect_task_type(self, y: pd.Series) -> str:
        if not pd.api.types.is_numeric_dtype(y):
            return "classification"
        if y.nunique() <= 20 and np.allclose(y.dropna() % 1, 0):
            return "classification"
        return "regression"

    def _scoring(self, task_type: str) -> str:
        return "f1_weighted" if task_type == "classification" else "neg_root_mean_squared_error"

    def _target_description(self, y: pd.Series, task_type: str) -> str:
        if task_type == "classification":
            share = y.value_counts(normalize=True).sort_index()
            parts = ", ".join(f"class {k}: {v:.0%}" for k, v in share.items())
            return f"Classification target with {len(share)} classes ({parts})."
        return (
            f"Regression target with range {y.min():,.2f} to {y.max():,.2f}, "
            f"mean {y.mean():,.2f}, standard deviation {y.std():,.2f}."
        )

    # ------------------------------------------------------------------
    # model building
    # ------------------------------------------------------------------
    def _build_model(self, model_name: str, task_type: str, params: Optional[Dict[str, Any]] = None):
        params = dict(params or {})
        clf = task_type == "classification"
        if model_name == "logistic_regression":
            return LogisticRegression(max_iter=2000, **params)
        if model_name == "linear_regression":
            return LinearRegression()
        if model_name == "random_forest":
            cls = RandomForestClassifier if clf else RandomForestRegressor
            return cls(random_state=42, n_jobs=-1, **params)
        if model_name == "xgboost":
            if clf:
                return xgb.XGBClassifier(eval_metric="logloss", verbosity=0, random_state=42, **params)
            return xgb.XGBRegressor(verbosity=0, random_state=42, **params)
        if model_name == "lightgbm":
            # LightGBM ignores 'subsample' unless bagging is switched on.
            if "subsample" in params:
                params["subsample_freq"] = 1
            cls = lgb.LGBMClassifier if clf else lgb.LGBMRegressor
            return cls(verbosity=-1, random_state=42, **params)
        raise ValueError(f"Unknown model '{model_name}'")

    # ------------------------------------------------------------------
    # step 1: compare the candidates
    # ------------------------------------------------------------------
    def _cross_validate_all(self, names, X, y, task_type, scoring) -> Dict[str, Dict[str, Any]]:
        results = {}
        for name in names:
            model = self._build_model(name, task_type, self.DEFAULT_PARAMS.get(name))
            scores = cross_val_score(model, X, y, cv=self.SELECTION_CV_FOLDS, scoring=scoring)
            results[name] = {
                "scoring_metric": scoring,
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "fold_scores": [round(float(s), 4) for s in scores],
            }
        return results

    def _fit_and_score_all(self, names, X_train, y_train, X_test, y_test, task_type, n_classes, metrics):
        """
        Fit every candidate once on the training split. The fitted models are
        reused, so nothing is trained twice. Records the full test metrics,
        the test score on the same scale as cross validation, and a train
        score used later to measure overfitting.
        """
        fitted, table, train_scores = {}, {}, {}
        for name in names:
            model = self._build_model(name, task_type, self.DEFAULT_PARAMS.get(name))
            model.fit(X_train, y_train)
            fitted[name] = model
            table[name] = self._full_metrics(model, X_test, y_test, task_type, n_classes)
            metrics[name]["test_score"] = self._cv_scale_score(model, X_test, y_test, task_type)
            train_scores[name] = self._gap_score(model, X_train, y_train, task_type)
        return fitted, table, train_scores

    def _recommend_model(self, metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        One standard error rule: among the models whose score is within one
        standard error of the best, recommend the simplest. A complex model
        has to clearly beat a simple one to earn its place.
        """
        best_name = max(metrics, key=lambda n: metrics[n]["mean"])
        best = metrics[best_name]
        se = best["std"] / np.sqrt(self.SELECTION_CV_FOLDS)
        band = [n for n, v in metrics.items() if v["mean"] >= best["mean"] - se - 1e-12]
        chosen = min(band, key=lambda n: (self.COMPLEXITY[n], -metrics[n]["mean"]))

        reasons = [f"{best_name} had the highest cross-validated score ({best['mean']:.4f})."]
        if len(band) > 1:
            reasons.append(
                f"{len(band)} models ({', '.join(band)}) scored within one standard error "
                f"({se:.4f}) of it, so the data cannot reliably tell them apart."
            )
            if chosen != best_name:
                reasons.append(
                    f"{chosen} is the simplest of them, so it is recommended. A simpler model is easier "
                    f"to explain and less likely to overfit, and nothing shows the complex model is better."
                )
            else:
                reasons.append(f"{chosen} is both the top scorer and the simplest of that group.")
        else:
            reasons.append(
                f"No other model came within one standard error ({se:.4f}), so {best_name} is a clear winner."
            )

        family_note = None
        linear = [n for n in metrics if n in self.LINEAR_MODELS]
        trees = [n for n in metrics if n not in self.LINEAR_MODELS]
        if linear and trees:
            best_linear = max(metrics[n]["mean"] for n in linear)
            best_tree = max(metrics[n]["mean"] for n in trees)
            if best_linear >= best["mean"] - se - 1e-12:
                family_note = "linear_ok"
            elif best_tree - best_linear > se:
                family_note = "trees_win"

        return {
            "model": chosen,
            "best_by_score": best_name,
            "within_noise": band,
            "std_error": float(se),
            "family": "linear" if chosen in self.LINEAR_MODELS else "tree",
            "family_note": family_note,
            "reasons": reasons,
        }

    def _add_signal_reasons(self, rec: Dict[str, Any], diag: Dict[str, Any]) -> None:
        """
        Whether a model family "wins" only means something if the model found
        real signal. On a model that is at chance, score differences are noise.
        """
        if diag["near_chance"]:
            rec["reasons"].append(
                "Note: the best model is barely better than guessing, so this names the least bad option, "
                "not a good one. Differences between the models here are mostly noise."
            )
            return
        if diag["too_perfect"]:
            return
        if diag["weak_but_ranks"]:
            rec["reasons"].append(
                "The signal is real but weak (R2 is low, though predictions still rank rows sensibly), so "
                "treat the differences between models with caution."
            )
            return
        if rec["family_note"] == "linear_ok":
            rec["reasons"].append(
                "A plain linear model is within the noise of the best model, which suggests the "
                "relationships in this data are mostly simple and additive."
            )
        elif rec["family_note"] == "trees_win":
            rec["reasons"].append(
                "Tree models beat the linear model by more than the noise, which suggests non-linear "
                "patterns or interactions between features."
            )

    # ------------------------------------------------------------------
    # metrics
    # ------------------------------------------------------------------
    def _cv_scale_score(self, model, X, y, task_type: str) -> float:
        preds = model.predict(X)
        if task_type == "classification":
            return float(f1_score(y, preds, average="weighted"))
        return float(-np.sqrt(mean_squared_error(y, preds)))

    def _gap_score(self, model, X, y, task_type: str) -> float:
        """F1 for classification, R2 for regression, used to compare train with test."""
        preds = model.predict(X)
        if task_type == "classification":
            return float(f1_score(y, preds, average="weighted"))
        return float(r2_score(y, preds))

    def _full_metrics(self, model, X_test, y_test, task_type: str, n_classes) -> Dict[str, float]:
        preds = model.predict(X_test)
        if task_type == "regression":
            y_arr = np.asarray(y_test, dtype=float)
            p_arr = np.asarray(preds, dtype=float)
            nonzero = y_arr != 0
            mape = (
                float(np.mean(np.abs((y_arr[nonzero] - p_arr[nonzero]) / y_arr[nonzero])) * 100)
                if nonzero.any() else float("nan")
            )
            return {
                "MAE": float(mean_absolute_error(y_test, preds)),
                "MSE": float(mean_squared_error(y_test, preds)),
                "R2": float(r2_score(y_test, preds)),
                "MAPE": mape,
            }

        auc = float("nan")
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                if n_classes == 2:
                    auc = float(roc_auc_score(y_test, proba[:, 1]))
                else:
                    auc = float(roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"))
        except Exception:
            pass
        return {
            "Accuracy": float(accuracy_score(y_test, preds)),
            "Precision": float(precision_score(y_test, preds, average="weighted", zero_division=0)),
            "Recall": float(recall_score(y_test, preds, average="weighted", zero_division=0)),
            "F1": float(f1_score(y_test, preds, average="weighted", zero_division=0)),
            "AUC_ROC": auc,
        }

    def _format_table(self, table: Dict[str, Dict[str, float]], task_type: str) -> str:
        df = pd.DataFrame(table).T
        if task_type == "regression":
            df = df[["MAE", "MSE", "R2", "MAPE"]]
            df.columns = ["MAE", "MSE", "R2 Score", "MAPE (%)"]
        else:
            df = df[["Accuracy", "Precision", "Recall", "F1", "AUC_ROC"]]
            df.columns = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
        return df.round(4).to_string()

    # ------------------------------------------------------------------
    # step 2: is tuning required?
    # ------------------------------------------------------------------
    def _diagnostics(self, model, X_test, y_test, test_metrics, train_score, task_type, n_train) -> Dict[str, Any]:
        """
        Facts that decide whether tuning is worth doing. Measures how much
        real signal the model found, whether the score is too good to be true,
        and how much it overfits.
        """
        rank_corr = None
        if task_type == "classification":
            metric, value = "AUC-ROC", test_metrics.get("AUC_ROC")
            test_score = test_metrics["F1"]
            value = None if value is None or np.isnan(value) else float(value)
            near_chance = value is not None and value < self.NEAR_CHANCE["classification"]
            weak_but_ranks = False
        else:
            metric, value = "R2", float(test_metrics["R2"])
            test_score = value
            # R2 is easily wrecked by a few extreme rows, so also check how
            # well predictions rank the rows. Pure noise scores near zero.
            rc = spearmanr(model.predict(X_test), y_test)[0]
            rank_corr = None if np.isnan(rc) else float(rc)
            rank_signal = rank_corr is not None and rank_corr >= self.RANK_SIGNAL
            low_r2 = value < self.NEAR_CHANCE["regression"]
            near_chance = low_r2 and not rank_signal
            weak_but_ranks = low_r2 and rank_signal

        too_perfect = value is not None and value > self.PERFECT

        if value is None:
            note = "Model quality could not be measured, so treat the conclusions with caution."
        elif near_chance:
            note = f"{metric} is {value:.2f}, barely better than guessing, so the features carry almost no real signal."
        elif weak_but_ranks:
            note = (
                f"R2 is only {value:.2f}, often because a few extreme rows dominate the error, but "
                f"predictions still rank rows sensibly (rank correlation {rank_corr:.2f}). The signal is real but weak."
            )
        elif too_perfect:
            note = f"{metric} is {value:.2f}, close to perfect. That is rare on real data and often means a feature leaks the answer."
        else:
            note = f"{metric} is {value:.2f}, so the model has found real signal."

        return {
            "metric": metric, "value": value, "rank_correlation": rank_corr,
            "near_chance": near_chance, "weak_but_ranks": weak_but_ranks, "too_perfect": too_perfect,
            "train_score": float(train_score), "test_score": float(test_score),
            "train_test_gap": float(train_score - test_score),
            "n_train": int(n_train), "reliability_note": note,
        }

    def _search_space(self, model_name: str, focus: str) -> Dict[str, tuple]:
        """
        What to search, as data, so the plan and the search always agree.
        'regularize' narrows the ranges toward simpler, more cautious models.
        """
        reg = focus == "regularize"
        if model_name == "logistic_regression":
            return {"C": ("float_log", 1e-3, 1.0 if reg else 1e2)}
        if model_name == "random_forest":
            return {
                "n_estimators": ("int", 50, 300),
                "max_depth": ("int", 3, 12 if reg else 30),
                "min_samples_split": ("int", 2, 20),
                "min_samples_leaf": ("int", 3 if reg else 1, 20 if reg else 10),
            }
        if model_name == "xgboost":
            return {
                "n_estimators": ("int", 50, 400),
                "max_depth": ("int", 3, 6 if reg else 10),
                "learning_rate": ("float_log", 0.01, 0.3),
                "subsample": ("float", 0.5, 0.9 if reg else 1.0),
                "colsample_bytree": ("float", 0.5, 0.9 if reg else 1.0),
            }
        if model_name == "lightgbm":
            return {
                "n_estimators": ("int", 50, 400),
                "num_leaves": ("int", 15, 63 if reg else 127),
                "min_child_samples": ("int", 30 if reg else 10, 100 if reg else 50),
                "learning_rate": ("float_log", 0.01, 0.3),
                "subsample": ("float", 0.5, 0.9 if reg else 1.0),
            }
        raise ValueError(f"No search space defined for '{model_name}'")

    def _suggest_params(self, trial, model_name: str, focus: str) -> Dict[str, Any]:
        params = {}
        for name, (kind, lo, hi) in self._search_space(model_name, focus).items():
            if kind == "int":
                params[name] = trial.suggest_int(name, lo, hi)
            elif kind == "float_log":
                params[name] = trial.suggest_float(name, lo, hi, log=True)
            else:
                params[name] = trial.suggest_float(name, lo, hi)
        return params

    def _plan_tuning(self, model_name: str, task_type: str, diag: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide, by rules, whether tuning is required, and if so which settings
        to search and why. Every reason is a measured fact, not an opinion.
        """
        skip: List[str] = []
        if model_name == "linear_regression":
            skip.append("Plain linear regression has no settings to tune, so there is nothing to search.")
        if diag["near_chance"]:
            skip.append(
                f"The model barely beats guessing ({diag['metric']} {diag['value']:.2f}). Tuning can only "
                f"squeeze more out of signal that exists, and these features carry almost none. Better "
                f"features would help, tuning would not."
            )
        if diag["too_perfect"]:
            skip.append(
                f"The score is already near perfect ({diag['metric']} {diag['value']:.2f}), which is rare on "
                f"real data and usually means a feature leaks the answer. There is no headroom to tune into, "
                f"and the leak should be found and removed first."
            )
        if diag["n_train"] < self.MIN_TRAIN_ROWS_FOR_TUNING:
            skip.append(
                f"There are only {diag['n_train']} training rows. With so little data a search mostly fits "
                f"noise in the validation folds instead of finding better settings."
            )
        if skip:
            return {"required": False, "reasons": skip, "focus": None, "params": []}

        reasons: List[str] = []
        focus = "balanced"
        gap, threshold = diag["train_test_gap"], self.OVERFIT_GAP[task_type]
        if model_name == "random_forest":
            # A forest scores near 100 percent on its own training rows by design,
            # so the train versus test gap says nothing about overfitting here.
            reasons.append(
                "The train versus test gap is not used as a warning sign for a random forest, because forests "
                "always score near-perfectly on their own training rows. The search covers both shallow and "
                "deep trees and lets the validation scores decide."
            )
        elif gap >= threshold:
            focus = "regularize"
            reasons.append(
                f"The model scores {diag['train_score']:.2f} on the data it trained on but only "
                f"{diag['test_score']:.2f} on unseen data, a gap of {gap:.2f}. That is overfitting, so the "
                f"search concentrates on settings that hold the model back from fitting noise."
            )
        else:
            gap_text = f"{gap:.2f}" if abs(gap) >= 0.005 else "0.00"
            reasons.append(
                f"Training and test scores are close (gap {gap_text}), so the model is not badly overfitting. "
                f"The search explores both simpler and more flexible settings."
            )

        if model_name == "logistic_regression":
            reasons.append("Logistic regression has one main setting, C, which controls how strongly it is held back from fitting noise.")
        elif model_name == "random_forest":
            reasons.append(
                "Random forests grow full-depth trees by default, so tree depth and leaf size are the "
                "settings most likely to matter. More trees rarely help beyond a few hundred."
            )
        else:
            reasons.append(
                "Boosted models are sensitive to the learning rate together with the number of trees, and to "
                "how large each tree is, so those are searched together."
            )
        if diag["weak_but_ranks"]:
            reasons.append("The signal is real but weak, so expect only a small gain at best.")

        params = [
            {
                "name": name,
                "meaning": self.PARAM_MEANING[name],
                "range": f"{lo:g} to {hi:g}",
            }
            for name, (kind, lo, hi) in self._search_space(model_name, focus).items()
        ]
        return {"required": True, "reasons": reasons, "focus": focus, "params": params}

    # ------------------------------------------------------------------
    # step 2b: run the search and compare with the default
    # ------------------------------------------------------------------
    def _run_tuning(self, model_name, task_type, plan, X_train, y_train, X_test, y_test,
                    scoring, n_classes, n_trials, timeout, run_id) -> Optional[Dict[str, Any]]:
        default_params = self.DEFAULT_PARAMS.get(model_name, {})
        baseline = self._build_model(model_name, task_type, default_params)
        baseline_cv = float(cross_val_score(baseline, X_train, y_train, cv=self.TUNING_CV_FOLDS, scoring=scoring).mean())
        baseline.fit(X_train, y_train)
        baseline_test = self._cv_scale_score(baseline, X_test, y_test, task_type)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

        def objective(trial):
            params = self._suggest_params(trial, model_name, plan["focus"])
            model = self._build_model(model_name, task_type, params)
            return float(cross_val_score(model, X_train, y_train, cv=self.TUNING_CV_FOLDS, scoring=scoring).mean())

        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        history = [
            {"trial": t.number, "score": float(t.value), "params": t.params}
            for t in study.trials if t.value is not None
        ]
        if not history:
            return None

        best_params = study.best_params
        tuned_cv = float(study.best_value)
        tuned = self._build_model(model_name, task_type, best_params)
        tuned.fit(X_train, y_train)
        tuned_test = self._cv_scale_score(tuned, X_test, y_test, task_type)

        comparison = self._compare(baseline_cv, tuned_cv, baseline_test, tuned_test, scoring)
        scores = [h["score"] for h in history]
        return {
            "model": tuned,
            "adopted": comparison["verdict"] == "meaningful improvement",
            "best_params": best_params,
            "history": history,
            "comparison": comparison,
            "metrics": self._full_metrics(tuned, X_test, y_test, task_type, n_classes),
            "param_importance": self._param_importance(study),
            "trial_stats": {
                "best": float(max(scores)), "median": float(np.median(scores)), "worst": float(min(scores)),
            },
            "n_trials": len(history),
            "plot_path": self._plot_history(history, baseline_cv, run_id),
        }

    def _compare(self, baseline_cv, tuned_cv, baseline_test, tuned_test, scoring) -> Dict[str, Any]:
        """
        The verdict is decided by rules, not the LLM. The test-set change is
        what counts, because Optuna already optimized the cross-validated
        score, so a gain there alone proves little. Tuned settings are only
        adopted when the test score improves by at least 1 percent.
        """
        cv_gain = tuned_cv - baseline_cv
        test_gain = tuned_test - baseline_test
        relative = (test_gain / abs(baseline_test) * 100) if baseline_test != 0 else 0.0

        if relative >= 1.0:
            verdict = "meaningful improvement"
        elif relative <= -1.0:
            verdict = "tuned model did worse on the test set"
        elif test_gain > 0:
            verdict = "marginal improvement, within normal noise"
        elif test_gain < 0:
            verdict = "marginal decline, within normal noise"
        else:
            verdict = "no change"

        return {
            "metric": scoring,
            "baseline_cv": float(baseline_cv), "tuned_cv": float(tuned_cv),
            "baseline_test": float(baseline_test), "tuned_test": float(tuned_test),
            "cv_gain": float(cv_gain), "test_gain": float(test_gain),
            "relative_test_gain_pct": float(relative), "verdict": verdict,
            "cv_test_mismatch": bool(cv_gain > 0 and test_gain <= 0),
        }

    def _param_importance(self, study) -> Dict[str, float]:
        try:
            importance = optuna.importance.get_param_importances(study)
            return {k: round(float(v), 3) for k, v in importance.items()}
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # plots
    # ------------------------------------------------------------------
    def _plot_confusion_matrix(self, model, X_test, y_test, label_mapping, run_id: str) -> str:
        cm = confusion_matrix(y_test, model.predict(X_test))
        if label_mapping:
            labels = [label_mapping[i] for i in sorted(label_mapping.keys())]
        else:
            labels = sorted(y_test.unique())

        out_dir = os.path.join(settings.REPORTS_DIR, run_id, "model_selection")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "confusion_matrix.png")

        fig = plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix (final model, test set)")
        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)
        return path

    def _plot_history(self, history, baseline_cv: float, run_id: str) -> str:
        out_dir = os.path.join(settings.REPORTS_DIR, run_id, "tuning")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "optimization_history.png")

        trials = [h["trial"] for h in history]
        scores = [h["score"] for h in history]
        fig = plt.figure(figsize=(8, 5))
        plt.scatter(trials, scores, color="steelblue", label="Trial score")
        plt.plot(trials, np.maximum.accumulate(scores), color="darkorange", linewidth=2, label="Best so far")
        plt.axhline(baseline_cv, color="gray", linestyle="--", label="Default settings (CV)")
        plt.xlabel("Trial")
        plt.ylabel("CV score (higher is better)")
        plt.title("Optuna Optimization History")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # tracking and summary
    # ------------------------------------------------------------------
    def _log_to_mlflow(self, model_name: str, model_metrics: Dict[str, Any], tuning: Optional[Dict[str, Any]]) -> None:
        try:
            from ml.mlflow_tracking import log_run
            log_run(
                run_name=f"model_selection_{model_name}",
                params={"model_name": model_name},
                metrics={
                    "cv_mean": model_metrics["mean"], "cv_std": model_metrics["std"],
                    "test_score": model_metrics.get("test_score", 0.0),
                },
            )
            if tuning:
                c = tuning["comparison"]
                log_run(
                    run_name=f"tuning_{model_name}",
                    params={"model_name": model_name, **{f"best_{k}": v for k, v in tuning["best_params"].items()}},
                    metrics={
                        "baseline_cv": c["baseline_cv"], "tuned_cv": c["tuned_cv"],
                        "baseline_test": c["baseline_test"], "tuned_test": c["tuned_test"],
                    },
                )
        except Exception as e:
            self.log(f"MLflow logging skipped ({e})")

    def _summarize(self, task_type, target_description, metrics, stats_table,
                   recommendation, diagnostics, plan, tuning) -> str:
        try:
            import ollama
            prompt = build_model_selection_prompt(
                task_type, target_description, metrics, stats_table,
                recommendation, diagnostics, plan, tuning,
            )
            response = ollama.chat(
                model=settings.LOCAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                think=False,
                options={"num_predict": 1200, "num_ctx": 4096},
            )
            text = response["message"]["content"].strip()
            if not text:
                raise ValueError("the model returned an empty answer")
            return text
        except Exception as e:
            self.log(f"Ollama call failed ({e}), falling back to template")
            return self._template_summary(recommendation, plan, tuning)

    def _template_summary(self, recommendation, plan, tuning) -> str:
        lines = [f"Recommended model: {recommendation['model']}."] + recommendation["reasons"]
        if plan["required"]:
            lines.append("Fine-tuning is required.")
        else:
            lines.append("Fine-tuning is not required.")
        lines += plan["reasons"]
        if tuning:
            c = tuning["comparison"]
            lines.append(
                f"Tuning result: test score {c['baseline_test']:.4f} to {c['tuned_test']:.4f} "
                f"({c['relative_test_gain_pct']:+.2f} percent), {c['verdict']}. "
                f"The {'tuned' if tuning['adopted'] else 'default'} model is used."
            )
        return " ".join(lines)


def _clean_feature_names(names):
    """
    LightGBM and XGBoost reject feature names containing characters such as
    , : [ ] { } " < >. One-hot columns built from text values hit this.
    """
    cleaned, seen = [], {}
    for name in names:
        c = re.sub(r"[^\w \-\.\(\)%/&'+]", "_", str(name))
        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 1
        cleaned.append(c)
    return cleaned


def _quick_prepare(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Standalone convenience only. Drops ID-like and free-text columns,
    one-hot encodes small categorical columns, and keeps numeric ones,
    so this file can be tested without running the full pipeline.
    """
    y = df[target]
    features = df.drop(columns=[target])
    keep = []
    for col in features.columns:
        series = features[col]
        n_unique = series.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(series)
        # text that is unique per row (IDs, free text) is useless as a feature
        if not is_numeric and n_unique > 0.9 * len(features):
            continue
        # integer column with one distinct value per row is a row ID
        if pd.api.types.is_integer_dtype(series) and n_unique == len(features):
            continue
        # text with too many categories would explode into hundreds of columns
        if not is_numeric and n_unique > 20:
            continue
        keep.append(col)
    if not keep:
        raise ValueError("No usable feature columns left after dropping IDs and free text.")
    prepared = pd.get_dummies(features[keep], drop_first=True, dtype=float)
    prepared.columns = _clean_feature_names(prepared.columns)
    prepared[target] = y.values
    return prepared.dropna()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m agents.model_selection_agent path/to/dataset.csv target_column")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    target = sys.argv[2]
    prepared = _quick_prepare(df, target)

    agent = ModelSelectionAgent()
    result = agent.run({"feature_df": prepared, "target_column": target, "run_id": "manual_test"})

    rec, plan = result["model_recommendation"], result["tuning_plan"]
    print("\n--- Model Comparison (test set) ---")
    print(result["stats_table_str"])
    print(f"\n--- Recommended model: {rec['model']} ---")
    for r in rec["reasons"]:
        print(f"  - {r}")
    print(f"\n--- Fine-tuning required: {'YES' if plan['required'] else 'NO'} ---")
    for r in plan["reasons"]:
        print(f"  - {r}")
    if result["tuning_ran"]:
        print("\n--- Default vs Tuned (test set) ---")
        print(result["comparison_table_str"])
        print(f"Best settings: {result['best_params']}")
        print(f"Verdict: {result['tuning_comparison']['verdict']} | using the {result['final_model_choice']} model")
    print("\n--- Summary ---")
    print(result["model_selection_summary"])
    if result["confusion_matrix_path"]:
        print(f"\nConfusion matrix saved to: {result['confusion_matrix_path']}")
    if result["tuning_plot_path"]:
        print(f"Optimization plot saved to: {result['tuning_plot_path']}")
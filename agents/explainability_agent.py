"""
Explainability Agent

Job: explain why the trained model makes the predictions it does. It
runs SHAP on a sample of the held-out test set, ranks the features by
how much they matter and in which direction they push predictions,
saves the plots, explains the single worst prediction in detail (with
LIME as a second opinion), and checks whether the model is even good
enough for those explanations to mean anything.

Inputs (from state):
    - feature_df: pandas DataFrame, output of the Feature Engineering Agent
    - target_column: str, name of the original target column
    - task_type: "classification" or "regression"
    - tuned_model (preferred) or trained_model: the fitted model to explain
    - model_name: str, optional, used for display
    - label_mapping: dict or None, from the Model Selection Agent
    - run_id: str, used to namespace where plots get saved

Outputs (added to state):
    - shap_values: 2D numpy array (rows x features) for the explained class
    - shap_sample: DataFrame of the rows that were explained
    - feature_importance: list of dicts, ranked, with share and direction
    - shap_plots: list of file paths (summary, bar, dependence, waterfall)
    - local_example: dict, the worst prediction and what drove it
    - explainability_flags: dict, checks the Critic Agent can read
    - focus_class: str or None, the class SHAP values refer to
    - explainability_summary: str, LLM-written explanation
"""
import os
import re
import time
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb

from agents.base_agent import BaseAgent
from prompts.explainability_prompt import build_explainability_prompt
from config.settings import settings

TREE_MODELS = (
    RandomForestClassifier, RandomForestRegressor,
    xgb.XGBClassifier, xgb.XGBRegressor,
    lgb.LGBMClassifier, lgb.LGBMRegressor,
)
LINEAR_MODELS = (LogisticRegression, LinearRegression)


class ExplainabilityAgent(BaseAgent):
    name = "explainability_agent"

    MAX_EXPLAIN_ROWS = 500   # SHAP on 500 rows is plenty and stays fast
    BACKGROUND_ROWS = 200
    TOP_N_PLOT = 15
    TOP_N_TABLE = 8
    SHAP_TIME_BUDGET_SECONDS = 90  # exact tree SHAP can be very slow on big forests

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        feature_df: pd.DataFrame = state["feature_df"]
        target_column: str = state["target_column"]
        task_type: str = state["task_type"]
        label_mapping = state.get("label_mapping")
        run_id = state.get("run_id", "default_run")

        model = state.get("tuned_model") or state.get("trained_model")
        if model is None:
            raise KeyError("Need 'tuned_model' or 'trained_model' in state to explain.")
        model_name = state.get("model_name", type(model).__name__)

        X_train, X_test, y_train, y_test = self._prepare_split(
            feature_df, target_column, task_type, label_mapping
        )
        raw_cols = list(X_train.columns)
        names = self._pretty_names(raw_cols)

        # SHAP on a sample of the test set, never the training data, so the
        # explanation describes behaviour on rows the model has not seen.
        sample_idx = X_test.sample(min(self.MAX_EXPLAIN_ROWS, len(X_test)), random_state=42).index
        X_explain = X_test.loc[sample_idx]
        y_explain = y_test.loc[sample_idx]

        budget = state.get("shap_time_budget_seconds", self.SHAP_TIME_BUDGET_SECONDS)
        rows_requested = len(X_explain)
        shap_out = self._compute_shap(model, X_train, X_explain, task_type, budget)
        X_explain = shap_out["X_used"]          # may be fewer rows if the model is large
        y_explain = y_explain.loc[X_explain.index]
        focus_values = shap_out["focus_values"]
        focus_index = shap_out["focus_index"]
        focus_label = self._decode(focus_index, label_mapping) if focus_index is not None else None

        importance_rows = self._rank_features(
            shap_out["importance"], focus_values, X_explain, raw_cols, names
        )
        top1_share = importance_rows[0]["share_pct"] if importance_rows else 0.0
        top3_share = float(sum(r["share_pct"] for r in importance_rows[:3]))

        performance = self._performance_check(model, X_test, y_test, task_type)
        flags = self._build_flags(performance, top1_share, task_type)
        flags["rows_explained"] = len(X_explain)
        flags["rows_requested"] = rows_requested
        if len(X_explain) < rows_requested:
            flags["reliability_note"] += (
                f" Only {len(X_explain)} of {rows_requested} rows were explained because this model "
                f"is large and exact SHAP is slow, so treat the rankings as rough."
            )

        local = self._local_example(
            model, X_explain, y_explain, focus_values, shap_out["base_value"],
            task_type, focus_index, label_mapping, raw_cols, names, X_train,
        )

        X_plot = X_explain.rename(columns=names)
        plots = self._make_plots(focus_values, X_plot, importance_rows, local, shap_out["base_value"], run_id)

        target_phrase = (
            "the predicted value"
            if task_type == "regression"
            else f"the model's probability of the class '{focus_label}'"
        )
        summary = self._summarize(
            task_type, model_name, target_phrase, performance, flags["reliability_note"],
            importance_rows[: self.TOP_N_TABLE], top1_share, top3_share, local, len(X_explain),
        )

        self.log(
            f"explained {len(X_explain)} rows, top feature "
            f"'{importance_rows[0]['feature']}' holds {top1_share:.1f}% of importance, "
            f"{len(plots)} plots saved"
        )

        return {
            "shap_values": focus_values,
            "shap_sample": X_explain,
            "feature_importance": importance_rows,
            "shap_plots": plots,
            "local_example": local,
            "explainability_flags": flags,
            "focus_class": focus_label,
            "explainability_summary": summary,
        }

    # ------------------------------------------------------------------
    # data helpers
    # ------------------------------------------------------------------
    def _prepare_split(self, feature_df, target_column, task_type, label_mapping):
        """
        Same split settings as the Model Selection and Tuning agents, so the
        test set here is the same held-out data those agents used.
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

        X = feature_df.drop(columns=[resolved])
        y = self._encode_target(feature_df[resolved], task_type, label_mapping)
        return train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if task_type == "classification" else None,
        )

    def _encode_target(self, y: pd.Series, task_type: str, label_mapping):
        if task_type != "classification" or pd.api.types.is_numeric_dtype(y):
            return y
        if label_mapping:
            reverse = {label: int(code) for code, label in label_mapping.items()}
            return y.astype(str).map(reverse).astype(int)
        encoder = LabelEncoder()
        return pd.Series(encoder.fit_transform(y), index=y.index, name=y.name)

    def _decode(self, code, label_mapping) -> str:
        if code is None:
            return "n/a"
        code = int(code)
        if label_mapping:
            return str(label_mapping.get(code, label_mapping.get(str(code), code)))
        return str(code)

    def _pretty_names(self, cols: List[str]) -> Dict[str, str]:
        """
        Feature Engineering renames columns like 'num__Sales' or
        'cat_onehot__Region_East'. Strip those prefixes for readable plots
        and text. Falls back to the raw names if stripping causes a clash.
        """
        pretty = {}
        for col in cols:
            if col.startswith("cat_freq__"):
                pretty[col] = col.split("__", 1)[1] + " (frequency)"
            elif "__" in col:
                pretty[col] = col.split("__", 1)[1]
            else:
                pretty[col] = col
        if len(set(pretty.values())) != len(pretty):
            return {c: c for c in cols}
        return pretty

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------
    def _compute_shap(self, model, X_train, X_explain, task_type: str, budget: float) -> Dict[str, Any]:
        background = X_train.sample(min(self.BACKGROUND_ROWS, len(X_train)), random_state=42)

        if isinstance(model, TREE_MODELS):
            explainer = shap.TreeExplainer(model)
            raw, X_explain = self._tree_shap_with_budget(explainer, X_explain, budget)
        elif isinstance(model, LINEAR_MODELS):
            explainer = shap.LinearExplainer(model, background)
            raw = explainer.shap_values(X_explain)
        else:
            # Slow generic fallback for any other model type.
            predict = model.predict_proba if hasattr(model, "predict_proba") else model.predict
            explainer = shap.KernelExplainer(predict, shap.kmeans(background, 15))
            X_explain = X_explain.iloc[:100]
            raw = explainer.shap_values(X_explain, nsamples=100)

        # Different SHAP and model versions return a list of per-class arrays
        # or a 3D array. Normalize to (rows, features) or (rows, features, classes).
        values = np.stack(raw, axis=-1) if isinstance(raw, list) else np.asarray(raw)

        expected = np.asarray(explainer.expected_value, dtype=float).ravel()

        if values.ndim == 3:
            n_classes = values.shape[2]
            importance = np.abs(values).mean(axis=(0, 2))
            if n_classes == 2:
                focus = 1
            else:
                preds = model.predict(X_explain)
                focus = int(pd.Series(preds).mode().iloc[0])
            focus_values = values[:, :, focus]
            base = float(expected[focus]) if len(expected) > focus else float(expected[0])
        else:
            focus_values = values
            importance = np.abs(values).mean(axis=0)
            base = float(expected[0])
            focus = 1 if task_type == "classification" else None

        return {
            "focus_values": focus_values,
            "importance": importance,
            "base_value": base,
            "focus_index": focus,
            "X_used": X_explain,
        }

    def _tree_shap_with_budget(self, explainer, X_explain, budget: float):
        """
        Exact tree SHAP cost grows with the size of the trees. A default
        random forest fit on tens of thousands of noisy rows can take
        seconds per row. Time the first 10 rows, work out how many rows fit
        in the budget, and explain only that many. Fast models still get
        every row.
        """
        first_n = min(10, len(X_explain))
        start = time.time()
        first = explainer.shap_values(X_explain.iloc[:first_n], check_additivity=False)
        per_row = (time.time() - start) / first_n
        fits = len(X_explain) if per_row <= 0 else int(budget / per_row)
        target = max(first_n, min(len(X_explain), fits))

        if target == first_n:
            return first, X_explain.iloc[:first_n]

        rest = explainer.shap_values(X_explain.iloc[first_n:target], check_additivity=False)
        if isinstance(first, list):
            raw = [np.concatenate([a, b], axis=0) for a, b in zip(first, rest)]
        else:
            raw = np.concatenate([first, rest], axis=0)
        return raw, X_explain.iloc[:target]

    def _rank_features(self, importance, focus_values, X_explain, raw_cols, names) -> List[Dict[str, Any]]:
        total = float(importance.sum()) or 1.0
        order = np.argsort(-importance)
        rows = []
        for rank, i in enumerate(order, start=1):
            feature_vals = X_explain.iloc[:, i].to_numpy(dtype=float)
            direction, corr = self._direction(feature_vals, focus_values[:, i])
            rows.append({
                "rank": rank,
                "feature": names[raw_cols[i]],
                "raw_feature": raw_cols[i],
                "mean_abs_shap": float(importance[i]),
                "share_pct": float(importance[i] / total * 100),
                "direction": direction,
                "correlation": corr,
            })
        return rows

    def _direction(self, feature_vals, shap_vals):
        if np.std(feature_vals) == 0 or np.std(shap_vals) == 0:
            return "no clear direction", 0.0
        corr = float(np.corrcoef(feature_vals, shap_vals)[0, 1])
        if np.isnan(corr):
            return "no clear direction", 0.0
        if corr > 0.2:
            return "higher values push the prediction up", round(corr, 2)
        if corr < -0.2:
            return "higher values push the prediction down", round(corr, 2)
        return "mixed, no consistent direction", round(corr, 2)

    # ------------------------------------------------------------------
    # reliability checks
    # ------------------------------------------------------------------
    def _performance_check(self, model, X_test, y_test, task_type: str) -> Dict[str, Any]:
        """
        Explaining a model that has learned nothing produces confident
        looking rankings of pure noise. Measure the model first.
        """
        try:
            if task_type == "classification":
                proba = model.predict_proba(X_test)
                if proba.shape[1] == 2:
                    value = float(roc_auc_score(y_test, proba[:, 1]))
                else:
                    value = float(roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"))
                return {"metric": "AUC-ROC", "value": value}
            preds = model.predict(X_test)
            # R2 is easily wrecked by a few extreme rows, so also measure how
            # well predictions rank rows (Spearman). Pure noise scores near 0.
            rank_corr = spearmanr(preds, y_test)[0]
            rank_corr = None if np.isnan(rank_corr) else float(rank_corr)
            return {"metric": "R2", "value": float(r2_score(y_test, preds)), "rank_corr": rank_corr}
        except Exception:
            return {"metric": "AUC-ROC" if task_type == "classification" else "R2", "value": None}

    def _build_flags(self, performance: Dict[str, Any], top1_share: float, task_type: str) -> Dict[str, Any]:
        value = performance["value"]
        rank_corr = performance.get("rank_corr")
        rank_signal = rank_corr is not None and rank_corr >= 0.2
        weak_r2_but_ranks = False

        if value is None:
            near_chance, too_perfect = False, False
        elif task_type == "classification":
            near_chance, too_perfect = value < 0.6, value > 0.98
        else:
            low_r2 = value < 0.05
            near_chance = low_r2 and not rank_signal
            weak_r2_but_ranks = low_r2 and rank_signal
            too_perfect = value > 0.98

        dominant = top1_share >= 50.0
        # A dominant feature alone is normal (one strong driver). It only
        # looks like leakage when the model is also suspiciously good.
        dominant_and_strong = dominant and value is not None and value > 0.95
        possible_leakage = too_perfect or dominant_and_strong

        if value is None:
            note = "Model quality could not be measured, so treat the explanations with caution."
        elif near_chance:
            note = (
                f"{performance['metric']} is {value:.2f}, barely better than guessing, so the "
                f"rankings below mostly reflect noise and are not real drivers."
            )
        elif weak_r2_but_ranks:
            note = (
                f"R2 is only {value:.2f}, often because a few extreme rows dominate the error, but "
                f"predictions still rank rows sensibly (rank correlation {rank_corr:.2f}). The "
                f"rankings below are moderately reliable, not strong."
            )
        elif too_perfect:
            note = (
                f"{performance['metric']} is {value:.2f}, close to perfect. That is rare on real "
                f"data and often means a feature leaks the answer."
            )
        else:
            note = f"{performance['metric']} is {value:.2f}, so the model has learned something real and the rankings are meaningful."

        if possible_leakage and dominant:
            note += (
                f" One feature also holds {top1_share:.0f} percent of all importance, so check "
                f"that feature first, it is the most likely leak."
            )
        elif dominant and not near_chance:
            note += (
                f" The model leans heavily on one feature ({top1_share:.0f} percent of importance), "
                f"which is fine if it makes sense for the problem."
            )

        return {
            "performance_metric": performance["metric"],
            "performance_value": value,
            "rank_correlation": rank_corr,
            "near_chance": near_chance,
            "weak_but_ranks_correctly": weak_r2_but_ranks,
            "suspiciously_perfect": too_perfect,
            "possible_leakage": possible_leakage,
            "dominant_feature": dominant,
            "top_feature_share_pct": top1_share,
            "reliability_note": note,
        }

    # ------------------------------------------------------------------
    # one worked example, plus LIME as a second opinion
    # ------------------------------------------------------------------
    def _local_example(self, model, X_explain, y_explain, focus_values, base_value, task_type,
                       focus_index, label_mapping, raw_cols, names, X_train) -> Dict[str, Any]:
        preds = model.predict(X_explain)

        if task_type == "regression":
            errors = np.abs(np.asarray(preds, dtype=float) - np.asarray(y_explain, dtype=float))
            row = int(np.argmax(errors))
            predicted = f"{float(preds[row]):.3f}"
            actual = f"{float(y_explain.iloc[row]):.3f}"
        else:
            true_codes = np.asarray(y_explain).astype(int)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_explain)
                col_for_true = np.searchsorted(model.classes_, true_codes)
                p_true = proba[np.arange(len(true_codes)), col_for_true]
                row = int(np.argmin(p_true))
            else:
                row = int(np.argmax(np.asarray(preds) != true_codes))
            predicted = self._decode(preds[row], label_mapping)
            actual = self._decode(true_codes[row], label_mapping)

        vals = focus_values[row]
        order = np.argsort(-np.abs(vals))[:6]
        contributions = [
            {
                "feature": names[raw_cols[i]],
                "value": float(X_explain.iloc[row, i]),
                "shap": float(vals[i]),
            }
            for i in order
        ]

        lime_top, agreement, lime_ok = self._lime_check(
            model, X_train, X_explain, row, task_type, focus_index, raw_cols, names, vals,
        )

        return {
            "row_position": row,
            "actual": actual,
            "predicted": predicted,
            "base_value": float(base_value),
            "top_contributions": contributions,
            "lime_available": lime_ok,
            "lime_top_features": lime_top,
            "lime_agreement": agreement,
        }

    def _lime_check(self, model, X_train, X_explain, row, task_type, focus_index,
                    raw_cols, names, shap_row_values):
        """
        Second opinion on the same row. LIME fits a small local model around
        the row, so it can disagree with SHAP. Agreement on the top 5
        features raises confidence in the explanation.
        """
        try:
            from lime.lime_tabular import LimeTabularExplainer

            pretty = [names[c] for c in raw_cols]
            explainer = LimeTabularExplainer(
                X_train.to_numpy(dtype=float),
                feature_names=pretty,
                mode="classification" if task_type == "classification" else "regression",
                discretize_continuous=True,
                random_state=42,
            )

            if task_type == "classification":
                def predict_fn(arr):
                    return model.predict_proba(pd.DataFrame(arr, columns=raw_cols))
                labels = (focus_index,)
            else:
                def predict_fn(arr):
                    return model.predict(pd.DataFrame(arr, columns=raw_cols))
                labels = None

            exp = explainer.explain_instance(
                X_explain.iloc[row].to_numpy(dtype=float),
                predict_fn,
                num_features=min(10, len(raw_cols)),
                num_samples=1000,
                **({"labels": labels} if labels else {}),
            )
            key = focus_index if task_type == "classification" else list(exp.local_exp.keys())[0]
            weights = sorted(exp.local_exp[key], key=lambda t: -abs(t[1]))
            lime_top = [pretty[i] for i, _ in weights[:5]]

            shap_top = [pretty[i] for i in np.argsort(-np.abs(shap_row_values))[:5]]
            agreement = len(set(lime_top) & set(shap_top))
            return lime_top, agreement, True
        except Exception as e:
            self.log(f"LIME check skipped ({e})")
            return [], 0, False

    # ------------------------------------------------------------------
    # plots
    # ------------------------------------------------------------------
    def _safe_name(self, text: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")[:40]

    def _make_plots(self, focus_values, X_plot, importance_rows, local, base_value, run_id) -> List[str]:
        out_dir = os.path.join(settings.REPORTS_DIR, run_id, "explainability")
        os.makedirs(out_dir, exist_ok=True)
        paths: List[str] = []

        def save(fig_title: str, filename: str):
            path = os.path.join(out_dir, filename)
            plt.gcf().suptitle(fig_title, y=1.04)
            plt.savefig(path, bbox_inches="tight")
            plt.close("all")
            paths.append(path)

        # beeswarm summary: every dot is one row, colour is the feature value
        try:
            shap.summary_plot(focus_values, X_plot, max_display=self.TOP_N_PLOT, show=False)
            save("SHAP Summary (each dot is one row)", "shap_summary.png")
        except Exception as e:
            plt.close("all")
            self.log(f"summary plot skipped ({e})")

        # bar chart of mean absolute SHAP, own drawing for reliable labels
        try:
            top = importance_rows[: self.TOP_N_PLOT][::-1]
            plt.figure(figsize=(8, max(3, 0.4 * len(top) + 1)))
            plt.barh([r["feature"] for r in top], [r["mean_abs_shap"] for r in top], color="steelblue")
            plt.xlabel("Mean absolute SHAP value")
            save("Feature Importance (SHAP)", "shap_importance_bar.png")
        except Exception as e:
            plt.close("all")
            self.log(f"bar plot skipped ({e})")

        # dependence plots for the top two features
        for r in importance_rows[:2]:
            try:
                shap.dependence_plot(r["feature"], focus_values, X_plot, interaction_index=None, show=False)
                save(f"Dependence: {r['feature']}", f"shap_dependence_{self._safe_name(r['feature'])}.png")
            except Exception as e:
                plt.close("all")
                self.log(f"dependence plot for {r['feature']} skipped ({e})")

        # waterfall for the single worst prediction
        try:
            row = local["row_position"]
            explanation = shap.Explanation(
                values=focus_values[row],
                base_values=base_value,
                data=X_plot.iloc[row].to_numpy(dtype=float),
                feature_names=list(X_plot.columns),
            )
            shap.plots.waterfall(explanation, max_display=10, show=False)
            save("Why the model got this row most wrong", "shap_waterfall_worst_case.png")
        except Exception as e:
            plt.close("all")
            self.log(f"waterfall plot skipped ({e})")

        return paths

    # ------------------------------------------------------------------
    # LLM summary
    # ------------------------------------------------------------------
    def _summarize(self, task_type, model_name, target_phrase, performance, reliability_note,
                   importance_rows, top1_share, top3_share, local, n_rows) -> str:
        try:
            import ollama
            prompt = build_explainability_prompt(
                task_type, model_name, target_phrase, performance, reliability_note,
                importance_rows, top1_share, top3_share, local, n_rows,
            )
            response = ollama.chat(
                model=settings.LOCAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                think=False,
                options={"num_predict": 800},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            self.log(f"Ollama call failed ({e}), falling back to template")
            return self._template_summary(model_name, importance_rows, reliability_note)

    def _template_summary(self, model_name, importance_rows, reliability_note) -> str:
        top = ", ".join(f"{r['feature']} ({r['share_pct']:.1f}%)" for r in importance_rows[:3])
        return f"Top features for {model_name}: {top}. {reliability_note}"


if __name__ == "__main__":
    import sys
    from agents.model_selection_agent import ModelSelectionAgent, _quick_prepare

    if len(sys.argv) < 3:
        print("Usage: python -m agents.explainability_agent path/to/dataset.csv target_column [model_name]")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    target = sys.argv[2]
    model_name = sys.argv[3] if len(sys.argv) > 3 else "random_forest"

    prepared = _quick_prepare(df, target)
    y = prepared[target]
    is_clf = (not pd.api.types.is_numeric_dtype(y)) or (y.nunique() <= 20 and np.allclose(y % 1, 0))
    task_type = "classification" if is_clf else "regression"

    label_mapping = None
    if is_clf and not pd.api.types.is_numeric_dtype(y):
        classes = sorted(y.astype(str).unique())
        label_mapping = {i: c for i, c in enumerate(classes)}

    agent = ExplainabilityAgent()
    X_train, X_test, y_train, y_test = agent._prepare_split(prepared, target, task_type, label_mapping)

    tuner = ModelSelectionAgent()
    model = tuner._build_model(model_name, task_type, tuner.DEFAULT_PARAMS.get(model_name, {}))
    model.fit(X_train, y_train)

    result = agent.run({
        "feature_df": prepared, "target_column": target, "task_type": task_type,
        "trained_model": model, "model_name": model_name,
        "label_mapping": label_mapping, "run_id": "manual_test",
    })

    print("\n--- Feature Ranking ---")
    ranking = pd.DataFrame(result["feature_importance"])[["rank", "feature", "share_pct", "direction"]]
    ranking["share_pct"] = ranking["share_pct"].round(1)
    print(ranking.head(10).to_string(index=False))
    print("\n--- Explainability Summary ---")
    print(result["explainability_summary"])
    print(f"\nChecks: {result['explainability_flags']['reliability_note']}")
    print("Plots saved:")
    for p in result["shap_plots"]:
        print(f"  {p}")
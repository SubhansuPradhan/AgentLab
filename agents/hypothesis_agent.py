"""
Hypothesis Generation Agent

Job: before any model is trained, look at the cleaned data and the
target, and produce a short list of testable hypotheses: which features
should drive the target, which should not matter, which look like data
leaks. After the Explainability Agent has run, verify() compares each
hypothesis with what SHAP actually found.

How it decides: rules, not the LLM. Every feature is tested on its own
against the target with a small cross-validated decision tree. That gives
an honest score (AUC-ROC for classification, R2 for regression), a
direction, and a near-perfect score means a probable leak. The LLM only
writes the plain-language reasoning around numbers it cannot change.

Inputs for run() (from state):
    - cleaned_df: pandas DataFrame, output of the Validation Agent
    - target_column: str, the column to predict
    - task_type: "classification" or "regression", optional (auto-detected)
    - eda_stats: dict from the EDA Agent, optional, used as context

Outputs of run() (added to state):
    - hypotheses: list of dicts, each with claim, evidence and a check
    - feature_screening: list of dicts, every tested feature, best first
    - leakage_suspects: list of column names that predict the target
      almost perfectly on their own
    - excluded_columns: dict of column -> why it was not tested
    - target_info: dict, task type, classes and positive class
    - hypothesis_summary: str, LLM-written plan

Inputs for verify() (from state, after the Explainability Agent):
    - hypotheses, feature_screening, excluded_columns (from run())
    - feature_importance and explainability_flags (Explainability Agent)

Outputs of verify() (added to state):
    - hypothesis_results: list of dicts with a status for each hypothesis
    - hypothesis_scorecard: dict, count per status
    - hypothesis_verification_summary: str, readable comparison for the report

In LangGraph, register run as the "hypothesis" node and verify as a
second node that runs after "explainability".
"""
import re
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from agents.base_agent import BaseAgent
from prompts.hypothesis_prompt import build_hypothesis_prompt
from config.settings import settings

ID_TOKENS = {"id", "uuid", "guid", "zip", "postal", "pincode", "index", "code"}


class HypothesisAgent(BaseAgent):
    name = "hypothesis_agent"

    MAX_DRIVERS = 3
    MAX_NO_EFFECT = 1
    MAX_LEAKS = 3
    MAX_CATEGORIES = 30       # more categories than this cannot be tested reliably
    LEAK_SCORE = 0.98         # a single feature this good is almost surely a leak
    STRONG_SCORE = {"classification": 0.90, "regression": 0.85}  # unusually strong for ONE feature
    MIN_ROWS_PER_FEATURE = 50
    MIN_CLASS_ROWS = 10       # classes smaller than this are left out of the screen

    # used by verify()
    TOP_K = 3                 # a driver should land in the top 3 of the SHAP ranking
    LOW_SHARE = 2.0           # under 2 percent of SHAP importance counts as barely used

    # ------------------------------------------------------------------
    # step 1: generate hypotheses before modeling
    # ------------------------------------------------------------------
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = state["cleaned_df"]
        target: str = state["target_column"]
        if target not in df.columns:
            raise KeyError(f"Target column '{target}' not found. Available columns: {list(df.columns)}")

        df = df.loc[df[target].notna()].copy()
        task_type = state.get("task_type") or self._detect_task_type(df[target])

        # A class with only a handful of rows cannot be tested reliably.
        rare_dropped: List[str] = []
        if task_type == "classification":
            counts = df[target].value_counts()
            rare = counts[counts < self.MIN_CLASS_ROWS].index
            if len(rare):
                rare_dropped = [str(c) for c in rare]
                df = df.loc[~df[target].isin(rare)].copy()

        if df[target].nunique() < 2:
            raise ValueError(
                f"Target '{target}' has fewer than two distinct values after cleaning, "
                f"so there is nothing to predict."
            )
        y, target_info = self._encode_target(df[target], task_type)
        target_info["rare_classes_dropped"] = rare_dropped
        n_rows = len(df)

        screening: List[Dict[str, Any]] = []
        excluded: Dict[str, str] = {}
        for col in df.columns:
            if col == target:
                continue
            reason = self._exclusion_reason(col, df[col], n_rows)
            if reason:
                excluded[col] = reason
                continue
            row = self._evaluate_feature(col, df[col], y, task_type, target_info)
            if row is None:
                excluded[col] = "not enough usable values to test"
                continue
            screening.append(row)
        screening.sort(key=lambda r: -r["solo_score"])

        leaks = [r for r in screening if r["solo_score"] >= self.LEAK_SCORE][: self.MAX_LEAKS]
        leak_names = [r["feature"] for r in leaks]
        candidates = [r for r in screening if r["feature"] not in leak_names]

        signal_min = self._signal_min(task_type, n_rows)
        drivers = [r for r in candidates if r["solo_score"] >= signal_min][: self.MAX_DRIVERS]
        weak = [r for r in candidates if r["solo_score"] < signal_min]
        no_effect = []
        if len(candidates) >= 3 and weak:
            no_effect = sorted(weak, key=lambda r: r["solo_score"])[: self.MAX_NO_EFFECT]

        hypotheses: List[Dict[str, Any]] = []
        for r in leaks:
            hypotheses.append(self._leak_hypothesis(r, target, target_info))
        for r in drivers:
            hypotheses.append(self._driver_hypothesis(r, target, task_type, target_info))
        for r in no_effect:
            hypotheses.append(self._no_effect_hypothesis(r, target, target_info))
        if not drivers:
            hypotheses.append(self._weak_signal_hypothesis(candidates, target, target_info, bool(leaks)))
        for i, h in enumerate(hypotheses, start=1):
            h["id"] = f"H{i}"

        summary = self._summarize(
            target, task_type, target_info, n_rows, hypotheses, screening,
            leak_names, excluded, self._eda_context(state.get("eda_stats")),
        )

        self.log(
            f"tested {len(screening)} features, built {len(hypotheses)} hypotheses, "
            f"{len(leak_names)} suspected leak(s), {len(excluded)} column(s) left out"
        )

        return {
            "hypotheses": hypotheses,
            "feature_screening": screening,
            "leakage_suspects": leak_names,
            "excluded_columns": excluded,
            "target_info": target_info,
            "hypothesis_summary": summary,
        }

    # ------------------------------------------------------------------
    # target handling
    # ------------------------------------------------------------------
    def _detect_task_type(self, y: pd.Series) -> str:
        if not pd.api.types.is_numeric_dtype(y):
            return "classification"
        if y.nunique() <= 20 and np.allclose(y.dropna() % 1, 0):
            return "classification"
        return "regression"

    def _encode_target(self, series: pd.Series, task_type: str):
        """
        Classification classes are sorted, and the positive class of a binary
        target is the last one (the same class 1 the Explainability Agent
        explains), so directions mean the same thing in both agents.
        """
        if task_type == "regression":
            y = series.astype(float)
            return y, {
                "task_type": "regression", "n_classes": None, "classes": None,
                "positive_class": None, "positive_code": None,
                "mean": float(y.mean()), "std": float(y.std()),
            }

        if pd.api.types.is_numeric_dtype(series):
            classes = sorted(series.unique())
            codes = np.searchsorted(classes, series.to_numpy())
            labels = [str(int(c)) if float(c).is_integer() else str(c) for c in classes]
        else:
            labels = sorted(series.astype(str).unique())
            lookup = {label: i for i, label in enumerate(labels)}
            codes = series.astype(str).map(lookup).to_numpy()

        y = pd.Series(codes, index=series.index)
        balance = {labels[i]: float((y == i).mean()) for i in range(len(labels))}
        return y, {
            "task_type": "classification", "n_classes": len(labels), "classes": labels,
            "positive_class": labels[-1], "positive_code": len(labels) - 1,
            "balance": balance,
        }

    # ------------------------------------------------------------------
    # screening: one feature at a time
    # ------------------------------------------------------------------
    def _exclusion_reason(self, col: str, series: pd.Series, n_rows: int) -> Optional[str]:
        tokens = set(re.split(r"[^a-z0-9]+", col.lower()))
        if tokens & ID_TOKENS:
            return "name suggests an identifier"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "date column, not tested"
        n_unique = series.nunique(dropna=True)
        if n_unique <= 1:
            return "constant column"
        is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        if pd.api.types.is_integer_dtype(series) and n_unique == n_rows:
            return "different value on every row (row identifier)"
        if not is_numeric:
            if n_unique > 0.9 * n_rows:
                return "different text on almost every row (identifier or free text)"
            if n_unique > self.MAX_CATEGORIES:
                return f"{n_unique} categories, too many to test reliably"
        return None

    def _evaluate_feature(self, col, series, y, task_type, target_info) -> Optional[Dict[str, Any]]:
        mask = series.notna()
        if mask.sum() < self.MIN_ROWS_PER_FEATURE:
            return None
        x, yy = series[mask], y[mask]
        if x.nunique() < 2:
            return None

        is_numeric = pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x)
        if is_numeric:
            design = x.to_numpy(dtype=float).reshape(-1, 1)
        else:
            design = pd.get_dummies(x.astype(str), dtype=float).to_numpy()

        score = self._solo_score(design, yy.to_numpy(), task_type, target_info["n_classes"])
        if score is None:
            return None

        rho = p_value = None
        groups = None
        if is_numeric and task_type == "regression" or (is_numeric and target_info["n_classes"] == 2):
            rho_raw, p_raw = spearmanr(x, yy)
            if not np.isnan(rho_raw):
                rho, p_value = float(rho_raw), float(p_raw)
        if not is_numeric:
            groups = self._group_stats(x, yy, task_type, target_info)

        return {
            "feature": col,
            "kind": "numeric" if is_numeric else "categorical",
            "metric": "AUC-ROC" if task_type == "classification" else "R2",
            "solo_score": round(score, 4),
            "rho": None if rho is None else round(rho, 3),
            "p_value": p_value,
            "groups": groups,
            "n_used": int(mask.sum()),
        }

    def _solo_score(self, design, y, task_type, n_classes) -> Optional[float]:
        """
        Cross-validated score of a shallow tree that sees only this one
        feature. The tree is small and regularized on purpose: it should pick
        up real patterns (including non-linear ones) but not memorize noise.
        """
        min_leaf = max(5, int(0.01 * len(y)))
        try:
            if task_type == "classification":
                tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=min_leaf, random_state=42)
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scoring = "roc_auc" if n_classes == 2 else "roc_auc_ovr_weighted"
            else:
                tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=min_leaf, random_state=42)
                cv = KFold(n_splits=3, shuffle=True, random_state=42)
                scoring = "r2"
            score = float(np.mean(cross_val_score(tree, design, y, cv=cv, scoring=scoring, error_score="raise")))
            return None if np.isnan(score) else score
        except Exception:
            return None

    def _group_stats(self, x, y, task_type, target_info) -> Optional[Dict[str, Any]]:
        if task_type == "classification" and target_info["n_classes"] != 2:
            return None
        frame = pd.DataFrame({"x": x.astype(str).to_numpy(), "y": y.to_numpy()})
        if task_type == "classification":
            frame["y"] = (frame["y"] == target_info["positive_code"]).astype(float)
        agg = frame.groupby("x")["y"].agg(["mean", "count"])
        agg = agg[agg["count"] >= max(30, int(0.01 * len(frame)))]
        if len(agg) < 2:
            return None
        hi, lo = agg["mean"].idxmax(), agg["mean"].idxmin()
        return {
            "highest": hi, "highest_value": float(agg.loc[hi, "mean"]),
            "lowest": lo, "lowest_value": float(agg.loc[lo, "mean"]),
        }

    def _signal_min(self, task_type: str, n_rows: int) -> float:
        """
        Smallest single-feature score that counts as real signal. Small
        datasets get a higher bar because noise looks like signal more often.
        """
        if task_type == "classification":
            return 0.5 + max(0.06, 3.0 / np.sqrt(n_rows))
        return 0.02

    # ------------------------------------------------------------------
    # turning evidence into hypotheses
    # ------------------------------------------------------------------
    def _target_phrase(self, target: str, info: Dict[str, Any]) -> str:
        if info["task_type"] == "classification" and info["n_classes"] == 2:
            return f"the chance of '{info['positive_class']}' in '{target}'"
        return f"'{target}'"

    def _effect_phrase(self, word: str, target: str, info: Dict[str, Any]) -> str:
        """'higher' or 'lower', worded correctly for each kind of target."""
        if info["task_type"] == "classification" and info["n_classes"] == 2:
            return f"a {word} chance of '{info['positive_class']}' in '{target}'"
        return f"{word} '{target}'"

    def _direction_of(self, rho: Optional[float]) -> str:
        if rho is None:
            return "none"
        if rho >= 0.1:
            return "up"
        if rho <= -0.1:
            return "down"
        return "nonlinear"

    def _driver_hypothesis(self, r, target, task_type, info) -> Dict[str, Any]:
        col, score = r["feature"], r["solo_score"]
        tp = self._target_phrase(target, info)
        alone = (
            f"AUC-ROC on its own is {score:.2f}" if task_type == "classification"
            else f"on its own it explains about {max(score, 0) * 100:.0f} percent of the variation in '{target}'"
        )

        expected_direction, expected_category = "none", None
        if r["kind"] == "numeric":
            expected_direction = self._direction_of(r["rho"])
            if expected_direction in ("up", "down"):
                word = "higher" if expected_direction == "up" else "lower"
                claim = (
                    f"Higher '{col}' goes with {self._effect_phrase(word, target, info)} "
                    f"(rank correlation {r['rho']:+.2f}); {alone}."
                )
            elif expected_direction == "nonlinear":
                claim = (
                    f"'{col}' affects {tp} in a non-linear way, with no simple rise or fall "
                    f"(rank correlation {r['rho']:+.2f}); {alone}."
                )
            else:
                claim = f"'{col}' differs between the classes of '{target}'; {alone}."
        else:
            g = r["groups"]
            if g and task_type == "regression":
                claim = (
                    f"'{col}' matters for '{target}': '{g['highest']}' has the highest average "
                    f"({g['highest_value']:,.2f}) and '{g['lowest']}' the lowest ({g['lowest_value']:,.2f}); {alone}."
                )
                expected_direction, expected_category = "up", g["highest"]
            elif g:
                claim = (
                    f"'{col}' matters for '{target}': '{g['highest']}' has the highest rate of "
                    f"'{info['positive_class']}' ({g['highest_value']:.0%}) and '{g['lowest']}' the lowest "
                    f"({g['lowest_value']:.0%}); {alone}."
                )
                expected_direction, expected_category = "up", g["highest"]
            else:
                claim = f"'{col}' differs between the classes of '{target}'; {alone}."

        unusually_strong = score >= self.STRONG_SCORE[task_type]
        if unusually_strong:
            claim += (
                f" This is unusually strong for a single feature, so check that '{col}' "
                f"is not derived from the target."
            )

        desc = f"After training, SHAP should rank '{col}' in the top {self.TOP_K}"
        if expected_direction in ("up", "down") and expected_category is None:
            desc += f", with higher values pushing predictions {expected_direction}."
        elif expected_direction == "nonlinear":
            desc += ", with a mixed direction rather than a simple rise or fall."
        elif expected_category:
            desc += f", with being in '{expected_category}' pushing predictions up."
        else:
            desc += "."

        return {
            "type": "driver", "feature": col, "claim": claim,
            "evidence": {**self._evidence(r), "unusually_strong": unusually_strong},
            "check": {
                "kind": "driver", "feature": col, "expected_direction": expected_direction,
                "expected_category": expected_category, "description": desc,
            },
        }

    def _no_effect_hypothesis(self, r, target, info) -> Dict[str, Any]:
        col, score = r["feature"], r["solo_score"]
        chance = "0.50" if r["metric"] == "AUC-ROC" else "0"
        claim = (
            f"'{col}' carries no usable signal about '{target}' on its own "
            f"({r['metric']} {score:.2f}, where {chance} means no relationship)."
        )
        return {
            "type": "no_effect", "feature": col, "claim": claim,
            "evidence": self._evidence(r),
            "check": {
                "kind": "no_effect", "feature": col,
                "description": (
                    f"After training, SHAP should give '{col}' under {self.LOW_SHARE:.0f} percent of "
                    f"total importance, or rank it in the bottom half."
                ),
            },
        }

    def _leak_hypothesis(self, r, target, info) -> Dict[str, Any]:
        col = r["feature"]
        claim = (
            f"'{col}' may leak the answer: on its own it predicts '{target}' almost perfectly "
            f"({r['metric']} {r['solo_score']:.2f}). Check whether '{col}' is only known after the "
            f"outcome, and drop it if so."
        )
        return {
            "type": "leakage", "feature": col, "claim": claim,
            "evidence": self._evidence(r),
            "check": {
                "kind": "leakage", "feature": col,
                "description": (
                    f"If '{col}' is kept in the model, SHAP should show it dominating "
                    f"(50 percent or more of importance) alongside a near-perfect score."
                ),
            },
        }

    def _weak_signal_hypothesis(self, candidates, target, info, has_leaks) -> Dict[str, Any]:
        metric = "AUC-ROC" if info["task_type"] == "classification" else "R2"
        best = candidates[0] if candidates else None
        best_text = f"the best is '{best['feature']}' with {metric} {best['solo_score']:.2f}" if best else "none could be tested"
        scope = "Apart from the suspected leaks, no" if has_leaks else "No"
        claim = (
            f"{scope} single feature predicts '{target}' clearly better than chance ({best_text}), so "
            f"expect a model close to chance unless features combine in ways this screen cannot see."
        )
        threshold = "AUC-ROC under 0.6" if info["task_type"] == "classification" else "R2 under 0.05"
        return {
            "type": "weak_signal", "feature": None, "claim": claim,
            "evidence": self._evidence(best) if best else {},
            "check": {
                "kind": "weak_signal", "feature": None,
                "description": f"After training, the model should score close to chance ({threshold}).",
            },
        }

    def _evidence(self, r: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "metric": r["metric"], "solo_score": r["solo_score"],
            "rho": r["rho"], "p_value": r["p_value"], "n_used": r["n_used"],
        }

    # ------------------------------------------------------------------
    # LLM summary
    # ------------------------------------------------------------------
    def _eda_context(self, eda_stats: Optional[Dict[str, Any]]) -> str:
        if not eda_stats:
            return "not available"
        parts = []
        pairs = eda_stats.get("high_correlations") or []
        if pairs:
            parts.append(
                "highly correlated pairs: "
                + ", ".join(f"{p['columns'][0]} and {p['columns'][1]} ({p['correlation']:+.2f})" for p in pairs[:5])
            )
        skewed = eda_stats.get("skewed_columns") or []
        if skewed:
            parts.append("skewed columns: " + ", ".join(s["column"] for s in skewed[:5]))
        return "; ".join(parts) if parts else "nothing unusual was flagged"

    def _target_description(self, info: Dict[str, Any]) -> str:
        if info["task_type"] == "regression":
            return f"Numeric target with mean {info['mean']:,.2f} and standard deviation {info['std']:,.2f}."
        balance = ", ".join(f"'{k}' {v:.0%}" for k, v in info["balance"].items())
        text = f"Classification target with {info['n_classes']} classes: {balance}."
        if info.get("rare_classes_dropped"):
            text += f" Classes with too few rows were left out of the screen: {', '.join(info['rare_classes_dropped'])}."
        return text

    def _summarize(self, target, task_type, info, n_rows, hypotheses, screening,
                   leak_names, excluded, eda_context) -> str:
        model = settings.HYPOTHESIS_MODEL
        try:
            import ollama
            prompt = build_hypothesis_prompt(
                target, task_type, self._target_description(info), n_rows, hypotheses,
                screening[:8], len(screening), leak_names, excluded, eda_context,
            )
            # gpt-oss cannot switch thinking off and ignores think=False. It needs
            # a level, and its reasoning tokens count against num_predict, so a
            # small budget can leave nothing for the actual answer.
            is_gpt_oss = "gpt-oss" in model.lower()
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                think="low" if is_gpt_oss else False,
                options={"num_predict": 2500 if is_gpt_oss else 1000},
            )
            text = response["message"]["content"].strip()
            if not text:
                raise ValueError("the model returned an empty answer")
            return text
        except Exception as e:
            self.log(f"Ollama call failed ({e}), falling back to template")
            return self._template_summary(hypotheses, leak_names)

    def _template_summary(self, hypotheses, leak_names) -> str:
        lines = [f"{h['id']} ({h['type']}): {h['claim']}" for h in hypotheses]
        if leak_names:
            lines.append(f"Suspected leaks: {', '.join(leak_names)}.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # step 2: verify the hypotheses against SHAP, after explainability
    # ------------------------------------------------------------------
    def verify(self, state: Dict[str, Any]) -> Dict[str, Any]:
        hypotheses = state["hypotheses"]
        importance = state["feature_importance"]
        flags = state.get("explainability_flags", {})

        known_cols = [r["feature"] for r in state.get("feature_screening", [])]
        known_cols += list(state.get("excluded_columns", {}).keys())
        grouped = self._group_importance(importance, known_cols)

        results = [self._judge(h, grouped, flags) for h in hypotheses]
        scorecard = dict(Counter(r["status"] for r in results))
        text = self._format_results(results, scorecard)

        self.log("verified " + ", ".join(f"{n} {s}" for s, n in scorecard.items()))
        return {
            "hypothesis_results": results,
            "hypothesis_scorecard": scorecard,
            "hypothesis_verification_summary": text,
        }

    def _source_column(self, shap_name: str, known_cols: List[str]) -> str:
        """
        SHAP works on encoded columns such as 'Region_East' or 'City (frequency)'.
        Map each one back to the original column it came from, preferring the
        longest matching name so 'Ship Mode' is not confused with 'Ship'.
        """
        matches = [
            c for c in known_cols
            if shap_name == c or shap_name.startswith(c + "_") or shap_name == f"{c} (frequency)"
        ]
        return max(matches, key=len) if matches else shap_name

    def _direction_word(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        if "up" in text:
            return "up"
        if "down" in text:
            return "down"
        return "mixed"

    def _group_importance(self, importance, known_cols) -> Dict[str, Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = {}
        for row in importance:
            src = self._source_column(row["feature"], known_cols)
            g = grouped.setdefault(src, {"share": 0.0, "direction": None, "dummies": {}})
            g["share"] += float(row["share_pct"])
            word = self._direction_word(row.get("direction"))
            if row["feature"] == src:
                g["direction"] = word
            else:
                g["dummies"][row["feature"]] = word
        ordered = sorted(grouped.items(), key=lambda kv: -kv[1]["share"])
        for rank, (_, g) in enumerate(ordered, start=1):
            g["rank"] = rank
        return grouped

    def _judge(self, h: Dict[str, Any], grouped, flags) -> Dict[str, Any]:
        check = h["check"]
        kind, feature = check["kind"], check.get("feature")
        base = {"id": h["id"], "type": h["type"], "feature": feature, "claim": h["claim"],
                "shap_rank": None, "shap_share_pct": None}
        near_chance = bool(flags.get("near_chance"))
        perf = flags.get("performance_value")
        metric = flags.get("performance_metric", "score")

        leak_dominates = bool(flags.get("possible_leakage")) and bool(flags.get("dominant_feature"))

        if kind == "weak_signal":
            if near_chance:
                return {**base, "status": "confirmed",
                        "detail": f"The model scored {perf:.2f} on {metric}, close to chance, as expected."}
            if leak_dominates:
                return {**base, "status": "inconclusive",
                        "detail": (f"The model scored {perf:.2f} on {metric}, but a suspected leak dominates it, so this "
                                   f"cannot be tested until the leak is removed and the model retrained.")}
            good = perf is not None and ((metric == "AUC-ROC" and perf >= 0.65) or (metric == "R2" and perf >= 0.10))
            if good:
                return {**base, "status": "contradicted",
                        "detail": (f"The model scored {perf:.2f} on {metric}, so features do carry signal. "
                                   f"It most likely comes from combinations of features, which the one-at-a-time screen cannot see.")}
            return {**base, "status": "inconclusive", "detail": "The model score sits between clearly weak and clearly useful."}

        info = grouped.get(feature)
        if info is None:
            if kind == "leakage":
                return {**base, "status": "inconclusive",
                        "detail": f"'{feature}' was removed before training, so it could not leak into the model."}
            return {**base, "status": "inconclusive",
                    "detail": f"'{feature}' was not in the model (it was dropped before training)."}

        if kind in ("driver", "no_effect"):
            if near_chance:
                return {**base, "status": "inconclusive",
                        "detail": "The model is barely better than chance, so its importance ranking cannot confirm or contradict anything."}
            if leak_dominates:
                return {**base, "status": "inconclusive",
                        "detail": "A suspected leak dominates the model, so its ranking of other features says little. Remove the leak and retrain."}

        share, rank, n_source = info["share"], info["rank"], len(grouped)
        base.update(shap_rank=rank, shap_share_pct=round(share, 1))
        where = f"rank {rank} of {n_source}, {share:.1f} percent of importance"

        if kind == "driver":
            ignored = share < self.LOW_SHARE or rank > max(self.TOP_K, int(np.ceil(0.66 * n_source)))
            if ignored:
                return {**base, "status": "contradicted", "detail": f"The model barely uses '{feature}' ({where})."}
            if rank > self.TOP_K:
                return {**base, "status": "inconclusive", "detail": f"'{feature}' has moderate importance ({where}), outside the top {self.TOP_K}."}

            expected = check.get("expected_direction", "none")
            category = check.get("expected_category")
            if category:
                found = info["dummies"].get(f"{feature}_{category}")
            else:
                found = info["direction"]

            if expected == "none":
                return {**base, "status": "confirmed", "detail": f"'{feature}' is a top driver ({where})."}
            if found is None:
                return {**base, "status": "partly confirmed",
                        "detail": f"'{feature}' is a top driver ({where}), but the direction could not be checked."}
            if expected == "nonlinear":
                if found == "mixed":
                    return {**base, "status": "confirmed", "detail": f"'{feature}' is a top driver ({where}) with a mixed direction, as expected."}
                return {**base, "status": "partly confirmed",
                        "detail": f"'{feature}' is a top driver ({where}), but SHAP shows a fairly simple '{found}' effect."}
            if found == expected:
                return {**base, "status": "confirmed", "detail": f"'{feature}' is a top driver ({where}) and pushes predictions {found}, as expected."}
            if found == "mixed":
                return {**base, "status": "partly confirmed",
                        "detail": f"'{feature}' is a top driver ({where}), but its direction is mixed instead of '{expected}'."}
            return {**base, "status": "contradicted",
                    "detail": f"'{feature}' is a top driver ({where}) but pushes predictions {found}, opposite to the hypothesis."}

        if kind == "no_effect":
            bottom_third = rank > int(np.ceil(2 * n_source / 3))
            clearly_used = rank <= min(self.TOP_K, int(np.ceil(n_source / 3)))
            interaction_note = (
                "The screen tests one feature at a time, so this usually means it matters "
                "in combination with other features."
            )
            if share < self.LOW_SHARE or bottom_third:
                return {**base, "status": "confirmed", "detail": f"'{feature}' plays a minor role ({where})."}
            if clearly_used:
                return {**base, "status": "contradicted",
                        "detail": f"The model relies on '{feature}' ({where}). {interaction_note}"}
            return {**base, "status": "inconclusive",
                    "detail": f"'{feature}' has middling importance ({where}), so 'no signal' is not supported. {interaction_note}"}

        if kind == "leakage":
            if share >= 50 or (rank == 1 and flags.get("possible_leakage")):
                return {**base, "status": "confirmed",
                        "detail": f"'{feature}' dominates the model ({where}), consistent with a leak. Remove it and retrain."}
            if share < self.LOW_SHARE:
                return {**base, "status": "contradicted",
                        "detail": f"The model barely uses '{feature}' ({where}). Another feature probably carries the same information."}
            return {**base, "status": "inconclusive", "detail": f"'{feature}' has moderate importance ({where})."}

        return {**base, "status": "inconclusive", "detail": "Unknown hypothesis type."}

    def _format_results(self, results, scorecard) -> str:
        head = "Hypothesis check against SHAP: " + ", ".join(f"{n} {s}" for s, n in scorecard.items()) + "."
        lines = [head]
        for r in results:
            lines.append(f"{r['id']} [{r['status']}] {r['claim']}\n    {r['detail']}")
        return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m agents.hypothesis_agent path/to/dataset.csv target_column [model_name]")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    target = sys.argv[2]
    agent = HypothesisAgent()
    state: Dict[str, Any] = {"cleaned_df": df, "target_column": target}
    state.update(agent.run(state))

    print("\n--- Target ---")
    print(state["target_info"])
    print("\n--- Feature screen (best first) ---")
    screen = pd.DataFrame(state["feature_screening"])[["feature", "kind", "metric", "solo_score", "rho"]]
    print(screen.head(10).to_string(index=False))
    print("\n--- Hypotheses ---")
    for h in state["hypotheses"]:
        print(f"{h['id']} [{h['type']}] {h['claim']}\n    check: {h['check']['description']}")
    print(f"\nSuspected leaks: {state['leakage_suspects'] or 'none'}")
    print(f"Left out: {state['excluded_columns'] or 'nothing'}")
    print("\n--- Plan ---")
    print(state["hypothesis_summary"])

    # optional: train a model, explain it, and check the hypotheses against SHAP
    if len(sys.argv) > 3:
        from agents.explainability_agent import ExplainabilityAgent
        from agents.tuning_agent import TuningAgent, _quick_prepare

        model_name = sys.argv[3]
        drop = list(state["leakage_suspects"])
        prepared = _quick_prepare(df.drop(columns=drop), target)
        y = prepared[target]
        is_clf = state["target_info"]["task_type"] == "classification"
        task_type = state["target_info"]["task_type"]
        label_mapping = None
        if is_clf and not pd.api.types.is_numeric_dtype(y):
            label_mapping = {i: c for i, c in enumerate(sorted(y.astype(str).unique()))}

        explainer = ExplainabilityAgent()
        X_train, X_test, y_train, y_test = explainer._prepare_split(prepared, target, task_type, label_mapping)
        tuner = TuningAgent()
        model = tuner._build_model(model_name, task_type, tuner.DEFAULT_PARAMS.get(model_name, {}))
        model.fit(X_train, y_train)

        state.update({
            "feature_df": prepared, "task_type": task_type, "trained_model": model,
            "model_name": model_name, "label_mapping": label_mapping, "run_id": "hypothesis_test",
        })
        state.update(explainer.run(state))
        state.update(agent.verify(state))
        print("\n--- Verification ---")
        print(state["hypothesis_verification_summary"])
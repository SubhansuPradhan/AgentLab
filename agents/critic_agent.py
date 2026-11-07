"""
Critic Agent

Job: review everything the pipeline produced and decide, by rules, whether
the result can be trusted, needs fixing, or should go to the report with
warnings. This is the project's research question in code: does a pipeline
that checks and corrects itself end up with better, more trustworthy
results than one that never checks?

How it decides: every check is a rule that measures something, so the
verdict is reproducible and can be defended. The LLM only explains the
verdict afterwards. The critic re-measures the model itself instead of
trusting the numbers upstream agents reported.

It only edits the pipeline when the evidence is deterministic: a column that
the Hypothesis Agent found predicts the target almost perfectly on its own
(a leak), or an identifier column that is still being used as a feature.
Everything else (no signal, overfitting, disagreement with the hypotheses,
small samples) is reported as a warning, because looping the same pipeline
cannot fix it and guessing at a fix would make things worse.

Verdicts:
    approve                 nothing that needs a caveat (minor notes at most)
    approve_with_warnings   problems the loop cannot fix, or the revision
                            limit was reached; goes to the report with caveats
    revise                  a fixable problem was found; the pipeline is
                            sent back to feature engineering
Revisions are capped (max_revisions, default 2) so it can never loop forever.

Ablation switch for the research comparison: set critic_enabled=False in
state. The critic then runs every check and records the flags, but always
approves and never changes the pipeline. Running each dataset once with it
on and once off gives the "with critic versus without" comparison, and the
recorded flags show exactly what the no-critic pipeline shipped silently.

Inputs (from state):
    - feature_df, target_column, task_type, label_mapping
    - final_model (or tuned_model, or trained_model): the model to review
    - model_name, model_metrics, final_model_choice, tuning_comparison:
      from the Model Selection Agent
    - feature_importance, explainability_flags, local_example:
      from the Explainability Agent
    - leakage_suspects, excluded_columns, feature_screening:
      from the Hypothesis Agent
    - hypothesis_results, hypothesis_scorecard: from Hypothesis verify()
    - critic_enabled (default True), revision_count (default 0),
      max_revisions (default 2)

Outputs of run() (added to state):
    - critique: dict with verdict, flags, severity counts, mode, and reasoning
    - loop_back_to: "feature_engineering" or None
    - revision_plan: dict describing the fix, or None
    - critic_summary: str, LLM-written explanation

apply_revision(state) is the graph node that carries the fix out: it drops
the flagged columns from cleaned_df and returns the updated state, so the
graph can send it back through feature engineering.
"""
import re
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from agents.base_agent import BaseAgent
from prompts.critic_prompt import build_critic_prompt
from config.settings import settings


class CriticAgent(BaseAgent):
    name = "critic_agent"

    MAX_REVISIONS = 2
    IDENTIFIER_MIN_SHARE = 2.0     # percent of importance before an identifier forces a re-run
    OVERFIT_GAP = {"classification": 0.15, "regression": 0.25}
    CV_TEST_SIGMAS = 3.0           # test score this many fold-standard-deviations below CV
    NEAR_CHANCE = {"classification": 0.6, "regression": 0.05}
    PERFECT = 0.98
    RANK_SIGNAL = 0.2
    MIN_TRAIN_ROWS = 300
    MIN_TEST_ROWS = 100
    MINORITY_SHARE = 0.10
    SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

    CHECKS_RUN = [
        "data leakage", "identifier columns used as features", "signal strength",
        "too-good-to-be-true score", "overfitting", "agreement with the hypotheses",
        "sample size", "explanation reliability", "class imbalance",
    ]

    # ------------------------------------------------------------------
    # main flow
    # ------------------------------------------------------------------
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        task_type: str = state.get("task_type", "classification")
        enabled = bool(state.get("critic_enabled", True))
        round_no = int(state.get("revision_count", 0))
        max_rev = int(state.get("max_revisions", self.MAX_REVISIONS))

        measured = self._measure(state, task_type)
        grouped = self._model_columns(state)

        checks = [
            self._check_leakage, self._check_identifiers, self._check_signal,
            self._check_perfect, self._check_overfitting, self._check_hypotheses,
            self._check_weak_signal, self._check_sample_size, self._check_explanations,
            self._check_imbalance, self._check_tuning,
        ]
        flags = [f for f in (c(state, task_type, measured, grouped) for c in checks) if f]
        flags.sort(key=lambda f: self.SEVERITY_ORDER[f["severity"]])

        decision = self._decide(flags, enabled, round_no, max_rev)
        summary = self._summarize(state, task_type, measured, flags, decision, enabled, round_no, max_rev)

        critique = {
            "verdict": decision["verdict"],
            "mode": "active" if enabled else "audit_only",
            "flags": flags,
            "severity_counts": dict(Counter(f["severity"] for f in flags)),
            "would_have_revised": decision["would_revise"],
            "revision_round": round_no,
            "max_revisions": max_rev,
            "limit_reached": decision["limit_reached"],
            "checks_run": self.CHECKS_RUN,
            "measured": measured,
            "reasoning": summary,
        }

        self.log(
            f"verdict={decision['verdict']} ({'active' if enabled else 'audit only'}), "
            f"{len(flags)} flag(s): "
            + (", ".join(f"{f['severity']}:{f['id']}" for f in flags) or "none")
        )

        return {
            "critique": critique,
            "loop_back_to": decision["loop_back_to"],
            "revision_plan": decision["revision_plan"],
            "critic_summary": summary,
        }

    def apply_revision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        The graph node that carries out the critic's fix. Drops the flagged
        columns from cleaned_df so the next pass through feature engineering
        never sees them. The target column is never dropped.
        """
        plan = state.get("revision_plan")
        if not plan:
            return {}
        df: pd.DataFrame = state["cleaned_df"]
        target = state.get("target_column")
        drop = [c for c in plan["columns"] if c in df.columns and c != target]

        history = list(state.get("revision_history", []))
        history.append({
            "round": plan["round"], "dropped": drop, "reasons": plan.get("reasons", []),
            "verdict_before": state.get("critique", {}).get("verdict"),
        })
        return {
            "cleaned_df": df.drop(columns=drop),
            "revision_count": int(state.get("revision_count", 0)) + 1,
            "dropped_by_critic": sorted(set(state.get("dropped_by_critic", [])) | set(drop)),
            "revision_history": history,
        }

    # ------------------------------------------------------------------
    # independent re-measurement
    # ------------------------------------------------------------------
    def _pick_model(self, state: Dict[str, Any]):
        for key in ("final_model", "tuned_model", "trained_model"):
            if state.get(key) is not None:
                return state[key]
        return None

    def _split(self, state: Dict[str, Any], task_type: str):
        """Same split settings as every other agent, so the test rows are the same."""
        feature_df, target = state["feature_df"], state["target_column"]
        if target in feature_df.columns:
            resolved = target
        else:
            candidates = [c for c in feature_df.columns if c.endswith(f"__{target}")]
            if not candidates:
                raise KeyError(f"Target column '{target}' not found in feature_df.")
            resolved = candidates[0]

        X, y = feature_df.drop(columns=[resolved]), feature_df[resolved]
        mapping = state.get("label_mapping")
        if task_type == "classification" and not pd.api.types.is_numeric_dtype(y):
            if mapping:
                reverse = {label: int(code) for code, label in mapping.items()}
                y = y.astype(str).map(reverse).astype(int)
            else:
                y = pd.Series(LabelEncoder().fit_transform(y), index=y.index, name=y.name)
        return train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if task_type == "classification" else None,
        )

    def _measure(self, state: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """
        Re-score the model on the held-out test rows instead of trusting the
        numbers an upstream agent reported. Returns whatever it could measure.
        """
        model = self._pick_model(state)
        if model is None or "feature_df" not in state:
            return {}
        try:
            X_train, X_test, y_train, y_test = self._split(state, task_type)
        except Exception:
            return {}

        out: Dict[str, Any] = {"n_train": int(len(X_train)), "n_test": int(len(X_test))}
        try:
            if task_type == "classification":
                proba = model.predict_proba(X_test)
                if proba.shape[1] == 2:
                    value = float(roc_auc_score(y_test, proba[:, 1]))
                else:
                    value = float(roc_auc_score(y_test, proba, multi_class="ovr", average="weighted"))
                train_score = float(f1_score(y_train, model.predict(X_train), average="weighted"))
                test_score = float(f1_score(y_test, model.predict(X_test), average="weighted"))
                share = pd.concat([y_train, y_test]).value_counts(normalize=True)
                out.update(
                    metric="AUC-ROC", value=value, rank_corr=None,
                    train_score=train_score, test_score=test_score,
                    minority_share=float(share.min()), minority_class=str(share.idxmin()),
                )
            else:
                preds = model.predict(X_test)
                rc = spearmanr(preds, y_test)[0]
                out.update(
                    metric="R2", value=float(r2_score(y_test, preds)),
                    rank_corr=None if np.isnan(rc) else float(rc),
                    train_score=float(r2_score(y_train, model.predict(X_train))),
                    test_score=float(r2_score(y_test, preds)),
                )
            out["gap"] = out["train_score"] - out["test_score"]
        except Exception:
            pass
        return out

    # ------------------------------------------------------------------
    # which original columns are in the model, and how much they matter
    # ------------------------------------------------------------------
    def _source_column(self, name: str, known: List[str]) -> str:
        """Map an encoded name such as 'Region_East' back to the column 'Region'."""
        matches = [c for c in known if name == c or name.startswith(c + "_") or name == f"{c} (frequency)"]
        return max(matches, key=len) if matches else name

    def _model_columns(self, state: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        importance = state.get("feature_importance") or []
        known = [r["feature"] for r in state.get("feature_screening", []) or []]
        known += list((state.get("excluded_columns") or {}).keys())
        known += [c for c in (state.get("leakage_suspects") or [])]
        grouped: Dict[str, Dict[str, float]] = {}
        for row in importance:
            src = self._source_column(row["feature"], known)
            g = grouped.setdefault(src, {"share": 0.0})
            g["share"] += float(row["share_pct"])
        for rank, (_, g) in enumerate(sorted(grouped.items(), key=lambda kv: -kv[1]["share"]), start=1):
            g["rank"] = rank
        return grouped

    # ------------------------------------------------------------------
    # the checks, each returns a flag dict or None
    # ------------------------------------------------------------------
    def _flag(self, fid, severity, title, evidence, fix_columns=None):
        return {
            "id": fid, "severity": severity, "title": title, "evidence": evidence,
            "fix": {"action": "drop_columns", "columns": list(fix_columns)} if fix_columns else None,
        }

    def _is_near_chance(self, m: Dict[str, Any], task_type: str, state: Dict[str, Any]) -> bool:
        if m.get("value") is None:
            return bool((state.get("explainability_flags") or {}).get("near_chance"))
        if task_type == "classification":
            return m["value"] < self.NEAR_CHANCE["classification"]
        rank_ok = m.get("rank_corr") is not None and m["rank_corr"] >= self.RANK_SIGNAL
        return m["value"] < self.NEAR_CHANCE["regression"] and not rank_ok

    def _check_leakage(self, state, task_type, m, grouped):
        suspects = state.get("leakage_suspects") or []
        present = [c for c in suspects if c in grouped]
        if not present:
            return None
        parts = ", ".join(f"'{c}' ({grouped[c]['share']:.1f} percent of importance)" for c in present)
        return self._flag(
            "leakage_in_model", "critical", "A suspected data leak is still in the model",
            f"The Hypothesis Agent found that {parts} predicts the target almost perfectly on its own, "
            f"and it is being used as a feature.",
            present,
        )

    def _check_identifiers(self, state, task_type, m, grouped):
        excluded = state.get("excluded_columns") or {}
        used = {c: r for c, r in excluded.items() if "identifier" in r and c in grouped}
        if not used:
            return None
        material = [c for c in used if grouped[c]["share"] >= self.IDENTIFIER_MIN_SHARE]
        listing = ", ".join(f"'{c}' ({grouped[c]['share']:.1f} percent)" for c in used)
        if material:
            return self._flag(
                "identifier_in_model", "high", "An identifier column is being used as a feature",
                f"{listing} looks like an identifier, and the model gives it real weight. It can memorize "
                f"rows instead of learning a pattern.",
                material,
            )
        return self._flag(
            "identifier_minor", "low", "An identifier column is in the model but barely used",
            f"{listing} looks like an identifier. It carries little weight, so no re-run is needed, but it "
            f"should be removed.",
        )

    def _check_signal(self, state, task_type, m, grouped):
        if not self._is_near_chance(m, task_type, state):
            return None
        if m.get("value") is not None:
            ev = f"{m['metric']} is {m['value']:.2f}"
            if m.get("rank_corr") is not None:
                ev += f" (rank correlation {m['rank_corr']:.2f})"
        else:
            ev = "the model scores close to chance"
        return self._flag(
            "no_signal", "high", "The model is barely better than guessing",
            f"{ev}, so the features carry almost no real signal. Sending the pipeline back cannot fix this, "
            f"better features or more data are needed.",
        )

    def _check_perfect(self, state, task_type, m, grouped):
        if m.get("value") is None or m["value"] <= self.PERFECT:
            return None
        if any(c in grouped for c in (state.get("leakage_suspects") or [])):
            return None
        top = ""
        if grouped:
            name, g = min(grouped.items(), key=lambda kv: kv[1]["rank"])
            top = f" The top feature is '{name}' with {g['share']:.0f} percent of importance."
        return self._flag(
            "score_too_good", "high", "The score is too good to be true",
            f"{m['metric']} is {m['value']:.2f}, which is rare on real data and often means a feature leaks "
            f"the answer.{top} No column was proven to be the leak, so a person has to check it.",
        )

    def _check_overfitting(self, state, task_type, m, grouped):
        if self._is_near_chance(m, task_type, state):
            return None
        model = self._pick_model(state)
        is_forest = model is not None and type(model).__name__.startswith("RandomForest")
        reasons, strong = [], False

        gap, thr = m.get("gap"), self.OVERFIT_GAP[task_type]
        if gap is not None and not is_forest and gap >= thr:
            reasons.append(
                f"it scores {m['train_score']:.2f} on the data it trained on but {m['test_score']:.2f} on "
                f"unseen data (gap {gap:.2f})"
            )
            strong = strong or gap >= 2 * thr

        name = state.get("model_name")
        entry = (state.get("model_metrics") or {}).get(name or "", {})
        if entry.get("test_score") is not None and state.get("final_model_choice") != "tuned":
            cv_gap = entry["mean"] - entry["test_score"]
            if cv_gap > max(self.CV_TEST_SIGMAS * entry["std"], 0.05 * abs(entry["mean"])):
                reasons.append(
                    f"cross validation predicted {entry['mean']:.3f} but the test set gave "
                    f"{entry['test_score']:.3f}, much worse than the folds varied"
                )
                strong = strong or len(reasons) > 1

        if not reasons:
            return None
        return self._flag(
            "overfitting", "high" if strong else "medium", "The model does worse on new data than it should",
            "Overfitting: " + " and ".join(reasons) + ". Consider fewer features, stronger regularization, or more data.",
        )

    def _check_hypotheses(self, state, task_type, m, grouped):
        results = state.get("hypothesis_results") or []
        bad = [r for r in results if r.get("status") == "contradicted" and r.get("type") != "leakage"]
        if not bad:
            return None
        listing = "; ".join(f"{r['id']} ({r.get('feature') or 'overall'})" for r in bad)
        return self._flag(
            "hypotheses_contradicted", "medium", "The model disagrees with the hypotheses",
            f"{len(bad)} hypothesis(es) were contradicted by the SHAP results: {listing}. This often means a "
            f"feature matters only in combination with others, or that preprocessing changed what a feature carries.",
        )

    def _check_weak_signal(self, state, task_type, m, grouped):
        flags = state.get("explainability_flags") or {}
        if task_type != "regression" or not flags.get("weak_but_ranks_correctly"):
            return None
        return self._flag(
            "weak_signal", "low", "The signal is real but weak",
            f"R2 is low ({m.get('value', float('nan')):.2f}) though predictions still rank rows sensibly, so "
            f"treat the model's accuracy claims with caution.",
        )

    def _check_sample_size(self, state, task_type, m, grouped):
        n_train, n_test = m.get("n_train"), m.get("n_test")
        if n_train is None:
            return None
        small = []
        if n_train < self.MIN_TRAIN_ROWS:
            small.append(f"only {n_train} training rows")
        if n_test < self.MIN_TEST_ROWS:
            small.append(f"only {n_test} test rows")
        if not small:
            return None
        return self._flag(
            "small_sample", "medium", "The dataset is small",
            f"There are {' and '.join(small)}, so the scores can move a lot with a different split.",
        )

    def _check_explanations(self, state, task_type, m, grouped):
        if self._is_near_chance(m, task_type, state):
            return None
        notes = []
        local = state.get("local_example") or {}
        if local.get("lime_available") and local.get("lime_agreement", 5) <= 1:
            notes.append(f"LIME agreed with SHAP on only {local['lime_agreement']} of the top 5 features")
        eflags = state.get("explainability_flags") or {}
        if eflags.get("rows_explained") and eflags.get("rows_requested") and eflags["rows_explained"] < eflags["rows_requested"]:
            notes.append(f"SHAP covered {eflags['rows_explained']} of {eflags['rows_requested']} planned rows")
        if not notes:
            return None
        return self._flag(
            "explanation_limits", "low", "The explanations have limits",
            "; ".join(notes) + ". Treat the feature rankings as rough.",
        )

    def _check_imbalance(self, state, task_type, m, grouped):
        if task_type != "classification" or m.get("minority_share") is None:
            return None
        if m["minority_share"] >= self.MINORITY_SHARE:
            return None
        return self._flag(
            "class_imbalance", "low", "The classes are imbalanced",
            f"The smallest class ('{m['minority_class']}') is only {m['minority_share']:.0%} of the data, so plain "
            f"accuracy is misleading. Rely on F1 and AUC-ROC.",
        )

    def _check_tuning(self, state, task_type, m, grouped):
        c = state.get("tuning_comparison")
        if not c or not c.get("cv_test_mismatch"):
            return None
        return self._flag(
            "tuning_mismatch", "low", "Tuning looked better in validation but not on the test set",
            f"Cross validation improved ({c['cv_gain']:+.4f}) but the test score did not ({c['test_gain']:+.4f}), "
            f"which usually means the search fit the validation folds by chance. The default settings are kept.",
        )

    # ------------------------------------------------------------------
    # the decision
    # ------------------------------------------------------------------
    def _decide(self, flags, enabled: bool, round_no: int, max_rev: int) -> Dict[str, Any]:
        fixable = [f for f in flags if f.get("fix")]
        columns = sorted({c for f in fixable for c in f["fix"]["columns"]})
        base = {"loop_back_to": None, "revision_plan": None, "limit_reached": False, "would_revise": bool(fixable)}

        if not enabled:
            return {**base, "verdict": "approve"}

        if fixable and round_no < max_rev:
            plan = {
                "action": "drop_columns", "columns": columns, "round": round_no + 1,
                "reasons": [f["title"] for f in fixable],
            }
            return {**base, "verdict": "revise", "loop_back_to": "feature_engineering", "revision_plan": plan}

        serious = [f for f in flags if f["severity"] in ("critical", "high", "medium")]
        limit = bool(fixable)
        verdict = "approve_with_warnings" if (serious or limit) else "approve"
        return {**base, "verdict": verdict, "limit_reached": limit}

    # ------------------------------------------------------------------
    # LLM explanation
    # ------------------------------------------------------------------
    def _performance_line(self, m: Dict[str, Any]) -> str:
        if m.get("value") is None:
            return "could not be measured"
        line = f"{m['metric']} {m['value']:.3f} on {m.get('n_test', '?')} unseen test rows"
        if m.get("rank_corr") is not None:
            line += f", rank correlation {m['rank_corr']:.2f}"
        if m.get("train_score") is not None:
            line += f"; train score {m['train_score']:.3f} versus test score {m['test_score']:.3f}"
        return line

    def _summarize(self, state, task_type, m, flags, decision, enabled, round_no, max_rev) -> str:
        model = settings.CRITIC_MODEL
        try:
            import ollama
            prompt = build_critic_prompt(
                task_type, decision["verdict"], "active" if enabled else "audit_only", round_no, max_rev,
                decision["limit_reached"],
                f"{state.get('model_name', 'unknown')} ({state.get('final_model_choice', 'default')} settings)",
                self._performance_line(m), flags, decision["revision_plan"], self.CHECKS_RUN,
                state.get("hypothesis_scorecard"),
            )
            is_gpt_oss = "gpt-oss" in model.lower()
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                think="low" if is_gpt_oss else False,
                options={"num_predict": 2500 if is_gpt_oss else 1000, "num_ctx": 4096},
            )
            text = response["message"]["content"].strip()
            if not text:
                raise ValueError("the model returned an empty answer")
            return text
        except Exception as e:
            self.log(f"Ollama call failed ({e}), falling back to template")
            return self._template_summary(flags, decision, enabled)

    def _template_summary(self, flags, decision, enabled) -> str:
        lines = [f"Verdict: {decision['verdict']}."]
        if not enabled:
            lines.append("The critic is switched off, so these problems were recorded but not acted on.")
        for f in flags:
            lines.append(f"[{f['severity']}] {f['title']}: {f['evidence']}")
        if not flags:
            lines.append("No problems were found by the checks: " + ", ".join(self.CHECKS_RUN) + ".")
        plan = decision["revision_plan"]
        if plan:
            lines.append(
                f"The pipeline is sent back to feature engineering with "
                f"{', '.join(repr(c) for c in plan['columns'])} removed (revision {plan['round']})."
            )
        if decision["limit_reached"]:
            lines.append("The revision limit was reached, so remaining fixable problems are passed on as warnings.")
        return "\n".join(lines)
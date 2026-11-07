"""
Report Agent

Job: compile everything the pipeline produced into one readable report.

Design choice: this agent does NOT ask an LLM to re-describe what every
other agent already found. Each specialist agent (Hypothesis, Model
Selection, Explainability, Critic) already wrote its own verified
summary, grounded in numbers that agent itself computed. The Report
Agent copies those sections in verbatim and only asks the LLM for two
short sections it alone can write: an Executive Summary and a
Conclusion that tie the whole run together. This means the report can
never contradict a number a specialist agent already checked.

Inputs (from state):
    - target_column, task_type, final_model_choice, model_name
    - feature_df or cleaned_df: used only to report row/column counts
    - hypothesis_summary, hypotheses, hypothesis_scorecard (optional)
    - model_selection_summary, stats_table_str, tuning_plan,
      tuning_comparison (optional), confusion_matrix_path (optional)
    - explainability_summary, feature_importance, shap_plots (optional)
    - critic_summary, critique (verdict, flags, mode)
    - run_id: used to namespace where the report file is saved
    - dataset_name: optional, falls back to "the dataset"

Outputs (added to state):
    - report_path: str, path to the saved markdown report
    - report_sections: dict, every section's text (for a Streamlit view
      to render section by section without re-reading the file)
    - report_summary: str, just the Executive Summary, for a quick look
"""
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from agents.base_agent import BaseAgent
from prompts.report_prompt import build_report_prompt
from config.settings import settings


class ReportAgent(BaseAgent):
    name = "report_agent"

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        run_id = state.get("run_id", "default_run")
        facts = self._gather_facts(state)

        exec_summary, conclusion = self._write_opening_and_closing(facts)

        sections: Dict[str, str] = {
            "Executive Summary": exec_summary,
            "Dataset and Target": self._dataset_section(facts, state),
            "Hypotheses": state.get("hypothesis_summary") or "Not generated for this run.",
            "Model Performance": self._model_section(state),
            "Explainability": self._explainability_section(state),
            "Critic's Review": self._critic_section(state),
            "Conclusion and Recommendations": conclusion,
        }

        report_md = self._render_markdown(facts, sections, state)
        report_path = self._save(report_md, run_id)

        self.log(f"report written to {report_path} ({len(report_md)} characters)")

        return {
            "report_path": report_path,
            "report_sections": sections,
            "report_summary": exec_summary,
        }

    # ------------------------------------------------------------------
    # gather the facts the LLM is allowed to use
    # ------------------------------------------------------------------
    def _row_count(self, state: Dict[str, Any]) -> int:
        for key in ("feature_df", "cleaned_df"):
            df = state.get(key)
            if isinstance(df, pd.DataFrame):
                return len(df)
        return 0

    def _gather_facts(self, state: Dict[str, Any]) -> Dict[str, Any]:
        critique = state.get("critique") or {}
        flags = critique.get("flags") or []
        scorecard = state.get("hypothesis_scorecard")
        measured = critique.get("measured") or {}

        headline_metric = measured.get("metric") or ("AUC-ROC" if state.get("task_type") == "classification" else "R2")
        headline_value = measured.get("value")
        headline_str = f"{headline_value:.3f}" if headline_value is not None else "not available"

        importance = state.get("feature_importance") or []
        top_feature, top_share = (importance[0]["feature"], f"{importance[0]['share_pct']:.1f}") if importance else ("not available", "n/a")

        plan = state.get("tuning_plan") or {}
        if not plan.get("required"):
            tuning_line = "not required (" + (plan.get("reasons", ["no reason recorded"])[0]) + ")"
        elif state.get("final_model_choice") == "tuned":
            c = state.get("tuning_comparison") or {}
            tuning_line = f"run and adopted, test score changed by {c.get('relative_test_gain_pct', 0):+.2f} percent"
        else:
            tuning_line = "run but not adopted, the default settings scored as well or better"

        if flags:
            flag_summary = "; ".join(f"[{f['severity']}] {f['title']}" for f in flags[:3])
            if len(flags) > 3:
                flag_summary += f"; and {len(flags) - 3} more"
        else:
            flag_summary = "none"

        return {
            "dataset_name": state.get("dataset_name", "the dataset"),
            "n_rows": self._row_count(state),
            "target": state.get("target_column", "unknown"),
            "task_type": state.get("task_type", "unknown"),
            "model_name": state.get("model_name", "unknown"),
            "model_choice": state.get("final_model_choice", "default"),
            "headline_metric": headline_metric,
            "headline_value": headline_str,
            "tuning_line": tuning_line,
            "top_feature": top_feature,
            "top_feature_share": top_share,
            "n_hypotheses": len(state.get("hypotheses") or []),
            "hypothesis_scoreline": ", ".join(f"{v} {k}" for k, v in (scorecard or {}).items()) or "not checked yet",
            "critic_verdict": critique.get("verdict", "not reviewed"),
            "critic_mode": critique.get("mode", "active"),
            "critic_flag_summary": flag_summary,
        }

    # ------------------------------------------------------------------
    # sections copied verbatim from upstream agents
    # ------------------------------------------------------------------
    def _dataset_section(self, facts: Dict[str, Any], state: Dict[str, Any]) -> str:
        lines = [
            f"Dataset: {facts['dataset_name']}, {facts['n_rows']} rows used for this run.",
            f"Target: '{facts['target']}' ({facts['task_type']}).",
        ]
        excluded = state.get("excluded_columns") or {}
        if excluded:
            lines.append(
                f"{len(excluded)} column(s) were left out before modeling: "
                + ", ".join(f"'{c}' ({r})" for c, r in list(excluded.items())[:6])
                + ("..." if len(excluded) > 6 else "")
            )
        dropped = state.get("dropped_by_critic") or []
        if dropped:
            lines.append(
                f"{len(dropped)} column(s) were removed by the critic during a revision: "
                + ", ".join(f"'{c}'" for c in dropped) + "."
            )
        return "\n\n".join(lines)

    def _model_section(self, state: Dict[str, Any]) -> str:
        parts = [state.get("model_selection_summary") or "Not generated for this run."]
        table = state.get("stats_table_str")
        if table:
            parts.append("```\n" + table + "\n```")
        return "\n\n".join(parts)

    def _explainability_section(self, state: Dict[str, Any]) -> str:
        return state.get("explainability_summary") or "Not generated for this run."

    def _critic_section(self, state: Dict[str, Any]) -> str:
        critique = state.get("critique") or {}
        header = f"**Verdict: {critique.get('verdict', 'not reviewed')}** (critic mode: {critique.get('mode', 'active')})"
        body = state.get("critic_summary") or "Not generated for this run."
        return header + "\n\n" + body

    # ------------------------------------------------------------------
    # the two sections the LLM writes
    # ------------------------------------------------------------------
    def _write_opening_and_closing(self, facts: Dict[str, Any]) -> (str, str):
        try:
            import ollama
            model = settings.REPORT_MODEL
            prompt = build_report_prompt(facts)
            is_gpt_oss = "gpt-oss" in model.lower()
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                think="low" if is_gpt_oss else False,
                options={"num_predict": 900, "num_ctx": 4096},
            )
            text = response["message"]["content"].strip()
            if not text:
                raise ValueError("the model returned an empty answer")
            return self._split_two_sections(text, facts)
        except Exception as e:
            self.log(f"Ollama call failed ({e}), falling back to template")
            return self._template_opening_and_closing(facts)

    def _split_two_sections(self, text: str, facts: Dict[str, Any]) -> (str, str):
        marker = "## Conclusion"
        idx = text.find(marker)
        if idx == -1:
            return text.replace("## Executive Summary", "").strip(), self._template_opening_and_closing(facts)[1]
        exec_part = text[:idx].replace("## Executive Summary", "").strip()
        conclusion_part = text[idx:].split("\n", 1)[1].strip() if "\n" in text[idx:] else ""
        return exec_part or self._template_opening_and_closing(facts)[0], conclusion_part or self._template_opening_and_closing(facts)[1]

    def _template_opening_and_closing(self, facts: Dict[str, Any]) -> (str, str):
        exec_summary = (
            f"This run modeled '{facts['target']}' ({facts['task_type']}) on {facts['n_rows']} rows of "
            f"{facts['dataset_name']}. The recommended model was {facts['model_name']} "
            f"({facts['model_choice']} settings), scoring {facts['headline_metric']} = {facts['headline_value']}. "
            f"Fine-tuning was {facts['tuning_line']}. The critic's verdict was "
            f"'{facts['critic_verdict']}' ({facts['critic_mode']} mode), with flags: {facts['critic_flag_summary']}."
        )
        conclusion = (
            f"The top driver of predictions was '{facts['top_feature']}' "
            f"({facts['top_feature_share']} percent of importance). "
            f"Hypothesis check: {facts['hypothesis_scoreline']}. "
            f"Before trusting this result, review the critic's verdict above and any flagged problems."
        )
        return exec_summary, conclusion

    # ------------------------------------------------------------------
    # assembly and saving
    # ------------------------------------------------------------------
    def _collect_plots(self, state: Dict[str, Any]) -> List[str]:
        plots = []
        for key in ("eda_plots", "shap_plots"):
            plots += state.get(key) or []
        for key in ("confusion_matrix_path", "tuning_plot_path"):
            if state.get(key):
                plots.append(state[key])
        return plots

    def _render_markdown(self, facts: Dict[str, Any], sections: Dict[str, str], state: Dict[str, Any]) -> str:
        lines = [
            f"# AgentLab Report: {facts['dataset_name']}",
            f"*Target: `{facts['target']}` ({facts['task_type']}) | Model: {facts['model_name']} "
            f"({facts['model_choice']}) | Critic verdict: {facts['critic_verdict']}*",
            "",
        ]
        for title, body in sections.items():
            lines.append(f"## {title}")
            lines.append(body)
            lines.append("")

        plots = self._collect_plots(state)
        if plots:
            lines.append("## Figures")
            for p in plots:
                name = os.path.splitext(os.path.basename(p))[0].replace("_", " ").title()
                lines.append(f"**{name}**")
                lines.append(f"![{name}]({p})")
                lines.append("")

        return "\n".join(lines)

    def _save(self, report_md: str, run_id: str) -> str:
        out_dir = os.path.join(settings.REPORTS_DIR, run_id)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "report.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_md)
        return path


if __name__ == "__main__":
    print("The Report Agent runs on the full pipeline's output state, not a raw CSV, "
          "so there is no standalone CLI for it. Run it as the last node after the "
          "Critic Agent, or build a small test state from the other agents' outputs.")
"""
Validation Agent

Job: take a raw uploaded dataset, check it for the usual data quality
problems, fix what's safe to fix automatically, and produce a plain
English summary of what it found. This is the first node in the
graph, everything downstream assumes the data it receives already
passed through here.

Inputs (from state):
    - raw_df: pandas DataFrame, the uploaded dataset as-is

Outputs (added to state):
    - cleaned_df: pandas DataFrame, after the safe automatic fixes
    - validation_report: dict of the raw stats (missing values,
      duplicates, dtypes, etc.)
    - validation_summary: str, the LLM-written plain English summary
"""
import sys
from typing import Any, Dict

import pandas as pd

from agents.base_agent import BaseAgent
from prompts.validation_prompt import build_validation_prompt
from config.settings import settings


class ValidationAgent(BaseAgent):
    name = "validation_agent"

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = state["raw_df"]

        stats = self._compute_stats(df)
        cleaned_df = self._apply_safe_fixes(df, stats)
        summary = self._summarize(stats)

        self.log(f"validated {len(df)} rows, {len(df.columns)} columns")

        return {
            "cleaned_df": cleaned_df,
            "validation_report": stats,
            "validation_summary": summary,
        }

    def _compute_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)

        return {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "missing_by_column": {
                col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
                for col in df.columns
                if missing[col] > 0
            },
            "duplicate_rows": int(df.duplicated().sum()),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "constant_columns": [
                col for col in df.columns if df[col].nunique(dropna=False) <= 1
            ],
            "high_cardinality_columns": [
                col
                for col in df.select_dtypes(include=["object", "string"]).columns
                if df[col].nunique() > 0.9 * len(df)
            ],
        }

    def _apply_safe_fixes(self, df: pd.DataFrame, stats: Dict[str, Any]) -> pd.DataFrame:
        """
        Only fixes that are safe to apply without human judgment:
        dropping exact duplicate rows and dropping columns that are
        entirely one constant value (they carry zero signal). Missing
        value imputation and encoding are NOT done here, that belongs
        to the Feature Engineering Agent, which needs to make
        judgment calls this agent shouldn't make silently.
        """
        cleaned = df.drop_duplicates()
        if stats["constant_columns"]:
            cleaned = cleaned.drop(columns=stats["constant_columns"])
        return cleaned.reset_index(drop=True)

    def _summarize(self, stats: Dict[str, Any]) -> str:
        """
        Calls the local Ollama model to turn the raw stats dict into a
        short human readable summary. Falls back to a plain templated
        string if Ollama isn't running or the model isn't pulled yet,
        so this agent never crashes your pipeline just because the
        LLM step failed, it degrades to something still useful.
        """
        prompt = build_validation_prompt(stats)
        try:
            import ollama

            response = ollama.chat(
                model=settings.OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                think=False,
            )
            return response["message"]["content"].strip()
        except Exception as e:
            self.log(
                f"Ollama call failed ({e}), falling back to template. "
                f"Check that `ollama serve` is running and "
                f"`ollama pull {settings.LOCAL_MODEL}` has been run."
            )
            return self._template_summary(stats)

    def _template_summary(self, stats: Dict[str, Any]) -> str:
        lines = [
            f"Dataset has {stats['n_rows']} rows and {stats['n_columns']} columns.",
        ]
        if stats["duplicate_rows"]:
            lines.append(f"Found and removed {stats['duplicate_rows']} duplicate rows.")
        if stats["missing_by_column"]:
            worst = max(stats["missing_by_column"].items(), key=lambda x: x[1]["pct"])
            lines.append(
                f"{len(stats['missing_by_column'])} columns have missing values, "
                f"worst is '{worst[0]}' at {worst[1]['pct']}% missing."
            )
        else:
            lines.append("No missing values found.")
        if stats["constant_columns"]:
            lines.append(
                f"Dropped {len(stats['constant_columns'])} constant column(s): "
                f"{', '.join(stats['constant_columns'])}."
            )
        if stats["high_cardinality_columns"]:
            lines.append(
                f"Warning: {', '.join(stats['high_cardinality_columns'])} "
                f"look like ID columns, consider excluding them from modeling."
            )
        return " ".join(lines)


def _run_standalone(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    agent = ValidationAgent()
    result = agent.run({"raw_df": df})

    print("\n--- Validation Summary ---")
    print(result["validation_summary"])
    print(f"\nCleaned dataset: {len(result['cleaned_df'])} rows "
          f"(was {len(df)}), {len(result['cleaned_df'].columns)} columns "
          f"(was {len(df.columns)})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m agents.validation_agent path/to/dataset.csv")
        sys.exit(1)
    _run_standalone(sys.argv[1])
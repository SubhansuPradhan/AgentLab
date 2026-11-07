"""
EDA Agent

Job: run exploratory data analysis on the cleaned dataset. Produces
summary statistics, correlation and skew checks, saved plots, and an
LLM-written description of what actually matters for modeling.

Inputs (from state):
    - cleaned_df: pandas DataFrame, output of the Validation Agent
    - run_id: str, used to namespace where plots get saved

Outputs (added to state):
    - eda_stats: dict of numeric/categorical summaries, correlations, skew
    - eda_plots: list of file paths to saved plot images
    - eda_summary: str, LLM-written description of what stands out
"""
import os
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from agents.base_agent import BaseAgent
from prompts.eda_prompt import build_eda_prompt
from config.settings import settings


class EDAAgent(BaseAgent):
    name = "eda_agent"

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = state["cleaned_df"]
        run_id = state.get("run_id", "default_run")

        stats = self._compute_stats(df)
        plot_paths = self._generate_plots(df, stats, run_id)
        summary = self._summarize(stats)

        self.log(f"analyzed {len(df)} rows, generated {len(plot_paths)} plots")

        return {
            "eda_stats": stats,
            "eda_plots": plot_paths,
            "eda_summary": summary,
        }

    def _compute_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        stats: Dict[str, Any] = {
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "numeric_summary": {},
            "categorical_summary": {},
            "high_correlations": [],
            "skewed_columns": [],
        }

        if numeric_cols:
            stats["numeric_summary"] = df[numeric_cols].describe().to_dict()

            skew = df[numeric_cols].skew()
            stats["skewed_columns"] = [
                {"column": col, "skew": round(float(val), 2)}
                for col, val in skew.items()
                if abs(val) > 1
            ]

            if len(numeric_cols) >= 2:
                corr = df[numeric_cols].corr()
                seen = set()
                for col_a in corr.columns:
                    for col_b in corr.columns:
                        if col_a == col_b or (col_b, col_a) in seen:
                            continue
                        seen.add((col_a, col_b))
                        value = corr.loc[col_a, col_b]
                        if abs(value) > 0.8:
                            stats["high_correlations"].append(
                                {"columns": [col_a, col_b], "correlation": round(float(value), 2)}
                            )

        for col in categorical_cols:
            value_counts = df[col].value_counts(normalize=True)
            stats["categorical_summary"][col] = {
                "unique_values": int(df[col].nunique()),
                "top_value": str(value_counts.index[0]) if len(value_counts) else None,
                "top_value_share": round(float(value_counts.iloc[0]), 2) if len(value_counts) else None,
            }

        return stats

    def _generate_plots(self, df: pd.DataFrame, stats: Dict[str, Any], run_id: str) -> list:
        out_dir = os.path.join(settings.REPORTS_DIR, run_id, "eda")
        os.makedirs(out_dir, exist_ok=True)
        paths = []

        numeric_cols = stats["numeric_columns"]
        categorical_cols = stats["categorical_columns"]

        # histograms, one per numeric column
        if numeric_cols:
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
            for i, col in enumerate(numeric_cols):
                sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
                axes[i].set_title(col)
            for j in range(len(numeric_cols), len(axes)):
                axes[j].axis("off")
            plt.tight_layout()
            hist_path = os.path.join(out_dir, "histograms.png")
            plt.savefig(hist_path)
            plt.close(fig)
            paths.append(hist_path)

        # correlation heatmap across all numeric columns
        if len(numeric_cols) >= 2:
            fig = plt.figure(figsize=(8, 6))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", center=0)
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            heatmap_path = os.path.join(out_dir, "correlation_heatmap.png")
            plt.savefig(heatmap_path)
            plt.close(fig)
            paths.append(heatmap_path)

        # boxplots, one per numeric column, for spotting outliers a histogram can hide
        if numeric_cols:
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
            for i, col in enumerate(numeric_cols):
                sns.boxplot(y=df[col].dropna(), ax=axes[i], color="skyblue")
                axes[i].set_title(col)
            for j in range(len(numeric_cols), len(axes)):
                axes[j].axis("off")
            plt.tight_layout()
            box_path = os.path.join(out_dir, "boxplots.png")
            plt.savefig(box_path)
            plt.close(fig)
            paths.append(box_path)

        # bar charts for categorical columns, top 10 categories each
        if categorical_cols:
            n_cols = min(2, len(categorical_cols))
            n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
            for i, col in enumerate(categorical_cols):
                top_vals = df[col].value_counts().head(10)
                sns.barplot(x=top_vals.values, y=top_vals.index.astype(str), ax=axes[i], color="steelblue")
                axes[i].set_title(f"{col} (top {len(top_vals)} categories)")
                axes[i].set_xlabel("Count")
            for j in range(len(categorical_cols), len(axes)):
                axes[j].axis("off")
            plt.tight_layout()
            bar_path = os.path.join(out_dir, "categorical_counts.png")
            plt.savefig(bar_path)
            plt.close(fig)
            paths.append(bar_path)

        # missing value map, shows where gaps sit across rows/columns, not just how many
        if df.isnull().values.any():
            fig = plt.figure(figsize=(10, 6))
            sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="Reds")
            plt.title("Missing Value Map (red = missing)")
            plt.tight_layout()
            missing_path = os.path.join(out_dir, "missing_value_map.png")
            plt.savefig(missing_path)
            plt.close(fig)
            paths.append(missing_path)

        # pairplot for the top 4 most correlated numeric columns
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr().abs()
            top_cols = corr.sum().sort_values(ascending=False).head(4).index.tolist()
            if len(top_cols) >= 2:
                pair_fig = sns.pairplot(df[top_cols].dropna())
                pair_fig.fig.suptitle("Pairwise Relationships (top correlated columns)", y=1.02)
                pair_path = os.path.join(out_dir, "pairplot.png")
                pair_fig.savefig(pair_path)
                plt.close(pair_fig.fig)
                paths.append(pair_path)

        return paths

    def _summarize(self, stats: Dict[str, Any]) -> str:
        try:
            import ollama
            prompt = build_eda_prompt(stats)
            response = ollama.chat(
                model=settings.OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                think=False,
            )
            return response["message"]["content"].strip()
        except Exception as e:
            self.log(f"Ollama call failed ({e}), falling back to template")
            return self._template_summary(stats)

    def _template_summary(self, stats: Dict[str, Any]) -> str:
        lines = [
            f"Dataset has {len(stats['numeric_columns'])} numeric and "
            f"{len(stats['categorical_columns'])} categorical columns."
        ]
        if stats["high_correlations"]:
            pair = stats["high_correlations"][0]
            lines.append(
                f"Found {len(stats['high_correlations'])} highly correlated column pair(s), "
                f"e.g. {pair['columns'][0]} and {pair['columns'][1]} at {pair['correlation']}."
            )
        if stats["skewed_columns"]:
            lines.append(f"{len(stats['skewed_columns'])} column(s) are notably skewed.")
        return " ".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m agents.eda_agent path/to/dataset.csv")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])
    agent = EDAAgent()
    result = agent.run({"cleaned_df": df, "run_id": "manual_test"})

    print("\n--- EDA Summary ---")
    print(result["eda_summary"])
    print(f"\nPlots saved to: {result['eda_plots']}")
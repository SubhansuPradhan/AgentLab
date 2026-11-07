"""
Feature Engineering Agent

Job: look at the cleaned dataset, decide the best cleaning and
preprocessing method for every column based on its actual
characteristics (missing %, skew, cardinality), apply those decisions
as a real sklearn pipeline, and explain the reasoning in plain
English.

Inputs (from state):
    - cleaned_df: pandas DataFrame from the Validation Agent
    - target_column: str, optional, excluded from preprocessing

Outputs (added to state):
    - feature_df: pandas DataFrame, fully numeric, ready for modeling
    - preprocessing_pipeline: the fitted sklearn ColumnTransformer
    - preprocessing_report: dict, per-column decisions and reasons
    - preprocessing_summary: str, LLM-written explanation
"""
from typing import Any, Dict

import pandas as pd
import re
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from agents.base_agent import BaseAgent
from prompts.feature_engineering_prompt import build_feature_engineering_prompt
from config.settings import settings


def _clean_feature_names(names):
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


class FeatureEngineeringAgent(BaseAgent):
    name = "feature_engineering_agent"


class FeatureEngineeringAgent(BaseAgent):
    name = "feature_engineering_agent"

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = state["cleaned_df"]
        target_column = state.get("target_column")

        recommendations = self._decide_strategy(df, target_column)
        feature_df, pipeline = self._apply_strategy(df, recommendations, target_column)
        summary = self._summarize(recommendations)

        self.log(f"processed {len(recommendations)} columns, output shape {feature_df.shape}")

        return {
            "feature_df": feature_df,
            "preprocessing_pipeline": pipeline,
            "preprocessing_report": recommendations,
            "preprocessing_summary": summary,
        }

    def _decide_strategy(self, df: pd.DataFrame, target_column: str) -> Dict[str, Dict[str, Any]]:
        recommendations = {}
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        for col in numeric_cols:
            if col == target_column:
                continue
            missing_pct = df[col].isnull().mean()
            skew = df[col].skew() if df[col].notna().sum() > 1 else 0

            if missing_pct == 0:
                impute, impute_reason = "none", "no missing values"
            elif missing_pct < 0.05:
                impute, impute_reason = "median", f"only {missing_pct:.1%} missing, median is a safe simple fill"
            else:
                impute, impute_reason = "median", (
                    f"{missing_pct:.1%} missing is significant, median chosen over mean "
                    f"because it resists distortion from outliers"
                )

            if abs(skew) > 1:
                scale, scale_reason = "RobustScaler", (
                    f"skew of {skew:.2f} means the column has extreme values, "
                    f"RobustScaler uses median/IQR so those extremes don't dominate the scaling"
                )
            else:
                scale, scale_reason = "StandardScaler", f"skew of {skew:.2f} is close to normal, standard scaling is fine"

            recommendations[col] = {
                "type": "numeric", "missing_pct": round(float(missing_pct), 3), "skew": round(float(skew), 2),
                "impute_strategy": impute, "impute_reason": impute_reason,
                "scale_strategy": scale, "scale_reason": scale_reason,
            }

        for col in categorical_cols:
            if col == target_column:
                continue
            cardinality = df[col].nunique()
            missing_pct = df[col].isnull().mean()
            row_ratio = cardinality / len(df)

            if row_ratio > 0.9:
                encode, encode_reason = "drop", (
                    f"{cardinality} unique values across {len(df)} rows means this is "
                    f"almost certainly an identifier, not a real feature, encoding it "
                    f"would let a model memorize rows instead of learning a pattern"
                )
            elif cardinality <= 10:
                encode, encode_reason = "one-hot", f"only {cardinality} categories, one-hot encoding keeps this simple and safe"
            else:
                encode, encode_reason = "frequency", (
                    f"{cardinality} categories is too many for one-hot without exploding "
                    f"column count, frequency encoding keeps it compact"
                )

            recommendations[col] = {
                "type": "categorical", "missing_pct": round(float(missing_pct), 3), "cardinality": int(cardinality),
                "impute_strategy": "most_frequent" if missing_pct > 0 else "none",
                "impute_reason": f"{missing_pct:.1%} missing, fill with the most common category" if missing_pct > 0 else "no missing values",
                "encode_strategy": encode, "encode_reason": encode_reason,
            }

        return recommendations

    def _apply_strategy(self, df: pd.DataFrame, recommendations: Dict[str, Dict[str, Any]], target_column: str):
        keep_numeric = [c for c, r in recommendations.items() if r["type"] == "numeric"]
        onehot_cat = [c for c, r in recommendations.items() if
                      r["type"] == "categorical" and r["encode_strategy"] == "one-hot"]
        freq_cat = [c for c, r in recommendations.items() if
                    r["type"] == "categorical" and r["encode_strategy"] == "frequency"]
        dropped = [c for c, r in recommendations.items() if
                   r["type"] == "categorical" and r["encode_strategy"] == "drop"]

        work_df = df.drop(columns=dropped, errors="ignore").copy()
        target_series = work_df[target_column] if target_column in work_df.columns else None

        # Only columns explicitly classified above ever reach the transformer.
        # Previously the target column (and anything else unclassified) fell
        # through remainder="passthrough" untouched. Even one such column
        # forces the WHOLE output matrix to object dtype, which is what
        # crashed XGBoost downstream even on columns that were fine on their own.
        feature_cols = keep_numeric + onehot_cat + freq_cat
        work_df = work_df[feature_cols] if feature_cols else work_df.drop(columns=[target_column], errors="ignore")

        for col in freq_cat:
            freq_map = work_df[col].value_counts(normalize=True)
            work_df[col] = work_df[col].map(freq_map)

        transformers = []
        if keep_numeric:
            transformers.append(("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), keep_numeric))
        if onehot_cat:
            transformers.append(("cat_onehot", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), onehot_cat))
        if freq_cat:
            transformers.append(("cat_freq", SimpleImputer(strategy="median"), freq_cat))

        if not transformers:
            feature_df, pipeline = work_df.copy(), None
        else:
            pipeline = ColumnTransformer(transformers, remainder="drop")
            transformed = pipeline.fit_transform(work_df)
            try:
                feature_names = pipeline.get_feature_names_out()
            except Exception:
                feature_names = [f"feature_{i}" for i in range(transformed.shape[1])]
            feature_names = _clean_feature_names(feature_names)
            feature_df = pd.DataFrame(transformed, columns=feature_names, index=work_df.index)

        if target_series is not None:
            feature_df[target_column] = target_series.values

        return feature_df, pipeline

    def _summarize(self, recommendations: Dict[str, Dict[str, Any]]) -> str:
        try:
            import ollama
            prompt = build_feature_engineering_prompt(recommendations)
            response = ollama.chat(
                model=settings.OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                think=False,
            )
            return response["message"]["content"].strip()
        except Exception as e:
            self.log(f"Ollama call failed ({e}), falling back to template")
            return self._template_summary(recommendations)

    def _template_summary(self, recommendations: Dict[str, Dict[str, Any]]) -> str:
        n_dropped = sum(1 for r in recommendations.values() if r.get("encode_strategy") == "drop")
        n_imputed = sum(1 for r in recommendations.values() if r.get("impute_strategy", "none") != "none")
        lines = [f"Processed {len(recommendations)} columns.", f"{n_imputed} needed imputation."]
        if n_dropped:
            lines.append(f"{n_dropped} column(s) dropped as likely identifiers.")
        return " ".join(lines)


if __name__ == "__main__":
    import sys
    df = pd.read_csv(sys.argv[1])
    agent = FeatureEngineeringAgent()
    result = agent.run({"cleaned_df": df})
    print("\n--- Preprocessing Summary ---")
    print(result["preprocessing_summary"])
    print(f"\nOutput shape: {result['feature_df'].shape}")
    print(result["feature_df"].head())
"""
Prompt templates live separately from agent logic so you can iterate
on wording without touching the Python code that calls the LLM.
"""
from typing import Any, Dict, List


def build_explainability_prompt(
    task_type: str,
    model_name: str,
    target_phrase: str,
    performance: Dict[str, Any],
    reliability_note: str,
    importance_rows: List[Dict[str, Any]],
    top1_share: float,
    top3_share: float,
    local_example: Dict[str, Any],
    n_rows: int,
) -> str:
    feature_lines = "\n".join(
        f"{r['rank']}. {r['feature']}: {r['share_pct']:.1f} percent of total importance, "
        f"direction = {r['direction']}"
        for r in importance_rows
    )

    perf_value = performance.get("value")
    perf_text = (
        f"{performance['metric']} = {perf_value:.3f}" if perf_value is not None
        else "could not be computed"
    )
    if perf_value is not None and performance.get("rank_corr") is not None:
        perf_text += (
            f" (rank correlation between predictions and actual values = "
            f"{performance['rank_corr']:.2f}, where 0 means no relationship)"
        )

    contrib_lines = "\n".join(
        f"- {c['feature']} (value after preprocessing: {c['value']:.2f}) "
        f"{'pushed the prediction UP' if c['shap'] > 0 else 'pushed the prediction DOWN'} "
        f"by {abs(c['shap']):.3f}"
        for c in local_example["top_contributions"]
    )

    if local_example.get("lime_available"):
        lime_text = (
            f"LIME, a second explanation method, agreed with SHAP on "
            f"{local_example['lime_agreement']} of the top 5 features for this example."
        )
    else:
        lime_text = "LIME was not available for this run."

    return f"""You are a senior data scientist explaining a trained model to
a colleague who has not looked at it yet. Write like you are talking
them through it, not filing a report. Plain language, short sentences,
but explain things properly.

Model explained: {model_name}
Problem type: {task_type}
What "up" and "down" mean below: they change {target_phrase}.
Rows used for the SHAP analysis: {n_rows}

Model quality check: {perf_text}
Reliability note: {reliability_note}

SHAP feature ranking (share of total importance, direction of effect):
{feature_lines}

Concentration: the top feature holds {top1_share:.1f} percent of all
importance, the top three together hold {top3_share:.1f} percent.

One example, the row the model got most wrong:
- Actual value: {local_example['actual']}
- Predicted value: {local_example['predicted']}
- Starting point before any feature is considered: {local_example['base_value']:.3f}
Main contributions for this row:
{contrib_lines}
{lime_text}

Write exactly four sections using these markdown headers, in this
order. Every section MUST contain real content, never leave one blank.

## What the Model Relies On
Start with one sentence explaining SHAP in plain words: it measures how
much each feature pushed a prediction up or down compared with an
average prediction. Then name the top three features and their shares,
and say whether the model leans on a few features or spreads its
attention widely. 3-4 sentences.

## How Each Feature Pushes Predictions
For the top three features, say in plain words what the direction means,
for example "higher values push the prediction up". If a direction is
mixed, say the effect depends on other factors or is not a simple
rise-or-fall. Do not invent effects that are not in the list above.
3-5 sentences.

## One Example Prediction
Walk through the example row: what the model predicted, what was
actually true, and which one or two features drove that prediction the
most. Note that this is the row the model got most wrong, so it shows a
weakness, not a typical case. Mention whether LIME agreed. 3-4 sentences.

## Anything to Watch
Be critical and direct. Use the reliability note as your anchor. If the
model is barely better than chance, say clearly that the rankings above
mostly reflect noise and should not be treated as real drivers. If the
score looks near perfect, or one feature dominates AND the score is
very high, say this can signal data leakage and that feature should be
checked before trusting the model. If one feature dominates but the
score is moderate, say the model leans heavily on that feature, which
is fine if it makes sense for the problem. If everything looks healthy,
say so and explain why in one sentence. 3-4 sentences.

Write the answer now, starting directly with "## What the Model Relies On".
No preamble, no closing remarks after the last section.
"""
"""
Prompt templates live separately from agent logic so you can iterate
on wording without touching the Python code that calls the LLM.
"""
from typing import Any, Dict, Optional


def build_model_selection_prompt(
    task_type: str,
    target_description: str,
    metrics: Dict[str, Dict[str, Any]],
    stats_table: Dict[str, Dict[str, float]],
    recommendation: Dict[str, Any],
    diagnostics: Dict[str, Any],
    plan: Dict[str, Any],
    tuning: Optional[Dict[str, Any]],
) -> str:
    nan = float("nan")

    if task_type == "classification":
        score_note = "The cross-validated score is F1 (weighted), between 0 and 1. Higher is better."
    else:
        score_note = (
            "The cross-validated score is negative RMSE, so a value closer to zero means smaller "
            "prediction error. A higher (less negative) score is better."
        )

    model_lines = []
    for name, vals in metrics.items():
        t = stats_table.get(name, {})
        if task_type == "classification":
            extra = f"test F1 {t.get('F1', nan):.3f}, test AUC-ROC {t.get('AUC_ROC', nan):.3f}"
        else:
            extra = f"test R2 {t.get('R2', nan):.3f}, test MAE {t.get('MAE', nan):.2f}"
        model_lines.append(
            f"- {name}: cross-validated score {vals['mean']:.4f} (+/- {vals['std']:.4f}), {extra}"
        )
    model_text = "\n".join(model_lines)

    rec_reasons = "\n".join(f"- {r}" for r in recommendation["reasons"])
    within = ", ".join(recommendation["within_noise"])

    plan_reasons = "\n".join(f"- {r}" for r in plan["reasons"])
    if plan["required"]:
        setting_lines = "\n".join(
            f"- {p['name']} (searched from {p['range']}): {p['meaning']}" for p in plan["params"]
        )
        plan_text = (
            f"Fine-tuning required: YES\n"
            f"Search focus: {plan['focus']}\n"
            f"Why it is required:\n{plan_reasons}\n"
            f"Settings that were searched:\n{setting_lines}"
        )
    else:
        plan_text = (
            f"Fine-tuning required: NO\n"
            f"Why it is not required:\n{plan_reasons}"
        )

    if tuning:
        c = tuning["comparison"]
        importance = ", ".join(f"{k} ({v:.2f})" for k, v in tuning["param_importance"].items()) or "not available"
        outcome_text = (
            f"Tuning was run ({tuning['n_trials']} Optuna trials).\n"
            f"- Default settings: cross-validated {c['baseline_cv']:.4f}, test {c['baseline_test']:.4f}\n"
            f"- Tuned settings: cross-validated {c['tuned_cv']:.4f}, test {c['tuned_test']:.4f}\n"
            f"- Test score change: {c['test_gain']:+.4f} ({c['relative_test_gain_pct']:+.2f} percent)\n"
            f"- Automatic verdict: {c['verdict']}\n"
            f"- Tuned model adopted: {'YES' if tuning['adopted'] else 'NO, the default settings are kept'}\n"
            f"- Best settings found: {tuning['best_params']}\n"
            f"- Which settings mattered most: {importance}"
        )
        if c["cv_test_mismatch"]:
            outcome_text += (
                "\n- The tuned model looked better in cross validation but NOT on the untouched "
                "test set, which usually means the search fit the validation folds by chance."
            )
    else:
        outcome_text = "No tuning was run."

    return f"""You are a senior data scientist briefing a colleague on which
model to use for their dataset, and whether fine-tuning is worth doing.
Write like you are walking them through your actual reasoning, not
filing a status update. Plain language, no jargon dump, but explain
things properly.

Problem type: {task_type}
{target_description}
{score_note}

Models compared (5-fold cross validation on the training split, then
scored once on a test set the models never saw):
{model_text}

Recommended model: {recommendation['model']}
Highest raw score: {recommendation['best_by_score']}
Models within one standard error ({recommendation['std_error']:.4f}) of the highest: {within}
How the recommendation was reached:
{rec_reasons}

Quality check: {diagnostics['reliability_note']}
Train versus test: the recommended model scores {diagnostics['train_score']:.3f} on the
data it trained on and {diagnostics['test_score']:.3f} on unseen data (gap {diagnostics['train_test_gap']:.3f}).
Training rows: {diagnostics['n_train']}

{plan_text}

{outcome_text}

Write exactly five sections using these markdown headers, in this
order. Every section MUST contain real content, never leave one blank.

## Problem Type
2-3 sentences. Say whether this is classification or regression and what
"good performance" means for it: for classification, F1 balances catching
both classes instead of just favouring the majority class; for
regression, the error is in the target's own units.

## Model Comparison
4-5 sentences. Name every model with its score. Say whether the top
models are separated by more than the noise level or are effectively
tied, and what that means in practice: a close race where several would
work, or a clear win for one.

## Recommended Model
4-5 sentences. Name the recommended model and explain why, using the
reasoning given above and not inventing new reasons. Explain what kind
of model it is: linear models (logistic or linear regression) suit
simple, additive relationships; tree models (random forest, XGBoost,
LightGBM) suit non-linear patterns and interactions between features.
If the recommended model is not the one with the highest raw score,
say plainly that it is simpler and within the noise of the best, so
there is no evidence the more complex model is better. If the quality
check says the model is barely better than chance, say plainly that no
model is truly good on this data and the recommendation only names the
least bad option.

## Fine-Tuning Recommendation
5-7 sentences. Start with a clear statement that fine-tuning is
required or not required, matching the facts above exactly.
- If NOT required: explain each reason in plain words and say what to do
  instead, for example fix a leaking column, improve the features, or
  collect more data. Do not suggest tuning anyway.
- If required: explain which settings were searched and why those ones,
  using this guide: max_depth or num_leaves is how detailed each tree can
  get; learning_rate with n_estimators is slow careful fitting versus fast
  coarse fitting; min_samples_leaf, min_samples_split or min_child_samples
  force more cautious splits; C (logistic regression) is regularization,
  where a smaller value means a simpler boundary; subsample and
  colsample_bytree below 1.0 add randomness that guards against
  overfitting. Then report what happened: default versus tuned test score,
  the size of the change, and the automatic verdict. A change of a fraction
  of a percent is noise and must not be described as an achievement. Say
  which model is used going forward. Never contradict the automatic verdict.

## Anything to Watch
3-5 sentences. Be critical and direct. Use the quality check as your
anchor. If the score is near perfect, say it may signal data leakage and
the features should be checked first. If the model is barely better than
chance, say the problem is the features, not the model. If the gap
between train and test is large, explain it means overfitting. If the
training set is small, say results may not be stable. If nothing is
concerning, say so and explain why in one sentence.

Write the answer now, starting directly with "## Problem Type". No
preamble, no closing remarks after the last section.
"""
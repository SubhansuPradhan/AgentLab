"""
Prompt templates live separately from agent logic so you can iterate
on wording without touching the Python code that calls the LLM.
"""
from typing import Any, Dict, List


def build_hypothesis_prompt(
    target: str,
    task_type: str,
    target_description: str,
    n_rows: int,
    hypotheses: List[Dict[str, Any]],
    screening_top: List[Dict[str, Any]],
    n_screened: int,
    leak_suspects: List[str],
    excluded: Dict[str, str],
    eda_context: str,
) -> str:
    if task_type == "classification":
        score_meaning = "AUC-ROC, where 0.50 means no relationship and 1.00 means a perfect prediction"
    else:
        score_meaning = "R2, where 0 means no relationship and 1.00 means a perfect prediction"

    hypothesis_lines = "\n".join(
        f"{h['id']} (type: {h['type']}): {h['claim']}\n"
        f"    Check after modeling: {h['check']['description']}"
        for h in hypotheses
    ) or "(none)"

    screen_lines = ", ".join(
        f"{r['feature']} {r['solo_score']:.2f}" for r in screening_top
    ) or "(none)"

    leak_text = ", ".join(leak_suspects) if leak_suspects else "none found"
    excluded_text = (
        "; ".join(f"{col} ({reason})" for col, reason in excluded.items())
        if excluded else "none"
    )

    return f"""You are a senior data scientist planning an analysis with a
colleague BEFORE any model is trained. Write like you are talking them
through your thinking, not filing a report. Plain language, short
sentences, but explain things properly.

Target: '{target}' ({task_type})
{target_description}
Rows: {n_rows}
Method: each feature was tested on its own against the target, using
cross validation, and scored with {score_meaning}.
Features tested: {n_screened}
Best single-feature scores: {screen_lines}

Hypotheses built from that evidence. The numbers and directions are
computed, so never change them or add new ones:
{hypothesis_lines}

Suspected data leaks: {leak_text}
Columns left out of testing: {excluded_text}
Context from the exploratory analysis: {eda_context}

Write exactly four sections using these markdown headers, in this
order. Every section MUST contain real content, never leave one blank.

## Target and Data
2-3 sentences. Say what is being predicted and what kind of problem it
is. For classification, mention the class balance. Say how many features
were testable and how many were left out, and whether that matters.

## Hypotheses to Test
Go through every hypothesis in order (H1, H2, and so on). For each one
write exactly three short parts on separate lines:
- The hypothesis in plain words. Keep every number and direction exactly
  as given.
- "Plausible reason:" one hedged sentence on why this could be true in
  the real world. This is a guess, not a finding. If the column name does
  not make the meaning clear, write that the meaning is unclear instead
  of inventing a reason.
- "Check:" one sentence on what should show up after modeling.
For a leak hypothesis, explain that a column only known after the outcome
cannot be used to predict that outcome. For a hypothesis of type
weak_signal, explain that a model would then score close to chance.

## Red Flags
Name any suspected leaks and any columns left out, with the reason. Then
add one sentence saying this screen tests one feature at a time, so it
cannot see features that only matter in combination, and that a
relationship here is an association, not proof of cause. If there are no
leaks and no excluded columns, say so plainly first.

## What Modeling Should Show
2-3 sentences. Say what result would support these hypotheses and what
would contradict them, for example which features should lead the SHAP
ranking, or that a near-chance score would fit a weak-signal hypothesis.
Use the context from the exploratory analysis if it is relevant, for
example a correlated feature that may share importance with a driver.

Write the answer now, starting directly with "## Target and Data". No
preamble, no closing remarks after the last section.
"""
"""
Prompt templates live separately from agent logic so you can iterate
on wording without touching the Python code that calls the LLM.
"""
from typing import Any, Dict, List, Optional


def build_critic_prompt(
    task_type: str,
    verdict: str,
    mode: str,
    revision_round: int,
    max_revisions: int,
    limit_reached: bool,
    model_line: str,
    performance_line: str,
    flags: List[Dict[str, Any]],
    plan: Optional[Dict[str, Any]],
    checks_run: List[str],
    scorecard: Optional[Dict[str, int]],
) -> str:
    if flags:
        flag_lines = []
        for f in flags:
            if f.get("fix"):
                fix = (
                    f"Automatic fix: drop {', '.join(repr(c) for c in f['fix']['columns'])} "
                    f"and re-run from feature engineering."
                )
            else:
                fix = "No automatic fix, a person has to decide."
            flag_lines.append(f"- [{f['severity'].upper()}] {f['title']}: {f['evidence']} {fix}")
        flag_text = "\n".join(flag_lines)
    else:
        flag_text = "(no problems found)"

    if plan:
        plan_text = (
            f"The pipeline will drop {', '.join(repr(c) for c in plan['columns'])} and run again "
            f"(revision {plan['round']} of at most {max_revisions})."
        )
    else:
        plan_text = "The pipeline will not be sent back."

    if mode == "audit_only":
        mode_text = (
            "The critic is switched OFF for this run (audit only). It recorded what it would have flagged "
            "but cannot change the pipeline, so the verdict is always approve."
        )
    else:
        mode_text = "The critic is ACTIVE and can send the pipeline back."

    limit_text = (
        "The revision limit has been reached, so problems that could still be fixed are being passed "
        "on as warnings instead."
        if limit_reached else "The revision limit has not been reached."
    )
    score_text = f"Hypothesis check against SHAP: {scorecard}" if scorecard else "Hypothesis check: not available"
    checks_text = ", ".join(checks_run)

    return f"""You are the independent reviewer of an automated machine learning
experiment, a skeptical senior data scientist who signs off before the
report is written. The checks below have ALREADY been run by code and the
verdict has ALREADY been decided. Your job is only to explain it in plain
words. Do not add findings that are not listed, do not change the verdict,
and do not soften a problem that is listed. Write like you are talking to
a colleague, short sentences, no jargon dump.

Problem type: {task_type}
Model reviewed: {model_line}
Independent re-measurement: {performance_line}
{score_text}
Checks that were run: {checks_text}
Critic mode: {mode_text}
Revision round: {revision_round} of at most {max_revisions}. {limit_text}

VERDICT: {verdict}
What happens next: {plan_text}

Problems found, most serious first:
{flag_text}

Write exactly four sections using these markdown headers, in this
order. Every section MUST contain real content, never leave one blank.

## Verdict
2-3 sentences. State the verdict in plain words using the meaning below,
and the single biggest reason for it.
- approve: nothing that needs a caveat was found (minor notes at most), the
  results can be reported.
- approve_with_warnings: the results go into the report, but with problems
  that a reader must be told about.
- revise: a fixable problem was found, so the pipeline is sent back.

## What the Checks Found
3-6 sentences. Go through the listed problems from most to least serious
and say what each means in practice, using the numbers given. For a
suspected leak, say the model is probably reading the answer from that
column. For an identifier column, say the model may be memorizing rows
instead of learning a pattern. For no signal, say the model is close to
guessing and the cause is the features, not the model. For overfitting,
say the model does much worse on new data than on data it has seen. If
the list is empty, say plainly that nothing was found and name the kinds
of problems that were checked.

## What Happens Next
2-3 sentences. Say exactly what happens next, matching the facts above.
If the pipeline is sent back, name the columns being dropped and say the
critic will review the new result. If it is not, say what goes into the
report. If the critic is OFF, say the problems listed were NOT acted on and
the results shipped as they were.

## What the Report Must Say
2-4 sentences. State the caveats a reader of the final report needs, one
per serious problem, in plain words. Never describe a result as reliable
if a serious problem is listed. If there are no problems, say the results
passed every check and name the one thing the checks cannot see: features
that only matter in combination, and whether the data itself is correct.

Write the answer now, starting directly with "## Verdict". No preamble, no
closing remarks after the last section.
"""
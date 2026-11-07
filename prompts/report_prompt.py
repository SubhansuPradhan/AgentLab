"""
Prompt templates live separately from agent logic so you can iterate
on wording without touching the Python code that calls the LLM.

The Report Agent only asks the LLM to write two sections: the Executive
Summary and the Conclusion. Every other section (data, hypotheses,
model performance, explainability, the critic's review) is the
specialist agent's own already-written summary, copied in verbatim.
That keeps the numbers in the report identical to the numbers each
agent actually verified, instead of a second model re-describing them
and risking a contradiction.
"""
from typing import Any, Dict


def build_report_prompt(facts: Dict[str, Any]) -> str:
    return f"""You are a senior data scientist writing the opening and
closing of a finished experiment report. The body of the report is
already written by the specialists who ran each step; you are not
shown their full text, only the facts below. Do not invent any number,
column name, or finding that is not listed here.

Dataset: {facts['dataset_name']} ({facts['n_rows']} rows)
Target: '{facts['target']}' ({facts['task_type']})
Recommended model: {facts['model_name']} ({facts['model_choice']} settings)
Headline score: {facts['headline_metric']} = {facts['headline_value']}
Fine-tuning: {facts['tuning_line']}
Top feature driving predictions: {facts['top_feature']} ({facts['top_feature_share']} percent of importance)
Hypotheses: {facts['n_hypotheses']} proposed before modeling, outcome: {facts['hypothesis_scoreline']}
Critic's verdict: {facts['critic_verdict']}
Serious problems the critic found: {facts['critic_flag_summary']}
Critic mode: {facts['critic_mode']}

Write exactly two sections using these markdown headers, in this order.
Every section MUST contain real content, never leave one blank.

## Executive Summary
4-6 sentences for someone who will only read this section. State what
was predicted, which model was used and how well it performed, whether
fine-tuning helped, and the critic's verdict in plain words. If the
verdict is anything other than a clean approval, say so here, do not
bury it. Do not describe the result as reliable if the critic found a
serious problem.

## Conclusion and Recommendations
4-6 sentences. Summarize what the experiment actually showed, name the
single most important caveat a reader should keep in mind, and
recommend one concrete next step (for example: fix a flagged column
and re-run, collect more data, or that the result is ready to use as
is). Do not recommend something contradicted by the critic's verdict.

Write the answer now, starting directly with "## Executive Summary".
No preamble, no closing remarks after the last section.
"""


def build_feature_engineering_prompt(recommendations: dict) -> str:
    return f"""You are a senior data scientist briefing a colleague on
cleaning and preprocessing decisions you made on their dataset. Explain
your reasoning like you're talking them through it, not filing a report.

Here are the decisions made, per column:

{recommendations}

Cover these things, in this order:

1. Summarize how many columns needed imputation and what strategy was
used most often, and why that strategy fits this kind of data.

2. Call out any column that was dropped entirely, and explain plainly
why (usually because it looked like an identifier, not a real feature).

3. Call out any column that needed special handling because of skew or
high cardinality, and explain what would have gone wrong if it had been
processed with the default, simplest method instead.

4. If nothing unusual came up at all, say so directly, a dataset that
needed no special handling is still worth stating plainly.

FORMAT THE OUTPUT EXACTLY LIKE THIS using these three markdown headers,
in this order, one short paragraph under each, 2-3 sentences max:

## What Was Done
## Notable Decisions
## Anything to Watch

Do not add sections beyond these three. Do not restate every column,
pick out only what's actually worth saying out loud. Write the summary
now, starting directly with "## What Was Done", no preamble.
"""
def build_eda_prompt(stats: dict) -> str:
    return f"""You are a senior data scientist handing off exploratory
analysis findings to a colleague who is about to build a model on this
data. Write like you're actually talking to them, not writing a report
for a stranger. Plain language, no jargon dump, no robotic tone within
each section.

Here's what the analysis found:

Numeric columns: {stats['numeric_columns']}
Categorical columns: {stats['categorical_columns']}
Skewed columns (skew > 1, meaning a long tail of extreme values): {stats['skewed_columns']}
Highly correlated column pairs (>0.8, meaning these two columns move together almost identically): {stats['high_correlations']}
Categorical column breakdown (unique value counts, most common value and its share): {stats['categorical_summary']}
Numeric summary (mean, min, max, spread per column): {stats['numeric_summary']}

Cover these things, in this order:

1. Start with the shape of the data in one sentence: how many numeric
vs categorical columns, and whether that's a reasonable mix for
modeling or lopsided in some way worth mentioning.

2. Call out anything that looks like a genuine modeling RISK, not just
an observation. Specifically:
   - A correlation above 0.8 between two columns is not just "these
     are related," explain what it actually means: if both go into
     the model, one of them is probably redundant, or worse, if one
     of them is suspiciously close to being the target itself, that's
     a leakage risk and the model's results will look better than
     they really are.
   - A skewed column isn't just a statistics footnote, explain what
     happens if it's fed into a model untouched: a few extreme values
     can dominate what the model learns, so it likely needs a log
     transform or similar before training.
   - A categorical column where one value dominates (say, over 80-90%
     of rows) is a near-constant column in disguise, explain that it
     will carry very little useful signal even though it's not
     technically constant.

3. If none of the above risks are present, say so plainly and
explicitly, don't stay silent on it. "Nothing here raises a red flag"
is a real, useful finding, not a non-answer.

4. End with one practical, concrete next step the person should
actually take before modeling, tied to something specific you found,
not a generic "clean your data" statement.

FORMAT THE OUTPUT EXACTLY LIKE THIS, using these four section headers
as plain markdown (## headers), in this exact order, one short
paragraph under each, 2-3 sentences max per section, no sub-bullets
inside a section unless there are 3+ risks to list:

## Data Shape
(one short paragraph, point 1 above)

## Risks Found
(one short paragraph per risk found, point 2 above — if multiple
risks, separate them with a line break each, not a wall of text)

## Overall Read
(one sentence, point 3 above — state clearly whether this data looks
safe to model or not)

## Next Step
(one sentence, point 4 above — one concrete action, not a list)

Do not add any section beyond these four. Do not repeat information
across sections. Do not restate every number from the stats block,
pick out only what's actually worth saying out loud. Write the summary
now, starting directly with "## Data Shape", no preamble before it.
"""
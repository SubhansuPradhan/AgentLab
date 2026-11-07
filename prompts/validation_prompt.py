"""
Prompt templates live separately from agent logic so you can iterate
on wording without touching the Python code that calls the LLM.
"""
from typing import Any, Dict

def build_validation_prompt(stats: Dict[str, Any]) -> str:
    return f"""You are a senior data scientist handing off a data quality
check to a colleague who's about to start modeling. Write like you're
actually briefing them, not generating a formal report. Plain language,
no jargon dump.

Here's what the check found:

Rows: {stats['n_rows']}
Columns: {stats['n_columns']}
Duplicate rows found and removed: {stats['duplicate_rows']}
Columns with missing values (count and percent missing per column): {stats['missing_by_column']}
Constant columns, dropped automatically because they carry zero useful information: {stats['constant_columns']}
Columns that look like ID fields (too many unique values to be a real feature): {stats['high_cardinality_columns']}

Cover these things, in this order:

1. State the dataset's overall size and whether it's in good shape or
needs real cleanup before anyone should trust results from it.

2. If there are duplicate rows, say how many were removed and note
that duplicates can quietly inflate how good a model looks during
testing if they aren't caught early.

3. If any column has missing values, name the worst one specifically
(highest percentage) and explain what that percentage actually means
in practice: under 5% missing is usually a minor fix, 5-20% needs a
real imputation decision, above 20% means the column may not be
reliable enough to use at all. Don't just repeat the number, say what
it means for someone about to build a model.

4. If constant columns were dropped, name them and explain briefly why
a column with only one repeated value can't help a model, it gives
zero signal to learn from.

5. If any column looks like an ID field, name it and explain the risk
plainly: if it accidentally gets included as a feature, the model can
end up "memorizing" rows instead of learning a real pattern, which
makes results look better than they actually are.

6. If none of the above problems exist at all, say so directly and
plainly, don't stay silent on it. A clean dataset is a real finding
worth stating, not something to skip past.

FORMAT THE OUTPUT EXACTLY LIKE THIS, using these three section headers
as plain markdown (## headers), in this exact order, one short
paragraph under each, 2-3 sentences max per section:

## Overview
(point 1 above — size and overall condition)

## Issues Found
(points 2-5 above — one short paragraph per issue that actually
applies, skip any that don't apply, don't force empty commentary onto
a problem that isn't there)

## Ready to Model?
(point 6 above, plus a one-line verdict either way — state clearly
whether this data is ready to move to the next step or needs fixing
first)

Do not add any section beyond these three. Do not restate every
number from the stats block, pick out only what's actually worth
saying out loud. Write the summary now, starting directly with
"## Overview", no preamble before it.
"""
"""
Experiment Memory — Week 10

Lets later runs reference past ones, e.g. "a dataset with this shape
and these column types previously worked best with XGBoost." This is
what the original spec called "long-term / research memory," kept
here as a plain SQL lookup instead of a vector database, since a
simple structured query is enough at this scale and doesn't add a
dependency you'd need to debug for little benefit.

Build notes:
    - query_similar_runs(dataset_shape, column_types) -> list of past
      ExperimentRun rows with similar characteristics, simple SQL
      filtering is fine, no embeddings needed.
    - Feed the top match or two into the Model Selection Agent's
      prompt as extra context, not as a hard override, the agent
      should still run its own cross validation.
"""
from typing import Any, Dict, List

from database.db import get_session
from database.models import ExperimentRun


def save_run(run_data: Dict[str, Any]) -> None:
    session = get_session()
    try:
        session.add(ExperimentRun(**run_data))
        session.commit()
    finally:
        session.close()


def query_similar_runs(n_rows: int, n_columns: int) -> List[ExperimentRun]:
    raise NotImplementedError("Build this in week 10")

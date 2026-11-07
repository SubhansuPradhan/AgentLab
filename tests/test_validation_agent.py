"""
Test for the Validation Agent, the one agent that's actually built.
Add a test_<agent>.py file like this for every agent as you build it,
following this same pattern: build a small synthetic dataframe with a
known problem, run the agent, assert it caught the problem.
"""
import pandas as pd

from agents.validation_agent import ValidationAgent


def test_detects_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    agent = ValidationAgent()
    result = agent.run({"raw_df": df})
    assert result["validation_report"]["duplicate_rows"] == 1
    assert len(result["cleaned_df"]) == 2


def test_detects_missing_values():
    df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
    agent = ValidationAgent()
    result = agent.run({"raw_df": df})
    assert "a" in result["validation_report"]["missing_by_column"]


def test_drops_constant_columns():
    df = pd.DataFrame({"a": [1, 1, 1], "b": [4, 5, 6]})
    agent = ValidationAgent()
    result = agent.run({"raw_df": df})
    assert "a" not in result["cleaned_df"].columns


def test_summary_is_nonempty_string():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    agent = ValidationAgent()
    result = agent.run({"raw_df": df})
    assert isinstance(result["validation_summary"], str)
    assert len(result["validation_summary"]) > 0

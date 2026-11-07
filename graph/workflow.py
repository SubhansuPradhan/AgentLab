"""
AgentLab workflow: wires every agent into one LangGraph pipeline.

    validate -> eda -> hypothesis -> feature_engineering -> model_selection
        -> explainability -> verify_hypotheses -> critic
                                                   |-- revise --> apply_revision --> feature_engineering
                                                   |-- else   --> report -> END

The critic is the only branch. When it finds a fixable problem (a leaked
column, an identifier used as a feature) it routes through apply_revision,
which drops the columns, and the pipeline runs again from feature
engineering. Revisions are capped by max_revisions, so it always ends.

Use:
    from graph.workflow import run_pipeline
    state = run_pipeline(raw_df, "Profit", dataset_name="superstore")
    print(state["report_path"], state["critique"]["verdict"])

Command line:
    python -m graph.workflow data.csv target_column
    python -m graph.workflow data.csv target_column --no-critic
    python -m graph.workflow data.csv target_column --ablation
    python -m graph.workflow --diagram
"""
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Tuple

import pandas as pd
from langgraph.graph import END, START, StateGraph

from agents.critic_agent import CriticAgent
from agents.eda_agent import EDAAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.hypothesis_agent import HypothesisAgent
from agents.model_selection_agent import ModelSelectionAgent
from agents.report_agent import ReportAgent
from agents.validation_agent import ValidationAgent
from graph.state import AgentLabState

NODES_PER_PASS = 9          # validate ... report, one clean run through
NODES_PER_REVISION = 6      # apply_revision + feature_engineering ... critic


def _timed(name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]):
    """
    Wrap an agent method as a graph node: time it, record it in step_log
    (the Streamlit monitor reads this), and name the failing step if it
    raises, so an error says where the pipeline stopped.
    """
    def node(state: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        try:
            update = fn(state) or {}
        except Exception as e:
            raise RuntimeError(f"Pipeline stopped in the '{name}' step: {type(e).__name__}: {e}") from e
        entry = {
            "step": name,
            "seconds": round(time.time() - start, 2),
            "revision": state.get("revision_count", 0),
        }
        return {**update, "step_log": list(state.get("step_log", [])) + [entry]}
    return node


def route_after_critic(state: Dict[str, Any]) -> str:
    """The only branch in the graph."""
    return "apply_revision" if state.get("loop_back_to") == "feature_engineering" else "report"


def build_graph():
    validation, eda = ValidationAgent(), EDAAgent()
    hypothesis, features = HypothesisAgent(), FeatureEngineeringAgent()
    modeling, explain = ModelSelectionAgent(), ExplainabilityAgent()
    critic, report = CriticAgent(), ReportAgent()

    g = StateGraph(AgentLabState)
    g.add_node("validate", _timed("validate", validation.run))
    g.add_node("eda", _timed("eda", eda.run))
    g.add_node("hypothesis", _timed("hypothesis", hypothesis.run))
    g.add_node("feature_engineering", _timed("feature_engineering", features.run))
    g.add_node("model_selection", _timed("model_selection", modeling.run))
    g.add_node("explainability", _timed("explainability", explain.run))
    g.add_node("verify_hypotheses", _timed("verify_hypotheses", hypothesis.verify))
    g.add_node("critic", _timed("critic", critic.run))
    g.add_node("apply_revision", _timed("apply_revision", critic.apply_revision))
    g.add_node("report", _timed("report", report.run))

    g.add_edge(START, "validate")
    g.add_edge("validate", "eda")
    g.add_edge("eda", "hypothesis")
    g.add_edge("hypothesis", "feature_engineering")
    g.add_edge("feature_engineering", "model_selection")
    g.add_edge("model_selection", "explainability")
    g.add_edge("explainability", "verify_hypotheses")
    g.add_edge("verify_hypotheses", "critic")
    g.add_conditional_edges(
        "critic", route_after_critic,
        {"apply_revision": "apply_revision", "report": "report"},
    )
    g.add_edge("apply_revision", "feature_engineering")
    g.add_edge("report", END)
    return g.compile()


def _initial_state(raw_df: pd.DataFrame, target_column: str, dataset_name: str, run_id: str,
                   critic_enabled: bool, max_revisions: int, n_trials: int,
                   timeout_seconds: int) -> Dict[str, Any]:
    if target_column not in raw_df.columns:
        raise KeyError(f"Target column '{target_column}' not found. Columns: {list(raw_df.columns)}")
    return {
        "raw_df": raw_df, "target_column": target_column, "dataset_name": dataset_name,
        "run_id": run_id, "critic_enabled": critic_enabled, "max_revisions": max_revisions,
        "revision_count": 0, "n_trials": n_trials, "timeout_seconds": timeout_seconds, "step_log": [],
    }


def _config(max_revisions: int) -> Dict[str, Any]:
    # LangGraph stops after a fixed number of steps. Allow for every possible
    # revision, plus headroom, so a legitimate run is never cut off.
    return {"recursion_limit": NODES_PER_PASS + NODES_PER_REVISION * max_revisions + 10}


def _make_run_id(dataset_name: str) -> str:
    return f"{dataset_name}_{datetime.now():%Y%m%d_%H%M%S}"


def run_pipeline(raw_df: pd.DataFrame, target_column: str, dataset_name: str = "dataset",
                 run_id: str = None, critic_enabled: bool = True, max_revisions: int = 2,
                 n_trials: int = 20, timeout_seconds: int = 300) -> Dict[str, Any]:
    """Run the whole pipeline and return the final state."""
    run_id = run_id or _make_run_id(dataset_name)
    state = _initial_state(raw_df, target_column, dataset_name, run_id, critic_enabled,
                           max_revisions, n_trials, timeout_seconds)
    return build_graph().invoke(state, config=_config(max_revisions))


def stream_pipeline(raw_df: pd.DataFrame, target_column: str, dataset_name: str = "dataset",
                    run_id: str = None, critic_enabled: bool = True, max_revisions: int = 2,
                    n_trials: int = 20, timeout_seconds: int = 300) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """
    Yield (node_name, update) as each step finishes, for live progress in the
    Streamlit app. Merge the updates yourself if you need the final state.
    """
    run_id = run_id or _make_run_id(dataset_name)
    state = _initial_state(raw_df, target_column, dataset_name, run_id, critic_enabled,
                           max_revisions, n_trials, timeout_seconds)
    for event in build_graph().stream(state, config=_config(max_revisions), stream_mode="updates"):
        for node, update in event.items():
            yield node, update


def run_ablation(raw_df: pd.DataFrame, target_column: str, dataset_name: str = "dataset",
                 **options) -> Dict[str, Dict[str, Any]]:
    """
    The research experiment: the same dataset run once with the critic active
    and once with it off. Compare the two summaries. A higher score with the
    critic off is not a win if the critic's flags show that score came from a
    leak, so read the flags next to the numbers.
    """
    base = _make_run_id(dataset_name)
    results = {}
    for label, enabled in (("critic_on", True), ("critic_off", False)):
        state = run_pipeline(raw_df, target_column, dataset_name, run_id=f"{base}_{label}",
                             critic_enabled=enabled, **options)
        c = state["critique"]
        results[label] = {
            "state": state,
            "summary": {
                "verdict": c["verdict"],
                "model": state.get("model_name"),
                "metric": c["measured"].get("metric"),
                "score": c["measured"].get("value"),
                "flags": [f"{f['severity']}:{f['id']}" for f in c["flags"]],
                "revisions": state.get("revision_count", 0),
                "dropped_by_critic": state.get("dropped_by_critic", []),
                "seconds": round(sum(s["seconds"] for s in state["step_log"]), 1),
                "report": state.get("report_path"),
            },
        }
    return results


def export_diagram() -> str:
    """Mermaid source of the graph, ready to paste into a slide or README."""
    return build_graph().get_graph().draw_mermaid()


def _print_run(state: Dict[str, Any]) -> None:
    c = state["critique"]
    print("\nSteps:")
    for s in state["step_log"]:
        print(f"  {s['step']:20s} {s['seconds']:7.1f}s   (revision round {s['revision']})")
    print(f"\nCritic verdict: {c['verdict']} ({c['mode']})")
    for f in c["flags"]:
        print(f"  [{f['severity']}] {f['title']}")
    if state.get("dropped_by_critic"):
        print(f"Columns dropped by the critic: {state['dropped_by_critic']}")
    print(f"\nReport saved to: {state['report_path']}")


def _main() -> None:
    p = argparse.ArgumentParser(description="Run the AgentLab pipeline.")
    p.add_argument("csv", nargs="?", help="path to the dataset")
    p.add_argument("target", nargs="?", help="column to predict")
    p.add_argument("--no-critic", action="store_true", help="audit only, the critic never changes the pipeline")
    p.add_argument("--ablation", action="store_true", help="run with the critic on and off and compare")
    p.add_argument("--diagram", action="store_true", help="print the Mermaid diagram and exit")
    p.add_argument("--max-revisions", type=int, default=2)
    args = p.parse_args()

    if args.diagram:
        print(export_diagram())
        return
    if not args.csv or not args.target:
        p.error("csv and target are required unless --diagram is used")

    df = pd.read_csv(args.csv)
    name = Path(args.csv).stem

    if args.ablation:
        out = run_ablation(df, args.target, name, max_revisions=args.max_revisions)
        print("\nAblation: same data, critic on versus off")
        for label, r in out.items():
            s = r["summary"]
            print(f"\n{label}\n  verdict: {s['verdict']}\n  model: {s['model']}\n  {s['metric']}: {s['score']}"
                  f"\n  flags: {s['flags']}\n  revisions: {s['revisions']}\n  dropped: {s['dropped_by_critic']}"
                  f"\n  seconds: {s['seconds']}\n  report: {s['report']}")
        return

    state = run_pipeline(df, args.target, name, critic_enabled=not args.no_critic,
                         max_revisions=args.max_revisions)
    _print_run(state)


if __name__ == "__main__":
    _main()
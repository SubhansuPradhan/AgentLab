"""
Shared state passed between every node in the AgentLab graph.

IMPORTANT: LangGraph silently DROPS any key a node returns that is not
declared here. There is no error, the value just disappears. So whenever
an agent starts returning a new key, add it to this file, or the next
agent will not see it.

total=False because the state fills up as the pipeline runs, so most keys
do not exist yet at the start.
"""
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd


class AgentLabState(TypedDict, total=False):
    # ---- supplied by the caller at the start of a run ----
    raw_df: pd.DataFrame
    dataset_name: str
    target_column: str
    run_id: str
    critic_enabled: bool            # False switches the critic to audit-only (ablation)
    max_revisions: int
    n_trials: int                   # Optuna trials for the tuning step
    timeout_seconds: int            # tuning time limit
    shap_time_budget_seconds: int

    # ---- Validation Agent ----
    cleaned_df: pd.DataFrame        # also rewritten by the critic's apply_revision
    validation_report: Dict[str, Any]
    validation_summary: str

    # ---- EDA Agent ----
    eda_stats: Dict[str, Any]
    eda_plots: List[str]
    eda_summary: str

    # ---- Hypothesis Agent: run() ----
    hypotheses: List[Dict[str, Any]]
    feature_screening: List[Dict[str, Any]]
    leakage_suspects: List[str]
    excluded_columns: Dict[str, str]
    target_info: Dict[str, Any]
    hypothesis_summary: str

    # ---- Feature Engineering Agent ----
    feature_df: pd.DataFrame
    preprocessing_pipeline: Any
    preprocessing_report: Dict[str, Any]
    preprocessing_summary: str

    # ---- Model Selection and Tuning Agent ----
    task_type: str
    model_name: str
    trained_model: Any
    tuned_model: Any
    final_model: Any
    final_model_choice: str
    model_recommendation: Dict[str, Any]
    tuning_plan: Dict[str, Any]
    tuning_ran: bool
    best_params: Dict[str, Any]
    tuning_history: List[Dict[str, Any]]
    tuning_comparison: Optional[Dict[str, Any]]
    param_importance: Dict[str, float]
    tuning_plot_path: Optional[str]
    model_metrics: Dict[str, Any]
    stats_table: Dict[str, Any]
    stats_table_str: str
    comparison_table: Dict[str, Any]
    comparison_table_str: str
    confusion_matrix_path: Optional[str]
    label_mapping: Optional[Dict[int, str]]
    train_test_split_info: Dict[str, Any]
    model_selection_summary: str

    # ---- Explainability Agent ----
    shap_values: Any
    shap_sample: Any
    feature_importance: List[Dict[str, Any]]
    shap_plots: List[str]
    local_example: Dict[str, Any]
    explainability_flags: Dict[str, Any]
    focus_class: Optional[str]
    explainability_summary: str

    # ---- Hypothesis Agent: verify() ----
    hypothesis_results: List[Dict[str, Any]]
    hypothesis_scorecard: Dict[str, int]
    hypothesis_verification_summary: str

    # ---- Critic Agent: run() and apply_revision() ----
    critique: Dict[str, Any]
    loop_back_to: Optional[str]
    revision_plan: Optional[Dict[str, Any]]
    critic_summary: str
    revision_count: int
    dropped_by_critic: List[str]
    revision_history: List[Dict[str, Any]]

    # ---- Report Agent ----
    report_path: str
    report_sections: Dict[str, str]
    report_summary: str

    # ---- added by the graph itself ----
    step_log: List[Dict[str, Any]]  # one entry per node that ran: step, seconds, revision round
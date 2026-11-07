"""
Thin wrapper around MLflow so every agent logs runs the same way
instead of each agent calling mlflow directly with slightly different
conventions.
"""
import os
import mlflow

from config.settings import settings

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
mlflow.set_experiment("agentlab")


def log_run(run_name: str, params: dict, metrics: dict):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

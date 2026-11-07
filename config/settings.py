"""
Central place for configuration. Every agent and module reads settings
from here instead of calling os.getenv() scattered across the codebase.

A value in .env only reaches the code if it is declared here. If you add a
new line to .env, add a matching attribute below, otherwise the agent that
uses it fails with "'Settings' object has no attribute ...".
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # LLM provider: Ollama is used for every agent
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    USE_LOCAL_LLM = LLM_PROVIDER == "ollama"

    # Local model: narration-only agents (Validation, EDA, Feature Engineering,
    # Model Selection, Tuning, Explainability). Fast, free, no rate limit.
    LOCAL_MODEL = os.getenv("LOCAL_MODEL", "llama3.2:3b")

    # Cloud models: agents where reasoning quality matters. These need
    # `ollama signin`, an internet connection, and use your cloud quota.
    CRITIC_MODEL = os.getenv("CRITIC_MODEL", "gpt-oss:120b-cloud")
    HYPOTHESIS_MODEL = os.getenv("HYPOTHESIS_MODEL", "gpt-oss:20b-cloud")
    REPORT_MODEL = os.getenv("REPORT_MODEL", "nemotron-3-nano:30b-cloud")

    # Older files may still read OLLAMA_MODEL, so keep it as an alias.
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", LOCAL_MODEL)

    # Database
    DATABASE_URL = os.getenv(
        "DATABASE_URL", "postgresql://agentlab:agentlab@localhost:5432/agentlab"
    )

    # MLflow
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")

    # Paths
    UPLOAD_DIR = "data/uploads"
    PROCESSED_DIR = "data/processed"
    REPORTS_DIR = "reports"


settings = Settings()
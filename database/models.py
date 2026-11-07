"""
SQLAlchemy models for experiment storage. Keep this simple, one table
for experiment runs is enough for a project this size, don't build
out a schema for features you're not using yet (that DuckDB/FAISS
setup from the original spec was cut for exactly this reason).
"""
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id = Column(Integer, primary_key=True)
    dataset_name = Column(String, nullable=False)
    model_name = Column(String)
    metrics = Column(JSON)          # cross-validated scores
    best_params = Column(JSON)      # from the Tuning Agent
    critique = Column(JSON)         # the Critic Agent's verdict + flags
    report_path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

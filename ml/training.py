"""
Shared model training and evaluation helpers used by the Model
Selection and Tuning Agents. Kept separate so this logic is testable
without an LLM in the loop.
"""
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb


CLASSIFICATION_MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "xgboost": xgb.XGBClassifier(eval_metric="logloss"),
    "lightgbm": lgb.LGBMClassifier(),
}

REGRESSION_MODELS = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "xgboost": xgb.XGBRegressor(),
    "lightgbm": lgb.LGBMRegressor(),
}


def evaluate_models(X, y, task: str = "classification", cv: int = 5, scoring: str = "f1"):
    """
    Runs cross validation for every candidate model and returns a
    dict of {model_name: {"mean": float, "scores": list}}. Task must
    be "classification" or "regression", scoring should match
    (e.g. "f1" or "roc_auc" for classification, "neg_root_mean_squared_error"
    for regression).
    """
    models = CLASSIFICATION_MODELS if task == "classification" else REGRESSION_MODELS
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        results[name] = {"mean": float(scores.mean()), "scores": scores.tolist()}
    return results

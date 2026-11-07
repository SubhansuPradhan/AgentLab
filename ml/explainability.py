"""
Shared SHAP/LIME helpers used by the Explainability Agent.
"""
import shap


def compute_shap_values(model, X):
    explainer = shap.Explainer(model, X)
    return explainer(X)


def top_features_by_importance(shap_values, feature_names, n: int = 10):
    import numpy as np
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    ranked = sorted(zip(feature_names, mean_abs), key=lambda x: -x[1])
    return ranked[:n]

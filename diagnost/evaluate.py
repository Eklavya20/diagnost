import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)
from .report import DiagnostReport


def evaluate(model, X, y, task="classification", sensitive_features=None):
    """
    Evaluate a model and return a DiagnostReport.

    Parameters
    ----------
    model : any scikit-learn compatible model
    X : array-like or DataFrame
    y : array-like — true labels or values
    task : str — "classification", "regression", or "clustering"
    sensitive_features : list of str — column names to evaluate fairness across

    Returns
    -------
    DiagnostReport
    """

    X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
    y = np.array(y)
    results = {}

    if task == "classification":
        results = _evaluate_classification(model, X, y, sensitive_features)
    elif task == "regression":
        results = _evaluate_regression(model, X, y, sensitive_features)
    elif task == "clustering":
        results = _evaluate_clustering(model, X)
    else:
        raise ValueError(f"Unknown task '{task}'. Choose from: classification, regression, clustering.")

    return DiagnostReport(results=results, task=task)


def _evaluate_classification(model, X, y, sensitive_features):
    feature_cols = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else list(X.columns)
    y_pred = model.predict(X[feature_cols])
    y_proba = model.predict_proba(X[feature_cols]) if hasattr(model, "predict_proba") else None

    results = {
        "task": "classification",
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "y_pred": y_pred,
        "y_true": y,
        "y_proba": y_proba,
        "subgroup_results": {},
    }

    if sensitive_features:
        for feature in sensitive_features:
            if feature in X.columns:
                results["subgroup_results"][feature] = _subgroup_classification(
                    X, y, y_pred, feature
                )

    return results


def _evaluate_regression(model, X, y, sensitive_features):
    feature_cols = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else list(X.columns)
    y_pred = model.predict(X[feature_cols])
    residuals = y - y_pred

    results = {
        "task": "regression",
        "mae": mean_absolute_error(y, y_pred),
        "mse": mean_squared_error(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "r2": r2_score(y, y_pred),
        "y_pred": y_pred,
        "y_true": y,
        "residuals": residuals,
        "subgroup_results": {},
    }

    if sensitive_features:
        for feature in sensitive_features:
            if feature in X.columns:
                results["subgroup_results"][feature] = _subgroup_regression(
                    X, y, y_pred, feature
                )

    return results


def _evaluate_clustering(model, X):
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    labels = model.labels_ if hasattr(model, "labels_") else model.predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    results = {
        "task": "clustering",
        "n_clusters": n_clusters,
        "cluster_sizes": pd.Series(labels).value_counts().to_dict(),
    }

    try:
        results["silhouette_score"] = silhouette_score(X, labels)
        results["davies_bouldin_score"] = davies_bouldin_score(X, labels)
    except Exception:
        results["silhouette_score"] = None
        results["davies_bouldin_score"] = None

    return results


def _subgroup_classification(X, y_true, y_pred, feature):
    subgroups = {}
    for group in X[feature].unique():
        mask = X[feature] == group
        subgroups[str(group)] = {
            "accuracy": accuracy_score(y_true[mask], y_pred[mask]),
            "f1": f1_score(y_true[mask], y_pred[mask], average="weighted", zero_division=0),
            "support": int(mask.sum()),
        }
    return subgroups


def _subgroup_regression(X, y_true, y_pred, feature):
    subgroups = {}
    for group in X[feature].unique():
        mask = X[feature] == group
        subgroups[str(group)] = {
            "mae": mean_absolute_error(y_true[mask], y_pred[mask]),
            "r2": r2_score(y_true[mask], y_pred[mask]),
            "support": int(mask.sum()),
        }
    return subgroups
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from diagnost import evaluate


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def classification_data():
    X, y = load_iris(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)

@pytest.fixture
def regression_data():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)

@pytest.fixture
def clustering_data():
    X, _ = load_iris(return_X_y=True, as_frame=True)
    return X


# ── Classification ────────────────────────────────────────────────────────────

def test_classification_returns_report(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    report = evaluate(model, X_test, y_test, task="classification")
    assert report is not None

def test_classification_metrics_present(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    report = evaluate(model, X_test, y_test, task="classification")
    for key in ["accuracy", "precision", "recall", "f1", "confusion_matrix"]:
        assert key in report.results

def test_classification_accuracy_range(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    report = evaluate(model, X_test, y_test, task="classification")
    assert 0.0 <= report.results["accuracy"] <= 1.0

def test_classification_with_sensitive_features(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    X_test = X_test.copy()
    X_test["gender"] = np.random.choice(["M", "F"], size=len(X_test))
    model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    report = evaluate(model, X_test, y_test, task="classification",
                      sensitive_features=["gender"])
    assert "gender" in report.results["subgroup_results"]


# ── Regression ────────────────────────────────────────────────────────────────

def test_regression_returns_report(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    model = LinearRegression().fit(X_train, y_train)
    report = evaluate(model, X_test, y_test, task="regression")
    assert report is not None

def test_regression_metrics_present(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    model = LinearRegression().fit(X_train, y_train)
    report = evaluate(model, X_test, y_test, task="regression")
    for key in ["mae", "mse", "rmse", "r2"]:
        assert key in report.results

def test_regression_rmse_positive(regression_data):
    X_train, X_test, y_train, y_test = regression_data
    model = LinearRegression().fit(X_train, y_train)
    report = evaluate(model, X_test, y_test, task="regression")
    assert report.results["rmse"] >= 0


# ── Clustering ────────────────────────────────────────────────────────────────

def test_clustering_returns_report(clustering_data):
    model = KMeans(n_clusters=3, random_state=42, n_init=10).fit(clustering_data)
    report = evaluate(model, clustering_data, y=None, task="clustering")
    assert report is not None

def test_clustering_metrics_present(clustering_data):
    model = KMeans(n_clusters=3, random_state=42, n_init=10).fit(clustering_data)
    report = evaluate(model, clustering_data, y=None, task="clustering")
    for key in ["n_clusters", "silhouette_score", "davies_bouldin_score"]:
        assert key in report.results

def test_clustering_n_clusters(clustering_data):
    model = KMeans(n_clusters=3, random_state=42, n_init=10).fit(clustering_data)
    report = evaluate(model, clustering_data, y=None, task="clustering")
    assert report.results["n_clusters"] == 3


# ── Edge Cases ────────────────────────────────────────────────────────────────

def test_invalid_task_raises(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    with pytest.raises(ValueError):
        evaluate(model, X_test, y_test, task="invalid_task")

def test_report_to_dict(classification_data):
    X_train, X_test, y_train, y_test = classification_data
    model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    report = evaluate(model, X_test, y_test, task="classification")
    d = report.to_dict()
    assert isinstance(d, dict)

def test_report_save(tmp_path, classification_data):
    X_train, X_test, y_train, y_test = classification_data
    model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    report = evaluate(model, X_test, y_test, task="classification")
    path = tmp_path / "report.json"
    report.save(str(path))
    assert path.exists()
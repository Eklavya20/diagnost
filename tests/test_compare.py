import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from diagnost.compare import compare


@pytest.fixture
def classification_setup():
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    lr = LogisticRegression(max_iter=200).fit(X_train, y_train)
    return {"Random Forest": rf, "Logistic Regression": lr}, X_test, y_test


def test_compare_returns_report(classification_setup):
    models, X_test, y_test = classification_setup
    report = compare(models, X_test, y_test, task="classification")
    assert report is not None

def test_compare_all_models_present(classification_setup):
    models, X_test, y_test = classification_setup
    report = compare(models, X_test, y_test, task="classification")
    for name in models:
        assert name in report.results

def test_compare_to_dataframe(classification_setup):
    models, X_test, y_test = classification_setup
    report = compare(models, X_test, y_test, task="classification")
    df = report.to_dataframe()
    assert list(df.index) == list(models.keys())
    assert "accuracy" in df.columns

def test_compare_invalid_models_type(classification_setup):
    _, X_test, y_test = classification_setup
    with pytest.raises(ValueError):
        compare([1, 2, 3], X_test, y_test)
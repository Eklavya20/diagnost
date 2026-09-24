import pytest
import pandas as pd
import numpy as np
from diagnost import inspect_dataset


@pytest.fixture
def clean_df():
    from sklearn.datasets import load_iris
    X, _ = load_iris(return_X_y=True, as_frame=True)
    return X

@pytest.fixture
def dirty_df():
    df = pd.DataFrame({
        "age": [25, 30, np.nan, 40, 200],
        "income": [50000, 60000, 55000, np.nan, 70000],
        "category": ["A", "B", "A", "C", "B"],
    })
    return df


def test_inspect_returns_dict(clean_df):
    result = inspect_dataset(clean_df, plot=False)
    assert isinstance(result, dict)

def test_inspect_shape(clean_df):
    result = inspect_dataset(clean_df, plot=False)
    assert result["shape"] == (150, 4)

def test_inspect_detects_missing(dirty_df):
    result = inspect_dataset(dirty_df, plot=False)
    assert "age" in result["missing"]
    assert "income" in result["missing"]

def test_inspect_no_missing(clean_df):
    result = inspect_dataset(clean_df, plot=False)
    assert result["missing"] == {}

def test_inspect_detects_correlations(clean_df):
    result = inspect_dataset(clean_df, plot=False)
    assert len(result["correlations"]) > 0

def test_inspect_detects_outliers(dirty_df):
    result = inspect_dataset(dirty_df, plot=False)
    assert "age" in result["outliers"]

def test_invalid_input_raises():
    with pytest.raises(ValueError):
        inspect_dataset([1, 2, 3])


def test_empty_dataframe_raises():
    with pytest.raises(ValueError, match="at least one row and one column"):
        inspect_dataset(pd.DataFrame())


def test_object_categorical_balance_is_supported(dirty_df):
    result = inspect_dataset(dirty_df, plot=False)
    assert result["class_balance"]["category"] == {"A": 2, "B": 2, "C": 1}

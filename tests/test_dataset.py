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
    result = inspect_dataset(clean_df)
    assert isinstance(result, dict)

def test_inspect_shape(clean_df):
    result = inspect_dataset(clean_df)
    assert result["shape"] == (150, 4)

def test_inspect_detects_missing(dirty_df):
    result = inspect_dataset(dirty_df)
    assert "age" in result["missing"]
    assert "income" in result["missing"]

def test_inspect_no_missing(clean_df):
    result = inspect_dataset(clean_df)
    assert result["missing"] == {}

def test_inspect_detects_correlations(clean_df):
    result = inspect_dataset(clean_df)
    assert len(result["correlations"]) > 0

def test_inspect_detects_outliers(dirty_df):
    result = inspect_dataset(dirty_df)
    assert "age" in result["outliers"]

def test_invalid_input_raises():
    with pytest.raises(ValueError):
        inspect_dataset([1, 2, 3])
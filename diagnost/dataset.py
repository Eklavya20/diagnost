import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def inspect_dataset(df):
    """
    Run diagnostics on a dataset before modelling.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    dict with dataset diagnostics
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    results = {
        "shape": df.shape,
        "missing": _check_missing(df),
        "class_balance": _check_class_balance(df),
        "correlations": _check_correlations(df),
        "outliers": _check_outliers(df),
    }

    _print_dataset_summary(results, df)
    _plot_dataset(df, results)

    return results


def _check_missing(df):
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    return {
        col: {"count": int(missing[col]), "pct": float(pct[col])}
        for col in df.columns if missing[col] > 0
    }


def _check_class_balance(df):
    """Check value distribution for categorical columns."""
    balance = {}
    for col in df.select_dtypes(include=["category", "str"]).columns:
        counts = df[col].value_counts()
        balance[col] = counts.to_dict()
    return balance


def _check_correlations(df):
    """Find highly correlated numeric feature pairs."""
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] < 2:
        return {}
    corr = numeric.corr().abs()
    pairs = {}
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if val > 0.85:
                pairs[f"{cols[i]} & {cols[j]}"] = round(float(val), 4)
    return pairs


def _check_outliers(df):
    """Detect outliers using IQR method."""
    outliers = {}
    for col in df.select_dtypes(include=np.number).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        n_outliers = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        if n_outliers > 0:
            outliers[col] = {
                "n_outliers": n_outliers,
                "pct": round(n_outliers / len(df) * 100, 2)
            }
    return outliers


def _print_dataset_summary(results, df):
    print("\n========== dataset report ==========")
    print(f"  Shape : {results['shape'][0]} rows x {results['shape'][1]} columns")

    if results["missing"]:
        print(f"\n  Missing Values:")
        for col, m in results["missing"].items():
            print(f"    {col}: {m['count']} ({m['pct']}%)")
    else:
        print("\n  ✓ No missing values.")

    if results["correlations"]:
        print(f"\n  ⚠ Highly Correlated Features (>0.85):")
        for pair, val in results["correlations"].items():
            print(f"    {pair}: r={val}")
    else:
        print("  ✓ No high correlations detected.")

    if results["outliers"]:
        print(f"\n  ⚠ Outliers Detected (IQR method):")
        for col, o in results["outliers"].items():
            print(f"    {col}: {o['n_outliers']} outliers ({o['pct']}%)")
    else:
        print("  ✓ No outliers detected.")

    print("=====================================\n")


def _plot_dataset(df, results):
    """Plot distributions for numeric columns."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        return

    n = len(numeric_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric_cols):
        ax.hist(df[col].dropna(), bins=20, color="steelblue", edgecolor="white")
        ax.set_title(col)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    plt.suptitle("Dataset — Feature Distributions", fontsize=13)
    plt.tight_layout()
    plt.show()
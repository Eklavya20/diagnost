import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


def check_drift(X_ref, X_new, threshold=0.05, plot=True):
    """
    Detect feature drift between a reference dataset and new data.

    Parameters
    ----------
    X_ref : DataFrame — reference/training data
    X_new : DataFrame — new/production data
    threshold : float — p-value threshold for drift detection (default 0.05)
    plot : bool — whether to visualise drift

    Returns
    -------
    dict with drift results per feature
    """

    X_ref = pd.DataFrame(X_ref)
    X_new = pd.DataFrame(X_new)

    if list(X_ref.columns) != list(X_new.columns):
        raise ValueError("X_ref and X_new must have the same columns.")

    results = {}

    for col in X_ref.columns:
        ref_vals = X_ref[col].dropna().values
        new_vals = X_new[col].dropna().values

        if _is_numeric(ref_vals):
            stat, p_value = stats.ks_2samp(ref_vals, new_vals)
            test_used = "Kolmogorov-Smirnov"
        else:
            stat, p_value = _chi_square_drift(ref_vals, new_vals)
            test_used = "Chi-Square"

        drifted = p_value < threshold

        results[col] = {
            "test": test_used,
            "statistic": round(float(stat), 4),
            "p_value": round(float(p_value), 4),
            "drifted": drifted,
            "verdict": f"{'⚠ DRIFT DETECTED' if drifted else '✓ No drift'} (p={round(float(p_value), 4)})"
        }

    _print_drift_summary(results, threshold)

    if plot:
        _plot_drift(X_ref, X_new, results)

    return results


def _is_numeric(arr):
    return np.issubdtype(arr.dtype, np.number)


def _chi_square_drift(ref_vals, new_vals):
    """Chi-square test for categorical feature drift."""
    all_cats = np.union1d(np.unique(ref_vals), np.unique(new_vals))
    ref_counts = np.array([np.sum(ref_vals == c) for c in all_cats], dtype=float)
    new_counts = np.array([np.sum(new_vals == c) for c in all_cats], dtype=float)

    # Avoid division by zero
    ref_counts += 1e-10
    new_counts += 1e-10

    stat, p_value = stats.chisquare(new_counts, f_exp=ref_counts * (new_counts.sum() / ref_counts.sum()))
    return stat, p_value


def _print_drift_summary(results, threshold):
    """Print plain-English drift summary."""
    drifted = [col for col, r in results.items() if r["drifted"]]

    print("\n========== drift report ==========")
    print(f"  Threshold : p < {threshold}")
    print(f"  Features checked : {len(results)}")
    print(f"  Features drifted : {len(drifted)}\n")

    for col, r in results.items():
        print(f"  {col:30s} {r['verdict']}")

    if drifted:
        print(f"\n  ⚠ Warning: Drift detected in {len(drifted)} feature(s): {', '.join(drifted)}")
        print("  Consider retraining your model or investigating data pipeline changes.")
    else:
        print("\n  ✓ No significant drift detected. Model inputs appear stable.")

    print("===================================\n")


def _plot_drift(X_ref, X_new, results):
    """Plot distribution comparison for drifted features."""
    drifted_cols = [col for col, r in results.items() if r["drifted"]]

    if not drifted_cols:
        print("No drifted features to plot.")
        return

    n = len(drifted_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, drifted_cols):
        ref_vals = X_ref[col].dropna().values
        new_vals = X_new[col].dropna().values

        if _is_numeric(ref_vals):
            ax.hist(ref_vals, bins=20, alpha=0.5, label="Reference", color="steelblue")
            ax.hist(new_vals, bins=20, alpha=0.5, label="New", color="tomato")
        else:
            cats = np.union1d(np.unique(ref_vals), np.unique(new_vals))
            x = np.arange(len(cats))
            ref_counts = [np.sum(ref_vals == c) for c in cats]
            new_counts = [np.sum(new_vals == c) for c in cats]
            ax.bar(x - 0.2, ref_counts, 0.4, label="Reference", color="steelblue")
            ax.bar(x + 0.2, new_counts, 0.4, label="New", color="tomato")
            ax.set_xticks(x)
            ax.set_xticklabels(cats, rotation=45)

        ax.set_title(f"{col}\np={results[col]['p_value']}")
        ax.legend()

    plt.suptitle("Drift — Distribution Comparison", fontsize=13)
    plt.tight_layout()
    plt.show()
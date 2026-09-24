import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def check_calibration(model, X, y, n_bins=10, plot=True):
    """
    Check how well a classification model's predicted probabilities
    match actual outcomes.

    Parameters
    ----------
    model : sklearn-compatible classifier with predict_proba
    X : array-like or DataFrame
    y : array-like — true binary or multiclass labels
    n_bins : int — number of calibration bins
    plot : bool — whether to display the reliability diagram

    Returns
    -------
    dict with calibration metrics
    """

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support predict_proba for calibration analysis.")
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")

    y = np.asarray(y)
    y_proba = model.predict_proba(X)
    classes = model.classes_
    if len(y) == 0:
        raise ValueError("y must contain at least one sample")
    if len(y) != len(y_proba):
        raise ValueError("X and y must contain the same number of samples")
    results = {}

    for i, cls in enumerate(classes):
        y_binary = (y == cls).astype(int)
        prob_true, prob_pred = calibration_curve(y_binary, y_proba[:, i], n_bins=n_bins)

        ece = _expected_calibration_error(y_binary, y_proba[:, i], n_bins)

        results[str(cls)] = {
            "expected_calibration_error": round(ece, 4),
            "brier_score": round(
                float(brier_score_loss(y_binary, y_proba[:, i])), 4
            ),
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }

        verdict = _calibration_verdict(ece)
        results[str(cls)]["verdict"] = verdict


    _print_calibration_summary(results)
    if plot:
        _plot_calibration(results, classes)

    return results


def _expected_calibration_error(y_true, y_proba, n_bins):
    """Calculate Expected Calibration Error (ECE)."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    if len(y_true) == 0:
        raise ValueError("y_true must contain at least one sample")
    if len(y_true) != len(y_proba):
        raise ValueError("y_true and y_proba must have the same length")
    if np.any((y_proba < 0) | (y_proba > 1)):
        raise ValueError("y_proba values must be between 0 and 1")

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_proba >= bins[i]) & (y_proba <= bins[i + 1])
        else:
            mask = (y_proba >= bins[i]) & (y_proba < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_proba[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return ece


def _calibration_verdict(ece):
    """Return a plain-English verdict based on ECE."""
    if ece < 0.05:
        return "Well calibrated — predicted probabilities are reliable."
    elif ece < 0.10:
        return "Moderately calibrated — minor over or underconfidence detected."
    else:
        return "Poorly calibrated — predicted probabilities are not reliable."


def _plot_calibration(results, classes):
    """Plot reliability diagrams for each class."""
    n = len(classes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        data = results[str(cls)]
        ax.plot(data["prob_pred"], data["prob_true"], marker="o", label="Model")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
        ax.set_title(f"Class: {cls}\nECE: {data['expected_calibration_error']}")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.legend()

    plt.suptitle("Calibration — Reliability Diagrams", fontsize=13)
    plt.tight_layout()
    plt.show()


def _print_calibration_summary(results):
    """Print plain-English calibration summary."""
    print("\n========== calibration report ==========")
    for cls, data in results.items():
        print(
            f"  Class {cls}: ECE={data['expected_calibration_error']}, "
            f"Brier={data['brier_score']} — {data['verdict']}"
        )
    print("=========================================\n")

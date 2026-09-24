import numpy as np
import pytest

from diagnost.calibration import _expected_calibration_error, check_calibration


class PerfectProbabilityModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        probabilities = np.asarray(X, dtype=float).reshape(-1)
        return np.column_stack([1 - probabilities, probabilities])


def test_ece_includes_probability_of_one():
    y_true = np.array([0, 1])
    y_proba = np.array([0.0, 1.0])
    assert _expected_calibration_error(y_true, y_proba, n_bins=10) == 0.0


def test_calibration_reports_brier_score():
    model = PerfectProbabilityModel()
    results = check_calibration(model, [[0.0], [1.0]], np.array([0, 1]), plot=False)
    assert results["0"]["brier_score"] == 0.0
    assert results["1"]["brier_score"] == 0.0


def test_calibration_rejects_empty_targets():
    model = PerfectProbabilityModel()
    with pytest.raises(ValueError, match="at least one sample"):
        check_calibration(model, [], np.array([]), plot=False)


def test_calibration_rejects_invalid_bin_count():
    model = PerfectProbabilityModel()
    with pytest.raises(ValueError, match="positive integer"):
        check_calibration(model, [[0.5]], np.array([1]), n_bins=0, plot=False)

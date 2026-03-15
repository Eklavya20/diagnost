import pandas as pd
from .evaluate import _evaluate_classification, _evaluate_regression, _evaluate_clustering


def compare(models, X, y=None, task="classification"):
    """
    Compare multiple models side by side.

    Parameters
    ----------
    models : dict — {"model name": model_object}
    X : array-like or DataFrame
    y : array-like — true labels (not needed for clustering)
    task : str — "classification", "regression", or "clustering"

    Returns
    -------
    CompareReport
    """

    if not isinstance(models, dict):
        raise ValueError("models must be a dict: {'name': model}")

    X = pd.DataFrame(X)
    results = {}

    for name, model in models.items():
        if task == "classification":
            results[name] = _evaluate_classification(model, X, y, sensitive_features=None)
        elif task == "regression":
            results[name] = _evaluate_regression(model, X, y, sensitive_features=None)
        elif task == "clustering":
            results[name] = _evaluate_clustering(model, X)
        else:
            raise ValueError(f"Unknown task '{task}'.")

    report = CompareReport(results=results, task=task)
    report.summary()
    return report


class CompareReport:
    """Holds and displays side-by-side model comparison results."""

    def __init__(self, results, task):
        self.results = results
        self.task = task

    def summary(self):
        print("\n========== model comparison report ==========")
        print(f"Task: {self.task.upper()}\n")

        if self.task == "classification":
            header = f"  {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
            print(header)
            print("  " + "-" * 65)
            for name, r in self.results.items():
                print(f"  {name:<25} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
                      f"{r['recall']:>10.4f} {r['f1']:>10.4f}")

        elif self.task == "regression":
            header = f"  {'Model':<25} {'MAE':>10} {'RMSE':>10} {'R²':>10}"
            print(header)
            print("  " + "-" * 55)
            for name, r in self.results.items():
                print(f"  {name:<25} {r['mae']:>10.4f} {r['rmse']:>10.4f} {r['r2']:>10.4f}")

        elif self.task == "clustering":
            header = f"  {'Model':<25} {'Clusters':>10} {'Silhouette':>12} {'Davies-Bouldin':>16}"
            print(header)
            print("  " + "-" * 65)
            for name, r in self.results.items():
                sil = f"{r['silhouette_score']:.4f}" if r['silhouette_score'] else "N/A"
                db = f"{r['davies_bouldin_score']:.4f}" if r['davies_bouldin_score'] else "N/A"
                print(f"  {name:<25} {r['n_clusters']:>10} {sil:>12} {db:>16}")

        print("\n" + "  " + "=" * 45)
        self._verdict()
        print("=============================================\n")

    def _verdict(self):
        """Print a plain-English winner."""
        if self.task == "classification":
            winner = max(self.results, key=lambda n: self.results[n]["f1"])
            print(f"\n  ✓ Best model by F1: {winner}")
        elif self.task == "regression":
            winner = min(self.results, key=lambda n: self.results[n]["mae"])
            print(f"\n  ✓ Best model by MAE: {winner}")
        elif self.task == "clustering":
            scores = {n: r["silhouette_score"] for n, r in self.results.items()
                      if r["silhouette_score"] is not None}
            if scores:
                winner = max(scores, key=scores.get)
                print(f"\n  ✓ Best model by Silhouette: {winner}")

    def to_dataframe(self):
        """Return comparison as a pandas DataFrame."""
        if self.task == "classification":
            rows = {name: {k: r[k] for k in ["accuracy", "precision", "recall", "f1"]}
                    for name, r in self.results.items()}
        elif self.task == "regression":
            rows = {name: {k: r[k] for k in ["mae", "mse", "rmse", "r2"]}
                    for name, r in self.results.items()}
        else:
            rows = {name: {k: r[k] for k in ["n_clusters", "silhouette_score", "davies_bouldin_score"]}
                    for name, r in self.results.items()}
        return pd.DataFrame(rows).T
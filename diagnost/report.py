import json


class DiagnostReport:
    """
    Holds the results of a diagnost evaluation and provides
    methods to view, summarise, and export them.
    """

    def __init__(self, results, task):
        self.results = results
        self.task = task

    def summary(self):
        """Print a plain-English summary to the console."""
        print("\n========== diagnost report ==========")
        print(f"Task: {self.task.upper()}\n")

        if self.task == "classification":
            print(f"  Accuracy  : {self.results['accuracy']:.4f}")
            print(f"  Precision : {self.results['precision']:.4f}")
            print(f"  Recall    : {self.results['recall']:.4f}")
            print(f"  F1 Score  : {self.results['f1']:.4f}")

            if self.results["subgroup_results"]:
                print("\n  Subgroup Performance:")
                for feature, groups in self.results["subgroup_results"].items():
                    print(f"\n    Feature: {feature}")
                    for group, metrics in groups.items():
                        print(f"      {group}: accuracy={metrics['accuracy']:.4f}, "
                              f"f1={metrics['f1']:.4f}, n={metrics['support']}")

        elif self.task == "regression":
            print(f"  MAE  : {self.results['mae']:.4f}")
            print(f"  RMSE : {self.results['rmse']:.4f}")
            print(f"  R²   : {self.results['r2']:.4f}")

            if self.results["subgroup_results"]:
                print("\n  Subgroup Performance:")
                for feature, groups in self.results["subgroup_results"].items():
                    print(f"\n    Feature: {feature}")
                    for group, metrics in groups.items():
                        print(f"      {group}: mae={metrics['mae']:.4f}, "
                              f"r2={metrics['r2']:.4f}, n={metrics['support']}")

        elif self.task == "clustering":
            print(f"  Clusters  : {self.results['n_clusters']}")
            if self.results["silhouette_score"] is not None:
                print(f"  Silhouette: {self.results['silhouette_score']:.4f}")
                print(f"  Davies-Bouldin: {self.results['davies_bouldin_score']:.4f}")
            print(f"  Cluster sizes: {self.results['cluster_sizes']}")

        print("\n=====================================\n")

    def to_dict(self):
        """Return results as a plain dictionary."""
        safe = {k: v for k, v in self.results.items()
                if k not in ("y_pred", "y_true", "y_proba", "residuals")}
        return safe

    def save(self, path="diagnost_report.json"):
        """Save results to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"Report saved to {path}")
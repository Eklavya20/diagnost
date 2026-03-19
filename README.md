# diagnost

**A model diagnostics library for data scientists.**  
Performance, calibration, drift detection, and dataset health checks, all in a few lines of Python.

[![PyPI version](https://badge.fury.io/py/diagnost.svg)](https://pypi.org/project/diagnost/)
[![Python](https://img.shields.io/pypi/pyversions/diagnost)](https://pypi.org/project/diagnost/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-24%20passed-brightgreen)](https://github.com/Eklavya20/diagnost)

## Why diagnost?

Most ML libraries help you build models. diagnost helps you **trust them**.

After training, the real questions start:
- Is my model actually reliable, or just accurate on average?
- Does it perform equally across different groups?
- Are its confidence scores meaningful?
- Has my data drifted since I trained it?

diagnost answers all of these, cleanly, quickly, and in plain English.

## Installation
```bash
pip install diagnost
```

## Quickstart
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import diagnost

X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier().fit(X_train, y_train)

report = diagnost.evaluate(model, X_test, y_test, task="classification")
report.summary()
```

## Features

### 1. Model Evaluation
Evaluate classification, regression, and clustering models with one call.
```python
# Classification
report = diagnost.evaluate(model, X_test, y_test, task="classification")

# Regression
report = diagnost.evaluate(model, X_test, y_test, task="regression")

# Clustering
report = diagnost.evaluate(model, X_test, task="clustering")
```

**Subgroup / fairness analysis** — check performance across sensitive groups:
```python
report = diagnost.evaluate(
    model, X_test, y_test,
    task="classification",
    sensitive_features=["gender", "age_group"]
)
report.summary()
```

### 2. Model Comparison
Compare multiple models side by side with a winner declared automatically.
```python
from diagnost.compare import compare

report = compare(
    models={"Random Forest": rf, "Logistic Regression": lr},
    X=X_test,
    y=y_test,
    task="classification"
)

df = report.to_dataframe()  # returns a pandas DataFrame
```

### 3. Calibration Analysis
Check whether your model's predicted probabilities are actually reliable.
```python
from diagnost.calibration import check_calibration

check_calibration(model, X_test, y_test)
```

Output includes:
- Expected Calibration Error (ECE) per class
- Plain-English verdict ("Well calibrated", "Poorly calibrated")
- Reliability diagram

### 4. Drift Detection
Detect whether your input data has shifted since training.
```python
from diagnost.drift import check_drift

check_drift(X_train, X_new)
```

- Kolmogorov-Smirnov test for numeric features
- Chi-Square test for categorical features
- Per-feature drift verdict with p-values
- Distribution plots for drifted features

### 5. Dataset Diagnostics
Inspect your dataset before modelling.
```python
results = diagnost.inspect_dataset(df)
```

Checks for:
- Missing values
- Highly correlated features (r > 0.85)
- Outliers (IQR method)
- Feature distributions (visual)

## Saving Reports
```python
report = diagnost.evaluate(model, X_test, y_test, task="classification")
report.save("report.json")  # exports as JSON
```

## Supported Model Types

| Task           | Supported Frameworks                      |
|----------------|-------------------------------------------|
| Classification | scikit-learn, XGBoost, LightGBM, CatBoost |
| Regression     | scikit-learn, XGBoost, LightGBM, CatBoost |
| Clustering     | scikit-learn                              |

Any model with a `.predict()` method will work.

## Requirements

- Python >= 3.9
- numpy, pandas, scipy, matplotlib, scikit-learn

## Contributing

Contributions are welcome. To get started:
```bash
git clone https://github.com/Eklavya20/diagnost.git
cd diagnost
python -m venv venv
venv\Scripts\activate      # Windows
pip install -e ".[dev]"
pytest tests/ -v
```

Please open an issue before submitting a large pull request.

## License

MIT License — free to use, modify, and distribute.  
See [LICENSE](LICENSE) for details.

## Author

**Eklavya Jumnani**  
MSc Data Science, FAU Erlangen-Nürnberg  
[GitHub](https://github.com/Eklavya20) · [LinkedIn](https://www.linkedin.com/in/eklavya-jumnani/)
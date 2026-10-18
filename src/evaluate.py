"""Evaluate the trained model on the test set and apply the quality gate."""
import argparse
import json
import os
import sys

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import data
from src.config import load_params


def compute_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def passes_quality_gate(metrics, min_r2):
    return metrics["r2"] >= min_r2


def main(timestamp, params_path="params.yaml", models_dir="models", metrics_dir="metrics"):
    params = load_params(params_path)

    model = joblib.load(os.path.join(models_dir, f"model_{timestamp}.joblib"))

    X, y = data.load_data()
    _, X_test, _, y_test = data.split_data(X, y, **params["data"])

    metrics = compute_metrics(y_test, model.predict(X_test))
    metrics["timestamp"] = timestamp

    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, f"metrics_{timestamp}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(json.dumps(metrics, indent=4))
    print(f"Metrics saved to {metrics_path}")

    min_r2 = params["evaluation"]["min_r2"]
    if not passes_quality_gate(metrics, min_r2):
        print(f"QUALITY GATE FAILED: R² {metrics['r2']:.4f} is below {min_r2}")
        sys.exit(1)  # a non-zero exit makes the GitHub Actions step fail
    print(f"Quality gate passed: R² {metrics['r2']:.4f} >= {min_r2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    main(args.timestamp, args.params)

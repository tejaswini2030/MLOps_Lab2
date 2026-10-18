"""Train the house price model and save it with a timestamp-based version."""
import argparse
import os

import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

from src import data
from src.config import load_params


def build_model(model_params):
    return HistGradientBoostingRegressor(**model_params)


def train_model(X_train, y_train, model_params):
    model = build_model(model_params)
    model.fit(X_train, y_train)
    return model


def main(timestamp, params_path="params.yaml", models_dir="models"):
    params = load_params(params_path)

    X, y = data.load_data()
    X_train, _, y_train, _ = data.split_data(X, y, **params["data"])
    print(f"Training on {len(X_train)} houses with {X_train.shape[1]} features")

    model = train_model(X_train, y_train, params["model"])

    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"model_{timestamp}.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", required=True, help="Version tag from GitHub Actions")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    main(args.timestamp, args.params)

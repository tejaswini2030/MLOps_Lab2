"""Unit tests that run in CI before any training happens.
They use small fake data, so they're fast and don't need to download anything."""
import json

import pandas as pd
import pytest
from sklearn.datasets import make_regression

from src.compare import load_json, should_promote
from src.config import load_params
from src.data import split_data
from src.evaluate import compute_metrics, passes_quality_gate
from src.train import train_model

SMALL_MODEL = {"max_iter": 20, "random_state": 0}


@pytest.fixture
def fake_data():
    X, y = make_regression(n_samples=200, n_features=8, noise=5, random_state=0)
    return pd.DataFrame(X), pd.Series(y)


def test_params_file_is_valid():
    params = load_params("params.yaml")
    assert {"data", "model", "evaluation"} <= params.keys()
    assert 0 < params["data"]["test_size"] < 1
    assert 0 < params["evaluation"]["min_r2"] <= 1


def test_split_sizes(fake_data):
    X, y = fake_data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25, random_state=0)
    assert len(X_train) == 150 and len(X_test) == 50
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test)


def test_split_is_reproducible(fake_data):
    X, y = fake_data
    first = split_data(X, y, random_state=1)[1]
    second = split_data(X, y, random_state=1)[1]
    assert first.index.equals(second.index)


def test_model_trains_and_predicts(fake_data):
    X, y = fake_data
    model = train_model(X, y, SMALL_MODEL)
    predictions = model.predict(X)
    assert predictions.shape == (len(X),)


def test_model_learns_something(fake_data):
    X, y = fake_data
    X_train, X_test, y_train, y_test = split_data(X, y, random_state=0)
    model = train_model(X_train, y_train, {"max_iter": 100, "random_state": 0})
    assert compute_metrics(y_test, model.predict(X_test))["r2"] > 0.5


def test_metrics_for_perfect_predictions():
    metrics = compute_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert metrics["rmse"] == 0 and metrics["mae"] == 0 and metrics["r2"] == 1


def test_metrics_known_values():
    metrics = compute_metrics([0.0, 0.0], [1.0, -1.0])
    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx(1.0)


def test_quality_gate():
    assert passes_quality_gate({"r2": 0.80}, min_r2=0.75)
    assert not passes_quality_gate({"r2": 0.70}, min_r2=0.75)


def test_promotion_logic():
    assert should_promote({"r2": 0.8}, None)[0]
    assert should_promote({"r2": 0.85}, {"r2": 0.80})[0]
    assert not should_promote({"r2": 0.80}, {"r2": 0.80})[0]
    assert not should_promote({"r2": 0.70}, {"r2": 0.80})[0]


def test_load_json_missing_file_returns_none(tmp_path):
    assert load_json(str(tmp_path / "missing.json")) is None
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"r2": 0.9}))
    assert load_json(str(path)) == {"r2": 0.9}

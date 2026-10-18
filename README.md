# Lab 2: CI/CD for a House Price Regression Model

This lab builds a CI/CD pipeline with GitHub Actions for a regression model that predicts
California house prices. Every change to the code or settings is automatically tested,
the model is retrained and checked against a quality bar, and a better model is released
as a new version.

## The model

- **Dataset:** California Housing (scikit-learn), about 20,000 districts with 8 features
  such as median income, house age, average rooms and location.
- **Target:** median house value, in units of $100,000.
- **Model:** `HistGradientBoostingRegressor`, with settings in `params.yaml`.
- **Metrics:** RMSE, MAE and R², measured on a 20% test set the model never trains on.

## Project structure

```
Lab_2/
├── params.yaml          # data split, model hyperparameters, quality threshold
├── requirements.txt
├── pytest.ini           # pytest settings
├── src/
│   ├── config.py        # loads params.yaml
│   ├── data.py          # loads and splits the dataset
│   ├── train.py         # trains and saves models/model_<timestamp>.joblib
│   ├── evaluate.py      # computes metrics and applies the quality gate
│   └── compare.py       # decides whether the new model beats the last release
└── tests/
    └── test_pipeline.py # unit tests that run in CI
```

The workflow lives at the repository root in `.github/workflows/lab2_house_prices.yml`.

## The pipeline

It runs on every push or pull request that changes `Lab_2/`, and can also be started from
the **Actions** tab with **Run workflow**.

**CI (job `test`):** installs dependencies, lints the code with flake8, and runs the pytest
unit tests. If anything fails, the pipeline stops here and no model is trained.

**CD (job `train-and-release`, only after the tests pass and not on pull requests):**

1. Creates a timestamp that becomes the model's version.
2. Trains the model on the training split.
3. Evaluates it on the test split. **Quality gate:** if R² is below `min_r2` in
   `params.yaml`, the job fails and nothing is released.
4. Downloads the metrics of the last released model and compares the two.
5. Uploads the model and metrics as a workflow artifact (kept for every run).
6. If the new model beats the last release, publishes it as a **GitHub Release**
   (`lab2-house-price-<timestamp>`) with the model file and its metrics attached.

Models are versioned through Releases rather than committed into the repository, which
keeps large binary files out of the Git history.

## Running it locally

```bash
cd Lab_2
pip install -r requirements.txt
pytest -v
python -m src.train --timestamp test1
python -m src.evaluate --timestamp test1
```

## Trying the pipeline

- **Improve the model:** change a value in `params.yaml` (for example `max_iter: 500`) and
  push. If R² goes up, a new release is created.
- **See the quality gate work:** set `min_r2: 0.99` and push. The evaluate step fails.
- **See CI block bad code:** break a test on purpose and push. Training never starts.

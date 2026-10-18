# Lab 2: CI/CD for a House Price Regression Model

This lab builds a CI/CD pipeline with GitHub Actions for a regression model that predicts
California house prices. Every push is automatically tested, the model is retrained and
checked against a quality bar, and every model that passes is versioned in the repository.
Models that beat the previous best are also published as a GitHub Release.

## The model

- **Dataset:** California Housing (scikit-learn), about 20,000 districts with 8 features
  such as median income, house age, average rooms and location.
- **Target:** median house value, in units of $100,000.
- **Model:** `HistGradientBoostingRegressor`, with settings in `params.yaml`.
- **Metrics:** RMSE, MAE and R², measured on a 20% test set the model never trains on.

## Project structure

```
MLOps_Lab2/
├── .github/
│   └── workflows/
│       └── lab2_house_prices.yml   # the CI/CD pipeline
├── .gitignore
├── params.yaml          # data split, model hyperparameters, quality threshold
├── requirements.txt
├── pytest.ini           # pytest settings
├── src/
│   ├── config.py        # loads params.yaml
│   ├── data.py          # loads and splits the dataset
│   ├── train.py         # trains and saves models/model_<timestamp>.joblib
│   ├── evaluate.py      # computes metrics and applies the quality gate
│   └── compare.py       # decides whether the new model beats the last release
├── tests/
│   └── test_pipeline.py # unit tests that run in CI
├── models/              # models committed by the pipeline (one per passing run)
└── metrics/             # metrics committed by the pipeline (one per passing run)
```

## The pipeline

It runs on every push to `main` and on every pull request, and can also be started from
the **Actions** tab with **Run workflow**.

**CI (job `test`):** installs dependencies and runs the pytest unit tests. If any test
fails, the pipeline stops here and no model is trained.

**CD (job `train-and-release`, only after the tests pass and not on pull requests):**

1. Creates a timestamp that becomes the model's version.
2. Trains the model on the training split.
3. Evaluates it on the test split. **Quality gate:** if R² is below `min_r2` in
   `params.yaml`, the job fails and nothing is versioned.
4. Downloads the metrics of the last released model and compares the two.
5. Commits the model to `models/` and its metrics to `metrics/`, so every model that
   passed the quality gate is versioned in the repository's history.
6. Uploads the model and metrics as a workflow artifact.
7. If the new model beats the last release, publishes it as a **GitHub Release**
   (`lab2-house-price-<timestamp>`) with the model file and its metrics attached.

The commit in step 5 is made by the GitHub Actions bot. It does not trigger the workflow
again, but it does mean your local copy falls behind, so run `git pull` before making
new changes.

## Running it locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
pytest -v
python -m src.train --timestamp local1
python -m src.evaluate --timestamp local1
```

Local `models/` and `metrics/` files are ignored by Git (see `.gitignore`), so test runs
on your machine are never committed. Only models trained by the pipeline are added.

## Trying the pipeline

- **Improve the model:** change a value in `params.yaml` (for example `max_iter: 500`) and
  push. A new model and metrics are committed, and if R² goes up, a new release is created.
- **See the quality gate work:** set `min_r2: 0.99` and push. The evaluate step fails and
  nothing is committed or released.
- **See CI block bad code:** break a test on purpose and push. Training never starts.

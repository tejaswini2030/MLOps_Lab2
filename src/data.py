"""Load and split the California Housing dataset."""
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def load_data():
    """Return features X (DataFrame) and target y (median house value, in $100,000s)."""
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Split into train and test sets. A fixed random_state keeps the split identical
    between train.py and evaluate.py, so the model is never evaluated on its training data."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

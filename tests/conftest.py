import pandas as pd
import pytest


@pytest.fixture
def df_folds():
    """3-fold synthetic dataset, 2 rows per fold."""
    return pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "y": [0, 1, 0, 1, 0, 1],
            "fold": [0, 0, 1, 1, 2, 2],
        }
    )


@pytest.fixture
def conf():
    return {"target": "y"}


@pytest.fixture
def train_fn():
    def _train_fn(df_train, conf):
        return {"mean_y": df_train[conf["target"]].mean()}

    return _train_fn


@pytest.fixture
def predict_fn():
    def _predict_fn(fit, df_test, conf):
        value = conf["dummy_param"] if "dummy_param" in conf else fit["mean_y"]
        return [value] * len(df_test)

    return _predict_fn


@pytest.fixture
def eval_fn():
    def _eval_fn(df_eval, conf):
        return pd.DataFrame({"mean_pred": [df_eval["pred"].mean()]})

    return _eval_fn

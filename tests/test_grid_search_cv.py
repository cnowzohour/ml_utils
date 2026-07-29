import copy

from ml_utils.cv import grid_search_cv


def test_grid_search_cv_evaluates_all_combinations(df_folds, predict_fn, eval_fn):
    conf = {"target": "y", "params": {}}
    param_grid = {"a": [1, 2], "b": [10, 20, 30]}

    def train_fn(df_train, conf):
        return {"mean_y": df_train[conf["target"]].mean()}

    result = grid_search_cv(df_folds, train_fn, predict_fn, eval_fn, conf, param_grid)

    assert len(result) == 6  # 2 * 3 combinations
    assert set(result["a"]) == {1, 2}
    assert set(result["b"]) == {10, 20, 30}
    assert len(result[["a", "b"]].drop_duplicates()) == 6


def test_grid_search_cv_passes_each_combination_to_train_fn(df_folds, predict_fn, eval_fn):
    conf = {"target": "y", "params": {}}
    param_grid = {"a": [1, 2], "b": [10, 20]}

    seen_params = []

    def train_fn(df_train, conf):
        seen_params.append(copy.deepcopy(conf["params"]))
        return {"mean_y": df_train[conf["target"]].mean()}

    grid_search_cv(df_folds, train_fn, predict_fn, eval_fn, conf, param_grid)

    expected_combinations = {(1, 10), (1, 20), (2, 10), (2, 20)}
    observed_combinations = {(p["a"], p["b"]) for p in seen_params}
    assert observed_combinations == expected_combinations


def test_grid_search_cv_does_not_mutate_base_conf(df_folds, train_fn, predict_fn, eval_fn):
    conf = {"target": "y", "params": {}}
    param_grid = {"a": [1, 2]}

    grid_search_cv(df_folds, train_fn, predict_fn, eval_fn, conf, param_grid)

    assert conf["params"] == {}


def test_grid_search_cv_with_pred_param_sweep(df_folds, train_fn, predict_fn, eval_fn):
    conf = {"target": "y", "params": {}}
    param_grid = {"a": [1, 2]}

    result = grid_search_cv(
        df_folds,
        train_fn,
        predict_fn,
        eval_fn,
        conf,
        param_grid,
        pred_param_name="dummy_param",
        pred_param_values=[10, 20],
    )

    assert len(result) == 4  # 2 param combinations x 2 pred_param values
    assert set(result["a"]) == {1, 2}
    assert set(result["dummy_param"]) == {10, 20}

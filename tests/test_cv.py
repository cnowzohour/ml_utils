from ml_utils.cv import cross_validation


def test_cross_validation_basic_output_shape(df_folds, conf, train_fn, predict_fn, eval_fn):
    result = cross_validation(df_folds, train_fn, predict_fn, eval_fn, conf)

    assert len(result) == 1
    expected_cols = {"mean_pred", "mean_pred_fold0", "mean_pred_fold1", "mean_pred_fold2"}
    assert expected_cols.issubset(result.columns)


def test_cross_validation_excludes_test_fold_from_training(
    df_folds, conf, predict_fn, eval_fn
):
    seen_train_folds = []

    def train_fn(df_train, conf):
        seen_train_folds.append(set(df_train["fold"].unique()))
        return {"mean_y": df_train[conf["target"]].mean()}

    cross_validation(df_folds, train_fn, predict_fn, eval_fn, conf)

    all_folds = set(df_folds["fold"].unique())
    for held_out_fold, train_folds in zip(sorted(all_folds), seen_train_folds):
        assert held_out_fold not in train_folds
        assert train_folds == all_folds - {held_out_fold}


def test_cross_validation_pred_param_sweep(df_folds, conf, train_fn, predict_fn, eval_fn):
    result = cross_validation(
        df_folds,
        train_fn,
        predict_fn,
        eval_fn,
        conf,
        pred_param_name="dummy_param",
        pred_param_values=[10, 20],
    )

    assert len(result) == 2
    assert list(result["dummy_param"]) == [10, 20]
    assert list(result["mean_pred"]) == [10, 20]


def test_cross_validation_no_sweep_by_default(df_folds, conf, train_fn, predict_fn, eval_fn):
    result = cross_validation(df_folds, train_fn, predict_fn, eval_fn, conf)

    assert len(result) == 1
    assert "dummy_param" not in result.columns


def test_cross_validation_cv_idx_basic_output_shape(
    df_folds, conf, train_fn, predict_fn, eval_fn
):
    conf = {**conf, "cv_idx": [{"train": [0, 1], "test": [2, 3]}, {"train": [2, 3], "test": [4, 5]}]}
    result = cross_validation(df_folds, train_fn, predict_fn, eval_fn, conf)

    assert len(result) == 1
    expected_cols = {"mean_pred", "mean_pred_fold0", "mean_pred_fold1"}
    assert expected_cols.issubset(result.columns)


def test_cross_validation_cv_idx_uses_positional_ranges(
    df_folds, conf, predict_fn, eval_fn
):
    seen_train_x = []
    seen_test_x = []

    def train_fn(df_train, conf):
        seen_train_x.append(list(df_train["x"]))
        return {"mean_y": df_train[conf["target"]].mean()}

    def predict_fn(fit, df_test, conf):
        seen_test_x.append(list(df_test["x"]))
        return [fit["mean_y"]] * len(df_test)

    conf = {**conf, "cv_idx": [{"train": [0, 1], "test": [2, 3]}, {"train": [2, 3], "test": [4, 5]}]}
    cross_validation(df_folds, train_fn, predict_fn, eval_fn, conf)

    assert seen_train_x == [[1, 2], [3, 4]]
    assert seen_test_x == [[3, 4], [5, 6]]


def test_cross_validation_cv_idx_takes_precedence_over_fold_column(
    df_folds, conf, train_fn, predict_fn, eval_fn
):
    conf = {**conf, "cv_idx": [{"train": [0, 1, 2], "test": [3, 4, 5]}]}
    result = cross_validation(df_folds, train_fn, predict_fn, eval_fn, conf)

    assert len(result) == 1
    assert "mean_pred_fold1" not in result.columns
    assert "mean_pred_fold0" in result.columns

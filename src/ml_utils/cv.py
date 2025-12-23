import numpy as np
import pandas as pd

from itertools import product
import copy

from typing import Callable, List, Optional


def stepwise_cv(
    df: pd.DataFrame,
    train_fn: Callable,
    predict_fn: Callable,
    eval_fn: Callable,
    conf: dict,
    forward: bool = True,
    start_feats: List = [],
    start_score: Optional[float] = None,
    criterion: str = "roc_auc",
    stop_early: bool = True,
):
    """
    Perform stepwise feature selection using cross-validation.

    Iteratively adds (forward=True) or removes (forward=False) single features,
    selecting at each step the feature set that maximizes the specified
    evaluation criterion computed via cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        Input data; must be compatible with `cross_validation`.
    train_fn, predict_fn, eval_fn : Callable
        Training, prediction, and evaluation functions used by
        `cross_validation`.
    conf : dict
        Configuration dictionary passed to `cross_validation`. Must contain
        a ``"feats"`` key listing available features.
    forward : bool, default True
        If True, perform forward selection; otherwise perform backward elimination.
    start_feats : list, default []
        Initial feature set.
    start_score : float or None, default None
        Baseline score for `start_feats`. If None, it is computed automatically.
    criterion : str, default "roc_auc"
        Metric name used to compare feature sets.
    stop_early : bool, default True
        If True, stop when no improvement over `start_score` is achieved.

    Returns
    -------
    list of dict
        Sequential record of each step, including evaluated features,
        candidate features, and corresponding scores.
    """

    print(
        f"Stepwise CV: forward={forward}, start_feats={start_feats} (n={len(start_feats)}), start_score={start_score}"
    )
    if start_score is None:
        if len(start_feats) == 0:
            start_score = float("-inf")
        else:
            print(f"Evaluating start_feats: {start_feats}...")
            conf_ = copy.deepcopy(conf)
            conf_["feats"] = start_feats
            cv_res = cross_validation(df, train_fn, predict_fn, eval_fn, conf_)
            start_score = cv_res[criterion].values[0]
            print(f"  score: {start_score}")

    candidate_feats = (
        [f for f in conf["feats"] if f not in start_feats]
        if forward
        else start_feats.copy()
    )
    candidate_feat_sets = []
    scores = []
    # Set min_candidates to 2 when backward selection to avoid fitting an empty model
    min_candidates = 1 if forward else 2
    if len(candidate_feats) >= min_candidates:
        for candidate_feat in candidate_feats:
            print(f"Evaluating candidate_feat: {candidate_feat}...")
            conf_ = copy.deepcopy(conf)
            if forward:
                conf_["feats"] = start_feats + [candidate_feat]
            else:
                conf_["feats"] = [f for f in start_feats if f != candidate_feat]
            candidate_feat_sets.append(conf_["feats"])
            cv_res = cross_validation(df, train_fn, predict_fn, eval_fn, conf_)
            scores.append(cv_res[criterion].values[0])
            print(f"  score: {scores[-1]}")
        best_i = np.argmax(scores)

    res_this = [
        {
            "feats": start_feats,
            "score": start_score,
            "candidate_feats": candidate_feats,
            "candidate_scores": scores,
        }
    ]

    if (len(candidate_feats) >= min_candidates) and (
        (not stop_early) or (scores[best_i] > start_score)
    ):
        res = stepwise_cv(
            df,
            train_fn,
            predict_fn,
            eval_fn,
            conf,
            forward=forward,
            start_feats=candidate_feat_sets[best_i],
            start_score=scores[best_i],
            criterion=criterion,
            stop_early=stop_early,
        )
        return res_this + res

    else:
        return res_this


def grid_search_cv(
    df: pd.DataFrame,
    train_fn: Callable,
    predict_fn: Callable,
    eval_fn: Callable,
    conf: dict,
    param_grid: dict,
    pred_param_grid: Optional[dict] = None,
):
    """
    Perform grid search over model hyperparameters using cross-validation.

    Evaluates all combinations in `param_grid` by updating `conf["params"]`
    and running `cross_validation` for each setting.

    Parameters
    ----------
    df : pd.DataFrame
        Input data; must be compatible with `cross_validation`.
    train_fn, predict_fn, eval_fn : Callable
        Training, prediction, and evaluation functions used by
        `cross_validation`.
    conf : dict
        Base configuration dictionary passed to `cross_validation`.
        Must contain a ``"params"`` key.
    param_grid : dict
        Mapping of parameter names to lists of values to evaluate.
    pred_param_grid : dict or None, optional
        Optional grid of prediction-time parameters passed through to
        `cross_validation`.

    Returns
    -------
    pd.DataFrame
        Concatenated evaluation results for all parameter combinations,
        with one column per tuned parameter.
    """

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_param_combinations = list(product(*param_values))

    evals = []
    for param_combination in all_param_combinations:
        params = {param_names[i]: param_combination[i] for i in range(len(param_names))}
        conf_ = copy.deepcopy(conf)
        conf_["params"].update(params)
        print(f"Evaluating params: {params}...")
        df_eval = cross_validation(
            df, train_fn, predict_fn, eval_fn, conf_, pred_param_grid=pred_param_grid
        )
        for name in param_names:
            df_eval[name] = params[name]
        evals.append(df_eval)

    df_evals = pd.concat(evals, ignore_index=True)
    return df_evals


def cross_validation(
    df: pd.DataFrame,
    train_fn: Callable,
    predict_fn: Callable,
    eval_fn: Callable,
    conf: dict,
    pred_param_grid: Optional[dict] = None,
):
    """
    Run cross-validation and aggregate evaluation metrics across folds.

    Trains a model on all but one fold, evaluates on the held-out fold, and
    repeats for all folds defined in ``df.fold``. Optionally evaluates multiple
    prediction-time parameter values (e.g., number of trees) without retraining.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing a ``fold`` column defining CV splits.
    train_fn, predict_fn, eval_fn : Callable
        Training, prediction, and evaluation functions.
    conf : dict
        Configuration dictionary passed to training, prediction, and evaluation.
        Must include a ``"target"`` key.
    pred_param_grid : dict or None, optional
        Optional grid of a single prediction-time parameter to evaluate
        without retraining.

    Returns
    -------
    pd.DataFrame
        Evaluation results aggregated across folds, including per-fold
        metrics and overall metrics, with one row per prediction-time
        parameter value (if any).
    """

    # pred_param_grid is only used at pred time (eg for XGB number of trees)
    assert (pred_param_grid is None) or (
        len(pred_param_grid) <= 1
    ), "Only one eval param supported currently"
    pred_param_name = list(pred_param_grid.keys())[0] if pred_param_grid else None
    pred_param_values = pred_param_grid[pred_param_name] if pred_param_grid else [None]

    # Prepare 2d lists of test and eval dfs: outer list over folds, inner list over eval
    # param (if any)
    dfs_test = []
    dfs_eval = []

    unique_folds = sorted(df.fold.unique())
    for fold in unique_folds:
        print(f"Processing fold {fold}...")
        df_train = df[df.fold != fold]
        df_test = df[df.fold == fold]

        fit = train_fn(df_train, conf)

        dfs_test.append([])
        dfs_eval.append([])

        for pred_param_value in pred_param_values:
            conf_ = copy.deepcopy(conf)
            if pred_param_grid:
                conf_[pred_param_name] = pred_param_value
            pred = predict_fn(fit, df_test, conf_)

            df_test_ = df_test[[conf["target"]]].copy()
            df_test_["pred"] = pred
            dfs_test[-1].append(df_test_)

            df_eval_ = eval_fn(df_test_, conf)
            df_eval_.columns = [f"{col}_fold{fold}" for col in df_eval_.columns]
            dfs_eval[-1].append(df_eval_)

    # Prepare list of eval dfs, one for each pred param value
    dfs_eval_joined = []
    for i_param, pred_param_value in enumerate(pred_param_values):
        df_test_all = pd.concat(
            [dfs_test[i_fold][i_param] for i_fold in range(len(unique_folds))]
        )
        df_eval_all = eval_fn(df_test_all, conf)
        # Concatenate columns from each fold, as well as the overall one
        df_eval = pd.concat(
            [df_eval_all]
            + [dfs_eval[i_fold][i_param] for i_fold in range(len(unique_folds))],
            axis=1,
        )
        if pred_param_value is not None:
            df_eval[pred_param_name] = pred_param_value
        dfs_eval_joined.append(df_eval)

    return pd.concat(dfs_eval_joined, ignore_index=True)


def train_gam(df: pd.DataFrame, conf: dict):
    from pygam import LogisticGAM

    feats = conf["feats"]
    terms = conf["terms"]
    X = df[feats].values
    y = df[conf["target"]].values
    gam = LogisticGAM(terms).fit(X, y)
    return gam


def predict_gam(fit, df: pd.DataFrame, conf: dict):
    feats = conf["feats"]
    X = df[feats].values
    pred = fit.predict_proba(X)
    return pred


def train_xgb(df: pd.DataFrame, conf: dict):
    import xgboost as xgb

    feats = conf["feats"]
    X = df[feats].values
    y = df[conf["target"]].values
    fit = xgb.train(
        params=conf["params"],
        dtrain=xgb.DMatrix(X, label=y, feature_names=feats),
        num_boost_round=conf["n_rounds"],
    )
    return fit


def predict_xgb(fit, df: pd.DataFrame, conf: dict):
    import xgboost as xgb

    feats = conf["feats"]
    X = df[feats].values
    n_trees = conf["n_trees"] if "n_trees" in conf else conf["n_rounds"]
    pred = fit.predict(
        xgb.DMatrix(X, feature_names=feats), iteration_range=[0, n_trees]
    )
    return pred


def eval_roc_auc(df: pd.DataFrame, conf: dict):
    from sklearn.metrics import roc_auc_score

    y_true = df[conf["target"]].values
    y_pred = df["pred"].values
    auc = roc_auc_score(y_true, y_pred)
    return pd.DataFrame({"roc_auc": [auc]})

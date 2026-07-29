# Changelog

## [1.0.4] - 2026-07-29

### Added
- `cross_validation`: support for arbitrary train/test splits via `conf["cv_idx"]`, a list of `{"train": [...], "test": [...]}` dicts giving 0-based row positions per fold (via `df.iloc`). Enables rolling-window CV for time series. Takes precedence over the `df.fold` column when present; falls back to the existing `fold`-column logic otherwise, so existing workflows are unaffected.
- `tests/test_cv.py`: tests for the `cv_idx` path covering output shape, correct positional train/test slicing, and precedence over the `fold` column

## [1.0.3] - 2026-07-29

### Fixed
- `cross_validation`: fixed `NameError` on every call caused by a leftover reference to `pred_param_grid` (removed in the flat-params refactor); now correctly guards on `pred_param_name is not None`

### Added
- `tests/test_cv.py`: tests for `cross_validation` covering output shape, fold exclusion, and the `pred_param_name`/`pred_param_values` sweep (including the no-sweep default path)
- `tests/test_grid_search_cv.py`: tests for `grid_search_cv` covering full combination coverage, params reaching `train_fn`, base `conf` not being mutated, and combining the grid with a `pred_param` sweep
- `pytest` as an optional `test` dependency; `pythonpath = ["src"]` pytest config so tests import `ml_utils` without an editable install

## [1.0.2] - 2026-07-29

### Fixed
- `stepwise_cv`: changed `start_feats` default from `[]` to `None` to avoid mutable default argument
- `feature_sweep_plot`: clip predictions before logit transform to avoid ±inf when model outputs exactly 0 or 1
- `cross_validation` / `grid_search_cv`: replaced `pred_param_grid: dict` with flat `pred_param_name: str` and `pred_param_values: list` parameters

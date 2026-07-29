# Changelog

## [Unreleased]

### Fixed
- `stepwise_cv`: changed `start_feats` default from `[]` to `None` to avoid mutable default argument
- `feature_sweep_plot`: clip predictions before logit transform to avoid ±inf when model outputs exactly 0 or 1
- `cross_validation` / `grid_search_cv`: replaced `pred_param_grid: dict` with flat `pred_param_name: str` and `pred_param_values: list` parameters

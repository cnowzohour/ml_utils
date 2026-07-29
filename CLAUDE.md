# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# ML Utils

Some utility code for machine learning workflows.

## Development Commands
- Run linter: `ruff check .`
- Run tests: `pytest`
- Run a single test: `pytest tests/test_file.py::test_name`

## Codebase Architecture

The library lives under `src/ml_utils/` and is structured around three independent modules:

- **`cv.py`** — Cross-validation engine and model adapters. The central function is `cross_validation`, which takes `train_fn`, `predict_fn`, `eval_fn` callables and a `conf` dict, splits on a pre-assigned `df.fold` column, and returns a DataFrame of per-fold and overall metrics. `grid_search_cv` and `stepwise_cv` compose on top of it. Built-in adapters for XGBoost (`train_xgb` / `predict_xgb`) and pyGAM (`train_gam` / `predict_gam`) follow the same callable signature. `eval_roc_auc` is the only built-in evaluator; others must be supplied by the caller.

- **`plotting.py`** — Model inspection plots. `feature_sweep_plot` sweeps individual features over a range while holding others constant. `pdp_plot` wraps sklearn's `PartialDependenceDisplay`. `EstimatorWrapper` is the glue class that makes arbitrary `(fit, conf)` pairs look like a sklearn estimator so sklearn inspection tools can consume them.

- **`util.py`** — Minimal helpers: `timed` for wall-clock logging, `save_pickle` / `load_pickle` for serialization.

The `conf` dict is the shared configuration contract across all `cv.py` functions. It must contain at minimum `"feats"` (feature list) and `"target"` (label column name); model-specific keys like `"params"` and `"n_rounds"` are added per adapter.

## Post-Task Requirements
- Whenever you create or modify code files, you must document the changes in `CHANGELOG.md` under an `[Unreleased]` section before completing the interaction.

## Workflow & Safety Rules
- **Proposal First:** For any non-trivial task, propose the planned file structure and changes in pseudo-code/text first. Wait for user approval before executing.
- **Incremental Progress:** Do not attempt to write an entire pipeline at once. Break tasks down and execute them one file at a time.
- **Ask, don't assume:** If something is unclear, ask before writing a single line. Never make silent assumptions about intent, architecture, or requirements.
- **Simplest solution first:** Implement the simplest solution for simple problems, better solutions for harder problems. Do not over-engineer or add flexibility that isn't needed yet.
- **Don't touch unrelated code:** Don't touch unrelated code but please do surface bad code or design smells you discover with me so we can address them as a separate issue.
- **Flag uncertainty explicitly:** If you're unsure about something, see "ask, don't assume" above. If it makes sense to do so, conduct a small, localised and low-risk experiment and bring the hypothesis and results to me to discuss. Confidence without certainty causes more damage than admitting a gap.

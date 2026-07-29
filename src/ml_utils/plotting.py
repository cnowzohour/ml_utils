import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable


def feature_sweep_plot(
    df_const: pd.DataFrame,
    feat_ranges: dict,
    pred_fn: Callable,
    title: str = "",
    plot_logits: bool = True,
    nrow: int = 1,
):
    """
    Plot model predictions while sweeping individual features over a specified range,
    holding all other features constant.

    For each feature, its column in ``df_const`` is replaced with linearly spaced values
    between the provided minimum and maximum, and the model is evaluated on these inputs.

    Parameters
    ----------
    df_const : pd.DataFrame
        Constant reference inputs (e.g., repeated mean feature values). The number of
        rows controls the sweep resolution.
    feat_ranges : dict
        Mapping of feature name to ``(vmin, vmax)`` sweep range.
    pred_fn : Callable
        Function mapping a DataFrame to predicted probabilities.
    title : str, default=""
        Figure title.
    plot_logits : bool, default=True
        Plot log-odds instead of probabilities.
    nrow : int, default=1
        Number of rows in the subplot grid.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    axs : numpy.ndarray of matplotlib.axes.Axes
        Axes for each feature plot.
    """

    ncol = len(feat_ranges) // nrow + int(len(feat_ranges) % nrow > 0)
    fig, axs = plt.subplots(
        nrow,
        ncol,
        figsize=(5 * ncol, 4 * nrow),
        sharey=True,
        squeeze=False,
    )
    axs = axs.flatten()
    for i, (feat, (vmin, vmax)) in enumerate(feat_ranges.items()):
        df_const_ = df_const.copy()
        df_const_[feat] = np.linspace(vmin, vmax, len(df_const))
        preds_pdp_ = pred_fn(df_const_)
        if plot_logits:
            preds_clipped = np.clip(preds_pdp_, 1e-7, 1 - 1e-7)
            logits = np.log(preds_clipped / (1 - preds_clipped))
            axs[i].plot(df_const_[feat], logits)
            ylabel = "Logits"
        else:
            axs[i].plot(df_const_[feat], preds_pdp_)
            ylabel = "Probability"
        if i == 0:
            axs[i].set_ylabel(ylabel)
        axs[i].set_xlabel(feat)
    fig.suptitle(title)

    return fig, axs


class EstimatorWrapper:
    """
    Adapter that exposes an arbitrary fitted model and prediction function through
    a scikit-learn–compatible estimator interface.

    This wrapper enables the use of sklearn inspection utilities (e.g.
    PartialDependenceDisplay) with non-sklearn models or custom prediction pipelines
    by providing the required attributes and a ``predict`` method.
    """

    def __init__(self, pred_fn, logit_transform: bool = False):
        # Scikit-learn checks for this attribute to make sure the estimator is fitted
        self.feature_names_in_ = []
        # Required for some sklearn inspections
        self._estimator_type = "regressor"
        self.pred_fn = pred_fn
        self.logit_transform = logit_transform

    def predict(self, X):
        preds = self.pred_fn(X)
        if self.logit_transform:
            preds = np.log(preds / (1 - preds))
        return preds

    def fit(self, X, y=None):
        pass


def pdp_plot(
    feat_ranges: dict,
    model_wrapper: EstimatorWrapper,
    df: pd.DataFrame,
    ylabel: str,
    title: str,
    grid_resolution: int = 200,
    nrow: int = 1,
):
    """
    Plot partial dependence curves for multiple features side by side.

    Uses sklearn's PartialDependenceDisplay with a wrapped estimator to compute
    and display PDPs on a shared y-axis. Feature-specific x-axis limits are
    applied from ``feat_ranges``.

    Parameters
    ----------
    feat_ranges : dict
        Mapping from feature name to ``(vmin, vmax)`` x-axis limits.
    model_wrapper : EstimatorWrapper
        scikit-learn–compatible wrapper exposing a ``predict`` method.
    df : pd.DataFrame
        Input data used to compute partial dependence.
    ylabel : str
        Label for the shared y-axis (shown on the left-most subplot).
    title : str
        Figure-level title.
    grid_resolution : int, default=200
        Number of points used to evaluate each partial dependence curve.
    nrow : int, default=1
        Number of rows in the subplot grid.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    axs : numpy.ndarray of matplotlib.axes.Axes
        Axes for each feature plot.
    """

    ncol = len(feat_ranges) // nrow + int(len(feat_ranges) % nrow > 0)
    features_to_plot = list(feat_ranges.keys())
    fig, axs = plt.subplots(
        nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False, sharey=True
    )
    axs = axs.flatten()
    from sklearn.inspection import PartialDependenceDisplay

    display = PartialDependenceDisplay.from_estimator(
        model_wrapper,
        df,
        features=features_to_plot,
        percentiles=(0, 1),
        grid_resolution=grid_resolution,
        ax=axs,
    )

    all_y_min = []
    all_y_max = []
    for ax, feat in zip(np.atleast_1d(axs).ravel(), features_to_plot):
        vmin, vmax = feat_ranges[feat]
        ax.set_xlim(vmin, vmax)

        # Rescale Y based on what is actually visible in this specific X-range
        # 1. Get the line object from the plot
        line = ax.get_lines()[0]
        x_data = line.get_xdata()
        y_data = line.get_ydata()

        # 2. Filter y_data to only include points within the visible vmin/vmax
        visible_y = y_data[(x_data >= vmin) & (x_data <= vmax)]

        if len(visible_y) > 0:
            all_y_min.append(np.min(visible_y))
            all_y_max.append(np.max(visible_y))

    # 3. Apply the global "visible" limits to all axes
    if all_y_min and all_y_max:
        y_min = min(all_y_min)
        y_max = max(all_y_max)
        # Add 10% padding so the lines don't touch the top/bottom edges
        padding = (y_max - y_min) * 0.1
        for ax in np.atleast_1d(axs).ravel():
            ax.set_ylim(y_min - padding, y_max + padding)

    axs[0].set_ylabel(ylabel)
    for ax in axs[1:]:
        ax.set_ylabel(None)
    fig.suptitle(title)

    return fig, axs

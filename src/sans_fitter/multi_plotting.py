"""Figure assembly for a simultaneous fit: one panel per dataset, on one figure.

The default is a stack rather than an overlay. Datasets in a joint fit routinely
differ by orders of magnitude in intensity and cover different Q ranges — that is
usually *why* they are being fitted together — so a single pair of axes would
compress the interesting one into a line. Each dataset gets its own axes, with
its residual panel tied to it, and the figure grows downwards.

Overlaying comparable curves is still useful and available through
``overlay=True``; it is offered as a second view rather than the only one.
"""

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .plotting import COMPONENT_CURVE_COLORS, error_bars, resolve_show, subset

__all__ = ['DatasetPanel', 'plot_multi']

#: Vertical pixels each dataset's panel (plus residuals) gets. Fixed per panel
#: rather than shared out of a fixed total, so five datasets produce a taller
#: figure instead of five unreadable strips.
PANEL_HEIGHT_PIXELS = 420
RESIDUAL_HEIGHT_PIXELS = 140
FIGURE_WIDTH_PIXELS = 900


@dataclass(slots=True)
class DatasetPanel:
    """One dataset's contribution to a multi-panel figure.

    Carries the observation arrays themselves rather than a dataset object, so a
    panel built from a fit result cannot silently pick up observations that
    belong to a different one. :meth:`from_dataset` is the convenience for the
    preview path, where "current data" is exactly what is wanted.
    """

    name: str
    model: str
    #: Full-length measured arrays.
    q: np.ndarray
    intensity: np.ndarray
    uncertainty: np.ndarray | None
    dq: np.ndarray | None
    #: Model intensities at the fitted points only.
    fitted_curve: np.ndarray
    #: Boolean selection of fitted points, or None when every point was used.
    fit_index: np.ndarray | None
    #: Residuals at the fitted points, or None when the data carries no dI.
    residuals: np.ndarray | None
    weight: float = 1.0
    #: Per-component curves on the fitted points, for a '+' mixture model.
    component_curves: dict[str, np.ndarray] | None = None

    @classmethod
    def from_dataset(
        cls,
        name: str,
        model: str,
        data: Any,
        fitted_curve: np.ndarray,
        fit_index: np.ndarray | None,
        residuals: np.ndarray | None,
        weight: float = 1.0,
        component_curves: dict[str, np.ndarray] | None = None,
    ) -> 'DatasetPanel':
        """Build a panel from a live ``Data1D``, for previews."""
        return cls(
            name=name,
            model=model,
            q=np.asarray(data.x),
            intensity=np.asarray(data.y),
            uncertainty=None if data.dy is None else np.asarray(data.dy),
            dq=None if getattr(data, 'dx', None) is None else np.asarray(data.dx),
            fitted_curve=fitted_curve,
            fit_index=fit_index,
            residuals=residuals,
            weight=weight,
            component_curves=component_curves,
        )

    def selection(self) -> np.ndarray:
        if self.fit_index is None:
            return np.ones(len(np.asarray(self.q)), dtype=bool)
        return np.asarray(self.fit_index, dtype=bool)


def _axis_reference(row: int) -> str:
    """Plotly's name for the x axis of row *row* in a single-column figure."""
    return 'x' if row == 1 else f'x{row}'


def plot_multi(
    panels: list[DatasetPanel],
    title: str,
    show_residuals: bool = True,
    log_scale: bool = True,
    show: bool | None = None,
    preview: bool = False,
    residual_label: str = 'Residuals (σ)',
    overlay: bool = False,
    show_components: bool = False,
) -> go.Figure:
    """Build the figure for a preview or a completed simultaneous fit.

    Args:
        panels: One entry per dataset, in the order they should appear.
        title: Figure title, normally carrying the joint goodness of fit.
        show_residuals: Give each dataset a residual panel beneath its curve.
        log_scale: Log axes, as for a single fit.
        show: ``True`` to call ``fig.show()``, ``False`` to only return the
            figure. The default follows the single-fit convention: show, except
            in a notebook where the returned figure is rendered anyway.
        preview: Label the curves as a model preview rather than a fit.
        residual_label: Y-axis label for the residual panels. The caller sets
            this so residuals in sigma units and objective residuals cannot be
            mistaken for each other.
        overlay: Draw every dataset on one pair of axes instead of stacking.
        show_components: For a '+' mixture model, overlay one dashed curve per
            component beneath the total. A no-op where no component curves were
            computed.

    Raises:
        ValueError: If there is nothing to plot, or a fitted curve does not
            match the number of fitted points in its dataset.
    """
    if not panels:
        raise ValueError('Nothing to plot: the analysis has no datasets.')

    for panel in panels:
        expected = int(panel.selection().sum())
        if len(np.asarray(panel.fitted_curve)) != expected:
            raise ValueError(
                f"Dataset '{panel.name}': the model curve has "
                f'{len(np.asarray(panel.fitted_curve))} points but {expected} were fitted.'
            )

    figure = (
        _overlay_figure(panels, log_scale, preview)
        if overlay
        else _stacked_figure(
            panels, show_residuals, log_scale, preview, residual_label, show_components
        )
    )

    figure.update_layout(title=title, template='plotly_white', width=FIGURE_WIDTH_PIXELS)
    if resolve_show(show):
        figure.show()
    return figure


def _stacked_figure(
    panels: list[DatasetPanel],
    show_residuals: bool,
    log_scale: bool,
    preview: bool,
    residual_label: str,
    show_components: bool,
) -> go.Figure:
    rows_per_panel = 2 if show_residuals else 1
    rows = len(panels) * rows_per_panel
    heights: list[float] = []
    for _ in panels:
        heights.append(PANEL_HEIGHT_PIXELS)
        if show_residuals:
            heights.append(RESIDUAL_HEIGHT_PIXELS)
    total = float(sum(heights))

    # One title per row; the residual rows under each panel get a blank so the
    # dataset name sits above its curve rather than above its residuals.
    titles: list[str] = []
    for panel in panels:
        weight = f' (weight {panel.weight:g})' if panel.weight != 1.0 else ''
        titles.append(f'{panel.name} — {panel.model}{weight}')
        titles.extend([''] * (rows_per_panel - 1))

    figure = make_subplots(
        rows=rows,
        cols=1,
        row_heights=[height / total for height in heights],
        # Deliberately not shared: Q ranges differ between datasets, and a
        # shared axis would force the widest range onto every panel.
        shared_xaxes=False,
        vertical_spacing=min(0.04, 0.5 / max(rows - 1, 1)),
        subplot_titles=titles,
    )

    axis_type = 'log' if log_scale else 'linear'
    for position, panel in enumerate(panels):
        data_row = position * rows_per_panel + 1
        for trace in _panel_traces(panel, preview, position == 0, show_components):
            figure.add_trace(trace, row=data_row, col=1)
        figure.update_yaxes(title_text='I(Q)', type=axis_type, row=data_row, col=1)
        figure.update_xaxes(type=axis_type, row=data_row, col=1)

        if not show_residuals:
            figure.update_xaxes(title_text='Q (Å⁻¹)', row=data_row, col=1)
            continue

        residual_row = data_row + 1
        if panel.residuals is not None:
            figure.add_trace(
                go.Scatter(
                    x=subset(panel.q, panel.selection()),
                    y=np.asarray(panel.residuals, dtype=float),
                    mode='markers',
                    name=f'{panel.name} residuals',
                    marker={'size': 5},
                    opacity=0.6,
                    showlegend=False,
                ),
                row=residual_row,
                col=1,
            )
        figure.add_hline(y=0, line_dash='dash', line_color='gray', row=residual_row, col=1)
        figure.update_yaxes(
            title_text=residual_label if panel.residuals is not None else 'Residuals (no dI)',
            row=residual_row,
            col=1,
        )
        figure.update_xaxes(
            title_text='Q (Å⁻¹)',
            type=axis_type,
            # Tie the residuals to the curve above them, so zooming one dataset
            # keeps its two panels aligned without dragging the others along.
            matches=_axis_reference(data_row),
            row=residual_row,
            col=1,
        )

    figure.update_layout(height=int(total) + 80 * len(panels))
    return figure


def _panel_traces(
    panel: DatasetPanel, preview: bool, legend: bool, show_components: bool = False
) -> list[go.Scatter]:
    index = panel.selection()
    excluded = ~index
    traces = [
        go.Scatter(
            x=subset(panel.q, index),
            y=subset(panel.intensity, index),
            error_y=error_bars(subset(panel.uncertainty, index)),
            error_x=error_bars(subset(panel.dq, index)),
            mode='markers',
            name='Experimental Data',
            legendgroup='data',
            showlegend=legend,
            opacity=0.6,
            marker={'size': 5, 'color': '#636efa'},
        )
    ]
    if excluded.any():
        traces.append(
            go.Scatter(
                x=subset(panel.q, excluded),
                y=subset(panel.intensity, excluded),
                mode='markers',
                name='Excluded Data',
                legendgroup='excluded',
                showlegend=legend,
                opacity=0.4,
                marker={'size': 5, 'color': 'lightgray'},
            )
        )
    traces.append(
        go.Scatter(
            x=subset(panel.q, index),
            y=np.asarray(panel.fitted_curve, dtype=float),
            mode='lines',
            name='Model (current parameters)' if preview else 'Joint fit',
            legendgroup='model',
            showlegend=legend,
            line={'color': 'red', 'width': 2},
        )
    )

    if show_components and panel.component_curves:
        q = subset(panel.q, index)
        for position, (label, curve) in enumerate(panel.component_curves.items()):
            values = np.asarray(curve, dtype=float)
            if len(values) != len(q):
                warnings.warn(
                    f"Component curve '{label}' of dataset '{panel.name}' has "
                    f'{len(values)} points but {len(q)} were fitted; omitting it.',
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            traces.append(
                go.Scatter(
                    x=q,
                    y=values,
                    mode='lines',
                    name=f'{panel.name}: {label}',
                    line={
                        'dash': 'dash',
                        'width': 1.5,
                        'color': COMPONENT_CURVE_COLORS[position % len(COMPONENT_CURVE_COLORS)],
                    },
                )
            )
    return traces


def _overlay_figure(panels: list[DatasetPanel], log_scale: bool, preview: bool) -> go.Figure:
    """Every dataset on one pair of axes, colour-coded by dataset."""
    figure = go.Figure()
    for position, panel in enumerate(panels):
        colour = COMPONENT_CURVE_COLORS[position % len(COMPONENT_CURVE_COLORS)]
        index = panel.selection()
        figure.add_trace(
            go.Scatter(
                x=subset(panel.q, index),
                y=subset(panel.intensity, index),
                error_y=error_bars(subset(panel.uncertainty, index)),
                mode='markers',
                name=panel.name,
                legendgroup=panel.name,
                opacity=0.5,
                marker={'size': 5, 'color': colour},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=subset(panel.q, index),
                y=np.asarray(panel.fitted_curve, dtype=float),
                mode='lines',
                name=f'{panel.name} {"model" if preview else "fit"}',
                legendgroup=panel.name,
                line={'color': colour, 'width': 2},
            )
        )
    axis_type = 'log' if log_scale else 'linear'
    figure.update_xaxes(title_text='Q (Å⁻¹)', type=axis_type)
    figure.update_yaxes(title_text='I(Q)', type=axis_type)
    figure.update_layout(height=600)
    return figure

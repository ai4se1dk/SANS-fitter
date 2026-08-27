import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data.loader import has_real_data
from .results import (
    MIN_POSTERIOR_PARAMETER_COUNT,
    FitResultContract,
    PosteriorSummary,
    resolve_fit_index,
)

DEFAULT_POSTERIOR_PREDICTIVE_DRAWS = 50
POSTERIOR_PAIR_SCATTER_MAX_POINTS = 750
PAIR_PLOT_CELL_SIZE_PIXELS = 250
TRACE_PLOT_ROW_HEIGHT_PIXELS = 200
POSTERIOR_HISTOGRAM_COLOR = 'rgba(120, 120, 120, 0.5)'
POSTERIOR_MEDIAN_LINE_COLOR = 'rgb(80, 80, 80)'
POSTERIOR_POINT_ESTIMATE_LINE_COLOR = 'rgb(214, 39, 40)'
POSTERIOR_INTERVAL_68_FILL_COLOR = 'rgba(99, 110, 250, 0.15)'
POSTERIOR_INTERVAL_95_LINE_COLOR = 'rgba(214, 39, 40, 0.5)'
POSTERIOR_BAND_FILL_COLOR = 'rgba(99, 110, 250, 0.25)'
POSTERIOR_DRAW_LINE_COLOR = 'rgba(140, 140, 140, 0.25)'
POSTERIOR_SCATTER_MARKER_COLOR = 'rgba(99, 110, 250, 0.25)'
POSTERIOR_POINT_ESTIMATE_TRACE_NAME = 'Best posterior sample'
POSTERIOR_PREDICTIVE_INTERVAL_TRACE_NAME = '95% credible interval'
MARGINAL_DENSITY_TRACE_NAME = 'Marginal density'
MEASURED_TRACE_NAME = 'Measured'

# Qualitative color cycle for per-component curves (plotly's default palette).
COMPONENT_CURVE_COLORS = [
    '#636efa',
    '#ef553b',
    '#00cc96',
    '#ab63fa',
    '#ffa15a',
    '#19d3f3',
    '#ff6692',
    '#b6e880',
    '#ff97ff',
    '#fecb52',
]


def _running_in_notebook() -> bool:
    """Return True when executing inside a Jupyter kernel.

    In notebooks a returned Figure is rendered by the frontend, so calling
    fig.show() as well would display the plot twice.
    """
    try:
        from IPython.core.getipython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and 'ZMQInteractiveShell' in type(shell).__name__


def _error_bars(arr) -> dict | None:
    """Build a plotly error-bar spec, or None when the column carries no data."""
    if not has_real_data(arr):
        return None
    return {'type': 'data', 'array': arr, 'visible': True}


def _subset(arr, index):
    """Slice an optional data column by a boolean index."""
    return None if arr is None else np.asarray(arr)[index]


def _resolve_show(show: bool | None) -> bool:
    return not _running_in_notebook() if show is None else show


def plot_fit(
    data,
    fit_result: FitResultContract | None,
    model_name: str | None,
    show_residuals: bool = True,
    log_scale: bool = True,
    show: bool | None = None,
    show_components: bool = False,
) -> go.Figure:
    """Plot experimental data and, when available, the fitted model curve.

    Args:
        show_components: When True and the fit result carries per-component
            curves ('+' mixture models), draw one dashed curve per component.
            A no-op when no component curves are available.
        show: If True, call fig.show(); if False, only return the figure.
            Default (None) shows the figure except in Jupyter, where the
            returned figure is rendered by the notebook itself.
    """
    if data is None:
        raise ValueError('No data to plot. Use load_data() first.')

    error_y = _error_bars(data.dy)
    error_x = _error_bars(data.dx)

    if fit_result is None:
        print('No fit results available. Plotting data only.')
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data.x,
                y=data.y,
                error_y=error_y,
                error_x=error_x,
                mode='markers',
                name='Data',
                opacity=0.6,
            )
        )
        fig.update_layout(
            title='SANS Data',
            xaxis_title='Q (Å⁻¹)',
            yaxis_title='I(Q)',
            xaxis_type='log' if log_scale else 'linear',
            yaxis_type='log' if log_scale else 'linear',
            template='plotly_white',
        )
        if _resolve_show(show):
            fig.show()
        return fig

    i_fit = fit_result.require_fitted_curve()
    index = resolve_fit_index(fit_result.artifacts.fit_index, len(data.x))
    if len(i_fit) != int(index.sum()):
        raise ValueError(
            f'Fitted curve length ({len(i_fit)}) does not match the number of '
            f'fitted points ({int(index.sum())}).'
        )

    q = _subset(data.x, index)
    y = _subset(data.y, index)
    dy = _subset(data.dy, index)
    dx = _subset(data.dx, index)
    residuals = (y - i_fit) / dy
    excluded = ~index

    if show_residuals:
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.75, 0.25],
            shared_xaxes=True,
            vertical_spacing=0.05,
        )
    else:
        fig = go.Figure()

    data_trace = go.Scatter(
        x=q,
        y=y,
        error_y=_error_bars(dy),
        error_x=_error_bars(dx),
        mode='markers',
        name='Experimental Data',
        opacity=0.6,
        marker={'size': 6},
    )

    excluded_trace = None
    if excluded.any():
        excluded_trace = go.Scatter(
            x=_subset(data.x, excluded),
            y=_subset(data.y, excluded),
            error_y=_error_bars(_subset(data.dy, excluded)),
            error_x=_error_bars(_subset(data.dx, excluded)),
            mode='markers',
            name='Excluded Data',
            opacity=0.4,
            marker={'size': 6, 'color': 'lightgray'},
        )

    fit_trace = go.Scatter(
        x=q,
        y=i_fit,
        mode='lines',
        name='Fitted Model',
        line={'color': 'red', 'width': 2},
    )

    component_traces = []
    if show_components and fit_result.artifacts.component_curves:
        for i, (label, curve) in enumerate(fit_result.artifacts.component_curves.items()):
            curve = np.asarray(curve)
            if len(curve) != len(q):
                warnings.warn(
                    f"Component curve '{label}' has {len(curve)} points but the fit "
                    f'has {len(q)}; omitting it from the plot.',
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            component_traces.append(
                go.Scatter(
                    x=q,
                    y=curve,
                    mode='lines',
                    name=label,
                    line={
                        'dash': 'dash',
                        'width': 1.5,
                        'color': COMPONENT_CURVE_COLORS[i % len(COMPONENT_CURVE_COLORS)],
                    },
                )
            )

    if show_residuals:
        fig.add_trace(data_trace, row=1, col=1)
        if excluded_trace is not None:
            fig.add_trace(excluded_trace, row=1, col=1)
        fig.add_trace(fit_trace, row=1, col=1)
        for trace in component_traces:
            fig.add_trace(trace, row=1, col=1)
        fig.add_trace(
            go.Scatter(
                x=q,
                y=residuals,
                mode='markers',
                name='Residuals',
                marker={'size': 6},
                opacity=0.6,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=0, line_dash='dash', line_color='gray', row=2, col=1)
        fig.update_xaxes(
            title_text='Q (Å⁻¹)',
            type='log' if log_scale else 'linear',
            row=2,
            col=1,
        )
        fig.update_yaxes(
            title_text='I(Q)',
            type='log' if log_scale else 'linear',
            row=1,
            col=1,
        )
        fig.update_yaxes(title_text='Residuals (σ)', row=2, col=1)
        fig.update_xaxes(type='log' if log_scale else 'linear', row=1, col=1)
    else:
        fig.add_trace(data_trace)
        if excluded_trace is not None:
            fig.add_trace(excluded_trace)
        fig.add_trace(fit_trace)
        for trace in component_traces:
            fig.add_trace(trace)
        fig.update_xaxes(
            title_text='Q (Å⁻¹)',
            type='log' if log_scale else 'linear',
        )
        fig.update_yaxes(
            title_text='I(Q)',
            type='log' if log_scale else 'linear',
        )

    fig.update_layout(
        title=f'SANS Fit: {model_name} (χ² = {fit_result.chisq:.4f})',
        template='plotly_white',
        height=800 if show_residuals else 500,
        width=900,
    )

    if _resolve_show(show):
        fig.show()
    return fig


def _resolve_posterior_params(
    posterior: PosteriorSummary, params: list[str] | None
) -> list[tuple[str, int]]:
    """Return (label, chain column) pairs for a requested parameter subset."""
    if params is None:
        return [(name, i) for i, name in enumerate(posterior.labels)]
    return [(name, posterior.index_of(name)) for name in params]


def plot_posterior_pairs(
    posterior: PosteriorSummary,
    params: list[str] | None = None,
    show_contours: bool = True,
    show: bool | None = None,
) -> go.Figure:
    """Corner plot: marginal densities on the diagonal, sample clouds below.

    Args:
        posterior: Posterior summary from a Bayesian fit.
        params: Optional subset of parameter names to plot (default: all).
        show_contours: Overlay density contours on the off-diagonal panels.
        show: Same display convention as plot_fit.
    """
    selected = _resolve_posterior_params(posterior, params)
    n = len(selected)
    if n < MIN_POSTERIOR_PARAMETER_COUNT:
        raise ValueError(
            f'A pair plot needs at least {MIN_POSTERIOR_PARAMETER_COUNT} varying '
            f'parameters (got {n}). Use plot_param_distribution() for a single '
            'parameter.'
        )

    samples = posterior.samples
    n_samples = samples.shape[0]
    if n_samples > POSTERIOR_PAIR_SCATTER_MAX_POINTS:
        rng = np.random.default_rng(0)
        scatter_rows = rng.choice(n_samples, POSTERIOR_PAIR_SCATTER_MAX_POINTS, replace=False)
    else:
        scatter_rows = np.arange(n_samples)

    fig = make_subplots(
        rows=n,
        cols=n,
        horizontal_spacing=0.03,
        vertical_spacing=0.03,
    )

    for row_pos, (row_name, row_index) in enumerate(selected, start=1):
        for col_pos, (col_name, col_index) in enumerate(selected, start=1):
            if col_pos > row_pos:
                continue
            if col_pos == row_pos:
                fig.add_trace(
                    go.Histogram(
                        x=samples[:, row_index],
                        histnorm='probability density',
                        marker={'color': POSTERIOR_HISTOGRAM_COLOR},
                        name=MARGINAL_DENSITY_TRACE_NAME,
                        showlegend=False,
                    ),
                    row=row_pos,
                    col=col_pos,
                )
                fig.add_vline(
                    x=posterior.median[row_name],
                    line_color=POSTERIOR_MEDIAN_LINE_COLOR,
                    line_width=1,
                    row=row_pos,
                    col=col_pos,
                )
                for bound in posterior.ci_95[row_name]:
                    fig.add_vline(
                        x=bound,
                        line_color=POSTERIOR_INTERVAL_95_LINE_COLOR,
                        line_dash='dash',
                        line_width=1,
                        row=row_pos,
                        col=col_pos,
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=samples[scatter_rows, col_index],
                        y=samples[scatter_rows, row_index],
                        mode='markers',
                        marker={'size': 3, 'color': POSTERIOR_SCATTER_MARKER_COLOR},
                        showlegend=False,
                    ),
                    row=row_pos,
                    col=col_pos,
                )
                if show_contours:
                    fig.add_trace(
                        go.Histogram2dContour(
                            x=samples[:, col_index],
                            y=samples[:, row_index],
                            ncontours=6,
                            contours={'coloring': 'lines'},
                            colorscale='Blues',
                            showscale=False,
                            line={'width': 1},
                            showlegend=False,
                        ),
                        row=row_pos,
                        col=col_pos,
                    )
            if row_pos == n:
                fig.update_xaxes(title_text=col_name, row=row_pos, col=col_pos)
            if col_pos == 1 and row_pos > 1:
                fig.update_yaxes(title_text=row_name, row=row_pos, col=col_pos)

    size = max(PAIR_PLOT_CELL_SIZE_PIXELS * n, 500)
    fig.update_layout(
        title='Posterior pair plot',
        template='plotly_white',
        height=size,
        width=size,
        showlegend=False,
    )

    if _resolve_show(show):
        fig.show()
    return fig


def plot_param_distribution(
    posterior: PosteriorSummary,
    param: str,
    bins: int = 50,
    show: bool | None = None,
) -> go.Figure:
    """Marginal posterior distribution for a single parameter.

    Shows the sample histogram with the median, the best posterior sample,
    the shaded 68% credible interval, and the 95% interval bounds.
    """
    index = posterior.index_of(param)
    values = posterior.samples[:, index]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=bins,
            histnorm='probability density',
            marker={'color': POSTERIOR_HISTOGRAM_COLOR},
            name=MARGINAL_DENSITY_TRACE_NAME,
        )
    )

    lo68, hi68 = posterior.ci_68[param]
    fig.add_vrect(
        x0=lo68,
        x1=hi68,
        fillcolor=POSTERIOR_INTERVAL_68_FILL_COLOR,
        line_width=0,
        annotation_text='68% CI',
        annotation_position='top left',
    )
    for bound in posterior.ci_95[param]:
        fig.add_vline(
            x=bound,
            line_color=POSTERIOR_INTERVAL_95_LINE_COLOR,
            line_dash='dash',
            line_width=1,
        )
    fig.add_vline(
        x=posterior.median[param],
        line_color=POSTERIOR_MEDIAN_LINE_COLOR,
        line_width=1.5,
        annotation_text='median',
    )
    fig.add_vline(
        x=posterior.best[param],
        line_color=POSTERIOR_POINT_ESTIMATE_LINE_COLOR,
        line_dash='dot',
        line_width=1.5,
        annotation_text='best',
    )

    fig.update_layout(
        title=f'Posterior distribution: {param}',
        xaxis_title=param,
        yaxis_title='Probability density',
        template='plotly_white',
        height=500,
        width=700,
        showlegend=False,
    )

    if _resolve_show(show):
        fig.show()
    return fig


def plot_posterior_predictive(
    data: Any,
    posterior: PosteriorSummary,
    model_eval: Callable[[dict[str, float]], np.ndarray],
    style: str = 'band',
    n_draws: int = DEFAULT_POSTERIOR_PREDICTIVE_DRAWS,
    fit_index: Any = None,
    log_scale: bool = True,
    show: bool | None = None,
) -> go.Figure:
    """Posterior predictive check: credible band and/or draws over the data.

    Args:
        data: Loaded SANS data object (x, y, dy).
        posterior: Posterior summary from a Bayesian fit.
        model_eval: Callable mapping {param_name: value} to a model intensity
            curve over the fitted Q points.
        style: 'band' (95% credible interval), 'draws' (sampled curves), or
            'band+draws'.
        n_draws: Number of posterior samples to evaluate. The chain is
            subsampled transparently; evaluation cost scales linearly.
        fit_index: Optional boolean mask of fitted points (None = all points).
        log_scale: Use log axes.
        show: Same display convention as plot_fit.
    """
    if data is None:
        raise ValueError('No data to plot. Use load_data() first.')
    if style not in ('band', 'draws', 'band+draws'):
        raise ValueError(f"style must be 'band', 'draws', or 'band+draws' (got '{style}').")

    index = resolve_fit_index(fit_index, len(data.x))
    q = np.asarray(data.x)[index]
    y = np.asarray(data.y)[index]
    dy = _subset(data.dy, index)

    n_samples = posterior.n_samples
    rng = np.random.default_rng(0)
    n_eval = min(n_draws, n_samples)
    chosen = rng.choice(n_samples, n_eval, replace=False)

    curves = []
    for row in posterior.samples[chosen]:
        curve = np.asarray(model_eval(dict(zip(posterior.labels, row, strict=True))))
        if len(curve) != len(q):
            raise ValueError(
                f'model_eval returned a curve of length {len(curve)}, expected '
                f'{len(q)} (the number of fitted points).'
            )
        curves.append(curve)
    curves = np.asarray(curves)

    fig = go.Figure()

    if style in ('band', 'band+draws'):
        band_lo, band_hi = np.percentile(curves, [2.5, 97.5], axis=0)
        fig.add_trace(
            go.Scatter(
                x=q,
                y=band_hi,
                mode='lines',
                line={'width': 0},
                showlegend=False,
                hoverinfo='skip',
            )
        )
        fig.add_trace(
            go.Scatter(
                x=q,
                y=band_lo,
                mode='lines',
                line={'width': 0},
                fill='tonexty',
                fillcolor=POSTERIOR_BAND_FILL_COLOR,
                name=POSTERIOR_PREDICTIVE_INTERVAL_TRACE_NAME,
            )
        )

    if style in ('draws', 'band+draws'):
        for draw_number, curve in enumerate(curves):
            fig.add_trace(
                go.Scatter(
                    x=q,
                    y=curve,
                    mode='lines',
                    line={'color': POSTERIOR_DRAW_LINE_COLOR, 'width': 1},
                    name='Posterior draws',
                    legendgroup='draws',
                    showlegend=draw_number == 0,
                    hoverinfo='skip',
                )
            )

    best_curve = np.asarray(model_eval(dict(posterior.best)))
    fig.add_trace(
        go.Scatter(
            x=q,
            y=best_curve,
            mode='lines',
            name=POSTERIOR_POINT_ESTIMATE_TRACE_NAME,
            line={'color': POSTERIOR_POINT_ESTIMATE_LINE_COLOR, 'width': 2},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=q,
            y=y,
            error_y=_error_bars(dy),
            mode='markers',
            name=MEASURED_TRACE_NAME,
            opacity=0.6,
            marker={'size': 6, 'color': 'rgb(50, 50, 50)'},
        )
    )

    fig.update_layout(
        title='Posterior predictive check',
        xaxis_title='Q (Å⁻¹)',
        yaxis_title='I(Q)',
        xaxis_type='log' if log_scale else 'linear',
        yaxis_type='log' if log_scale else 'linear',
        template='plotly_white',
        height=600,
        width=900,
    )

    if _resolve_show(show):
        fig.show()
    return fig


def plot_param_correlations(
    posterior: PosteriorSummary,
    threshold: float = 0.0,
    show: bool | None = None,
) -> go.Figure:
    """Heatmap of the posterior sample correlation matrix (lower triangle).

    Args:
        threshold: Hide cells with |correlation| below this value.
    """
    if posterior.n_params < MIN_POSTERIOR_PARAMETER_COUNT:
        raise ValueError(
            f'A correlation matrix needs at least {MIN_POSTERIOR_PARAMETER_COUNT} '
            f'varying parameters (got {posterior.n_params}).'
        )

    corr = np.corrcoef(posterior.samples, rowvar=False)
    display = corr.copy()
    # Mask the diagonal (self-correlation = 1) and the upper triangle so the
    # matrix reads as a single lower-triangular block.
    display[np.triu_indices_from(display)] = np.nan
    if threshold > 0:
        display[np.abs(display) < threshold] = np.nan

    text = np.where(np.isnan(display), '', np.round(display, 2).astype(str))

    fig = go.Figure(
        go.Heatmap(
            z=display,
            x=posterior.labels,
            y=posterior.labels,
            zmin=-1,
            zmax=1,
            colorscale='RdBu',
            reversescale=True,
            text=text,
            texttemplate='%{text}',
            colorbar={'title': 'ρ'},
            hoverongaps=False,
        )
    )

    fig.update_layout(
        title='Posterior parameter correlations',
        template='plotly_white',
        height=max(120 * posterior.n_params, 450),
        width=max(120 * posterior.n_params, 500),
        yaxis={'autorange': 'reversed'},
    )

    if _resolve_show(show):
        fig.show()
    return fig


def plot_trace(
    posterior: PosteriorSummary,
    params: list[str] | None = None,
    show: bool | None = None,
) -> go.Figure:
    """Trace plot of the MCMC chains for each parameter.

    Shows one panel per parameter with every chain drawn against the
    generation number. When per-chain data is unavailable from the sampler,
    falls back to a single trace over the combined (flattened) chain.
    """
    selected = _resolve_posterior_params(posterior, params)
    n = len(selected)

    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.4 / max(n, 4),
    )

    has_chains = posterior.chains is not None
    for row_pos, (name, param_index) in enumerate(selected, start=1):
        if has_chains:
            n_generations, n_chains, _ = posterior.chains.shape
            generations = np.arange(n_generations)
            for chain_number in range(n_chains):
                fig.add_trace(
                    go.Scatter(
                        x=generations,
                        y=posterior.chains[:, chain_number, param_index],
                        mode='lines',
                        line={'width': 1},
                        opacity=0.6,
                        showlegend=False,
                    ),
                    row=row_pos,
                    col=1,
                )
        else:
            fig.add_trace(
                go.Scatter(
                    y=posterior.samples[:, param_index],
                    mode='lines',
                    line={'width': 1},
                    showlegend=False,
                ),
                row=row_pos,
                col=1,
            )
        fig.update_yaxes(title_text=name, row=row_pos, col=1)

    fig.update_xaxes(
        title_text='Generation' if has_chains else 'Sample',
        row=n,
        col=1,
    )
    fig.update_layout(
        title='MCMC trace' + ('' if has_chains else ' (combined chain)'),
        template='plotly_white',
        height=max(TRACE_PLOT_ROW_HEIGHT_PIXELS * n, 400),
        width=900,
    )

    if _resolve_show(show):
        fig.show()
    return fig


# ---------------------------------------------------------------------------
# P(r) inversion plots
#
# These functions are duck-typed on the PrResult / DmaxScan attributes and
# must not import from the inversion package (which imports this module for the
# result-object plot wrappers).
# ---------------------------------------------------------------------------

PR_LINE_COLOR = 'rgb(99, 110, 250)'
PR_BAND_FILL_COLOR = POSTERIOR_BAND_FILL_COLOR
PR_FIT_CURVE_POINTS = 300
DMAX_SCAN_QUANTITIES = {
    'rg': ('rg', 'Rg (Å)'),
    'i0': ('i0', 'I(0)'),
    'data_chisq': ('data_chisq', 'Data χ²'),
    'oscillations': ('oscillations', 'Oscillations'),
    'positive_fraction': ('positive_fraction', 'Positive fraction'),
    'sigma_positive_fraction': ('sigma_positive_fraction', '1σ positive fraction'),
    'background': ('background', 'Background'),
}


def plot_pr_distribution(result, show: bool | None = None) -> go.Figure:
    """Plot P(r) with its 1-sigma uncertainty band.

    Args:
        result: A PrResult (duck-typed: needs r, pr, pr_err, rg,
            positive_fraction).
        show: Same display convention as plot_results().
    """
    r = np.asarray(result.r)
    pr = np.asarray(result.pr)
    pr_err = np.asarray(result.pr_err)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([r, r[::-1]]),
            y=np.concatenate([pr + pr_err, (pr - pr_err)[::-1]]),
            fill='toself',
            fillcolor=PR_BAND_FILL_COLOR,
            line={'width': 0},
            name='±1σ',
            hoverinfo='skip',
        )
    )
    fig.add_trace(
        go.Scatter(
            x=r,
            y=pr,
            mode='lines',
            name='P(r)',
            line={'color': PR_LINE_COLOR, 'width': 2},
        )
    )
    fig.add_hline(y=0, line_dash='dash', line_color='gray')
    fig.update_layout(
        title=(
            f'P(r) — Rg = {result.rg:.4g} Å, positive fraction = {result.positive_fraction:.3f}'
        ),
        xaxis_title='r (Å)',
        yaxis_title='P(r)',
        template='plotly_white',
        height=500,
        width=800,
    )

    if _resolve_show(show):
        fig.show()
    return fig


def plot_pr_fit(data, result, show: bool | None = None, log_scale: bool = True) -> go.Figure:
    """Plot measured I(q) against the P(r) fit, with residuals below.

    The theory curve is evaluated on a dense q grid via result.evaluate_iq;
    the stored accepted-point vectors (q_fit/iq_fit/sigma_fit) provide
    point-aligned residuals.

    Args:
        data: The dataset the inversion was run on.
        result: A PrResult (duck-typed: needs accepted, q_fit, iq_fit,
            sigma_fit, evaluate_iq).
        show: Same display convention as plot_results().
        log_scale: Use log axes. Pass False when intensities include zero or
            negative values (e.g. background-subtracted data) — a log axis
            silently omits such points.
    """
    accepted = np.asarray(result.accepted, dtype=bool)
    if accepted.size != np.asarray(data.x).size:
        raise ValueError(
            f'Result accepted-mask length ({accepted.size}) does not match '
            f'the data length ({np.asarray(data.x).size}).'
        )
    q = _subset(data.x, accepted)
    y = _subset(data.y, accepted)
    dy = _subset(data.dy, accepted)
    excluded = ~accepted

    q_dense = np.geomspace(q.min(), q.max(), PR_FIT_CURVE_POINTS)
    i_dense = result.evaluate_iq(q_dense)
    residuals = (y - np.asarray(result.iq_fit)) / np.asarray(result.sigma_fit)

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.75, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.05,
    )
    fig.add_trace(
        go.Scatter(
            x=q,
            y=y,
            error_y=_error_bars(dy),
            mode='markers',
            name='Experimental Data',
            opacity=0.6,
            marker={'size': 6},
        ),
        row=1,
        col=1,
    )
    if excluded.any():
        fig.add_trace(
            go.Scatter(
                x=_subset(data.x, excluded),
                y=_subset(data.y, excluded),
                error_y=_error_bars(_subset(data.dy, excluded)),
                mode='markers',
                name='Excluded Data',
                opacity=0.4,
                marker={'size': 6, 'color': 'lightgray'},
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=q_dense,
            y=i_dense,
            mode='lines',
            name='P(r) Fit',
            line={'color': 'red', 'width': 2},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=q,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker={'size': 6},
            opacity=0.6,
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_dash='dash', line_color='gray', row=2, col=1)
    fig.update_yaxes(title_text='I(Q)', type='log' if log_scale else 'linear', row=1, col=1)
    fig.update_yaxes(title_text='Residuals (σ)', row=2, col=1)
    fig.update_xaxes(type='log' if log_scale else 'linear', row=1, col=1)
    fig.update_xaxes(title_text='Q (Å⁻¹)', type='log' if log_scale else 'linear', row=2, col=1)
    fig.update_layout(
        title='P(r) inversion fit',
        template='plotly_white',
        height=600,
        width=800,
    )

    if _resolve_show(show):
        fig.show()
    return fig


def plot_dmax_scan(scan, quantity: str = 'rg', show: bool | None = None) -> go.Figure:
    """Plot a D_max scan quantity (or 'all' as a small-multiples grid).

    Args:
        scan: A DmaxScan (duck-typed: needs d_max_values and the quantity
            arrays named in DMAX_SCAN_QUANTITIES).
        quantity: One of the DMAX_SCAN_QUANTITIES keys, or 'all'.
        show: Same display convention as plot_results().
    """
    if quantity == 'all':
        names = list(DMAX_SCAN_QUANTITIES)
        n_cols = 2
        n_rows = -(-len(names) // n_cols)
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[DMAX_SCAN_QUANTITIES[name][1] for name in names],
        )
        for position, name in enumerate(names):
            attr, _ = DMAX_SCAN_QUANTITIES[name]
            fig.add_trace(
                go.Scatter(
                    x=scan.d_max_values,
                    y=getattr(scan, attr),
                    mode='lines+markers',
                    showlegend=False,
                ),
                row=position // n_cols + 1,
                col=position % n_cols + 1,
            )
        for row in range(1, n_rows + 1):
            fig.update_xaxes(title_text='D_max (Å)', row=row, col=1)
            fig.update_xaxes(title_text='D_max (Å)', row=row, col=2)
        fig.update_layout(
            title='D_max exploration',
            template='plotly_white',
            height=280 * n_rows,
            width=900,
        )
    else:
        if quantity not in DMAX_SCAN_QUANTITIES:
            available = ', '.join([*DMAX_SCAN_QUANTITIES, 'all'])
            raise ValueError(f"Unknown quantity '{quantity}'. Available: {available}")
        attr, label = DMAX_SCAN_QUANTITIES[quantity]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=scan.d_max_values,
                y=getattr(scan, attr),
                mode='lines+markers',
                name=label,
            )
        )
        fig.update_layout(
            title=f'D_max exploration — {label}',
            xaxis_title='D_max (Å)',
            yaxis_title=label,
            template='plotly_white',
            height=500,
            width=800,
        )

    if _resolve_show(show):
        fig.show()
    return fig

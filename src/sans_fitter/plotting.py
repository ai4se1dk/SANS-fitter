import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data_loader import _has_real_data
from .results import FitResultContract


def plot_fit(
    data,
    fit_result: FitResultContract | None,
    model_name: str | None,
    show_residuals: bool = True,
    log_scale: bool = True,
) -> go.Figure:
    """Plot experimental data and, when available, the fitted model curve."""
    if data is None:
        raise ValueError('No data to plot. Use load_data() first.')

    error_x = (
        {'type': 'data', 'array': data.dx, 'visible': True} if _has_real_data(data.dx) else None
    )

    if fit_result is None:
        print('No fit results available. Plotting data only.')
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data.x,
                y=data.y,
                error_y={'type': 'data', 'array': data.dy, 'visible': True},
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
        fig.show()
        return fig

    q = data.x
    i_fit = fit_result.require_fitted_curve()
    residuals = (data.y - i_fit) / data.dy

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
        x=data.x,
        y=data.y,
        error_y={'type': 'data', 'array': data.dy, 'visible': True},
        error_x=error_x,
        mode='markers',
        name='Experimental Data',
        opacity=0.6,
        marker={'size': 6},
    )

    fit_trace = go.Scatter(
        x=q,
        y=i_fit,
        mode='lines',
        name='Fitted Model',
        line={'color': 'red', 'width': 2},
    )

    if show_residuals:
        fig.add_trace(data_trace, row=1, col=1)
        fig.add_trace(fit_trace, row=1, col=1)
        fig.add_trace(
            go.Scatter(
                x=data.x,
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
        fig.add_trace(fit_trace)
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

    fig.show()
    return fig

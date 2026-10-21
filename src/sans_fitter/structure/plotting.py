"""Plotly views of a structure (SasView's *Draw Points*) and of a computed profile."""

import os
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative

from ..plotting import error_bars, resolve_show

#: SasView's Draw Points colours; other elements cycle through a qualitative palette.
ELEMENT_COLOURS = {
    'H': 'blue',
    'D': 'purple',
    'N': 'orange',
    'O': 'red',
    'C': 'green',
    'P': 'cyan',
}


def _thin(structure, max_points: int) -> np.ndarray:
    """Indices of at most *max_points* points to draw, chosen deterministically.

    Lattice points keep every s-th plane on each axis, so a lattice still looks
    like one. Other points keep the first point in each occupied voxel, with
    the voxel grown until few enough remain.
    """
    n = structure.n_points
    if n <= max_points:
        return np.arange(n)
    if structure.lattice is not None:
        stride = int(np.ceil((n / max_points) ** (1 / 3)))
        while True:
            keep = np.flatnonzero(np.all(structure.lattice.indices % stride == 0, axis=1))
            if keep.size <= max_points:
                return keep
            stride += 1
    low, high = structure.bounding_box
    voxel = float(np.prod(np.maximum(high - low, 1e-9)) / max_points) ** (1 / 3)
    while True:
        cells = np.floor((structure.xyz - low) / voxel).astype(np.int64)
        _, keep = np.unique(cells, axis=0, return_index=True)
        if keep.size <= max_points:
            return np.sort(keep)
        voxel *= 1.25


def plot_points(
    structure,
    *,
    color: str = 'auto',
    sld_solvent: float = 0.0,
    max_points: int = 50_000,
    show: bool | None = None,
) -> go.Figure:
    """Draw a structure's points in 3D, with equal length scales on every axis.

    Args:
        structure: A :class:`~sans_fitter.structure.Structure`.
        color: ``'element'`` (atoms), ``'sld'``, or ``'auto'``: element for
            atoms, SLD otherwise.
        sld_solvent: Solvent SLD, 10⁻⁶ Å⁻². In SLD colouring, points that
            match it do not scatter and are drawn translucent yellow.
        max_points: Draw at most this many points; calculations always use
            all of them. 50 000 keeps a browser's 3D view responsive.
        show: Same convention as ``plot_results``.
    """
    if color == 'auto':
        color = 'element' if structure.symbols is not None else 'sld'
    if color not in ('element', 'sld'):
        raise ValueError(f"color must be 'auto', 'element' or 'sld', got {color!r}.")
    if color == 'element' and structure.symbols is None:
        raise ValueError("color='element' needs element symbols (a PDB structure).")

    keep = _thin(structure, max_points)
    xyz = structure.xyz[keep]
    size = 3 if structure.kind == 'atoms' else 2
    traces: list[Any] = []

    def scatter(mask, name, **marker):
        return go.Scatter3d(
            x=xyz[mask, 0],
            y=xyz[mask, 1],
            z=xyz[mask, 2],
            mode='markers',
            name=name,
            marker={'size': size, **marker},
        )

    if color == 'element':
        symbols = structure.symbols[keep]
        extra = iter(qualitative.Dark24)
        for symbol in sorted(set(symbols), key=lambda s: (s not in ELEMENT_COLOURS, s)):
            colour = ELEMENT_COLOURS.get(symbol) or next(extra, 'black')
            traces.append(scatter(symbols == symbol, symbol, color=colour, opacity=0.7))
    else:
        sld = structure.sld[keep] * 1e6
        matched = structure.sld[keep] - sld_solvent * 1e-6 == 0  # as the calculator tests it
        traces.append(
            scatter(
                ~matched,
                'SLD',
                color=sld[~matched],
                colorscale='Viridis',
                colorbar={'title': 'SLD (1e-6/Å²)'},
                opacity=0.7,
            )
        )
        if matched.any():
            traces.append(scatter(matched, 'zero contrast', color='yellow', opacity=0.2))

    thinned = keep.size < structure.n_points
    if structure.bonds is not None and structure.bonds.size and not thinned:
        ends = structure.xyz[structure.bonds]  # (n_bonds, 2, 3)
        path = np.concatenate([ends, np.full((len(ends), 1, 3), np.nan)], axis=1).reshape(-1, 3)
        traces.append(
            go.Scatter3d(
                x=path[:, 0],
                y=path[:, 1],
                z=path[:, 2],
                mode='lines',
                name='bonds',
                line={'color': 'grey', 'width': 2},
            )
        )

    what = 'cell centroids' if structure.kind == 'cells' else structure.kind
    title = f'{os.path.basename(structure.source)}: {structure.n_points} {what}'
    if thinned:
        title += f' (showing {keep.size} of {structure.n_points} points)'
    fig = go.Figure(traces)
    fig.update_layout(
        title=title,
        scene={
            'aspectmode': 'data',
            'xaxis_title': 'x (Å)',
            'yaxis_title': 'y (Å)',
            'zaxis_title': 'z (Å)',
        },
        height=600,
    )
    if resolve_show(show):
        fig.show()
    return fig


def plot_profile(
    profile, *, data=None, log_scale: bool = True, show: bool | None = None
) -> go.Figure:
    """Plot a computed I(q), optionally over measured data.

    On log axes, non-positive intensities cannot be drawn: they are left out
    and counted in an annotation; the profile itself is not changed. A
    measured overlay is usually resolution-smeared while the profile is not.
    """
    q, intensity = profile.q, profile.intensity
    name = f'GSC {profile.structure_summary["basename"]}'
    if log_scale and not np.any(intensity > 0):
        log_scale = False  # nothing to draw on log axes, e.g. a zero background
    drawn = intensity > 0 if log_scale else np.ones(q.size, dtype=bool)

    fig = go.Figure()
    if data is not None:
        fig.add_trace(
            go.Scatter(
                x=data.x,
                y=data.y,
                error_y=error_bars(data.dy),
                mode='markers',
                name=getattr(data, 'title', '') or 'data',
                marker={'size': 5},
            )
        )
    fig.add_trace(go.Scatter(x=q[drawn], y=intensity[drawn], mode='lines', name=name))
    skipped = int(np.sum(~drawn))
    if skipped:
        fig.add_annotation(
            text=f'{skipped} non-positive points not shown on log axes',
            xref='paper',
            yref='paper',
            x=0.98,
            y=0.98,
            showarrow=False,
        )
    axis_type = 'log' if log_scale else 'linear'
    fig.update_layout(
        title=name,
        xaxis={'title': 'q (Å⁻¹)', 'type': axis_type},
        yaxis={'title': 'I(q) (cm⁻¹)', 'type': axis_type},
    )
    if resolve_show(show):
        fig.show()
    return fig

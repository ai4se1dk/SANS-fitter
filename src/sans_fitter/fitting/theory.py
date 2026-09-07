"""Model evaluation without fitting: the theory preview's building blocks."""

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np
from sasmodels.data import empty_data1D
from sasmodels.direct_model import DirectModel

from ..data.loader import has_real_data, validate_q_grid
from ..results import ParameterStateSnapshot, resolve_fit_index
from .base import extract_fit_index, link_radius_effective_dict, pd_is_active


def build_model_parameters(
    fit_state: ParameterStateSnapshot,
    overrides: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Canonical sasmodels keyword arguments for a parameter snapshot.

    Order: values, active polydispersity blocks, *overrides*, equality links
    (follower = target), the ``radius_effective`` link. An override therefore
    reaches a follower through its target. A ``<base>_pd`` override brings
    the block's companion settings along, because sasmodels ignores a width
    that arrives without ``_pd_n``.
    """
    pars: dict[str, Any] = {name: info['value'] for name, info in fit_state.params.items()}

    if fit_state.pd_enabled:
        for base_param in fit_state.polydisperse_param_names:
            pd_config = fit_state.polydisperse_params[base_param]
            if pd_is_active(pd_config):
                pars[f'{base_param}_pd'] = pd_config['pd']
                pars[f'{base_param}_pd_n'] = pd_config['pd_n']
                pars[f'{base_param}_pd_nsigma'] = pd_config['pd_nsigma']
                pars[f'{base_param}_pd_type'] = pd_config['pd_type']

    if overrides:
        pars.update(overrides)
        for name in overrides:
            base = name.removesuffix('_pd')
            if name.endswith('_pd') and f'{base}_pd_n' not in pars:
                pd_config = fit_state.polydisperse_params[base]
                pars[f'{base}_pd_n'] = pd_config['pd_n']
                pars[f'{base}_pd_nsigma'] = pd_config['pd_nsigma']
                pars[f'{base}_pd_type'] = pd_config['pd_type']

    for follower, target in fit_state.linked_params.items():
        if target in pars:
            pars[follower] = pars[target]

    link_radius_effective_dict(pars, fit_state.radius_effective_mode)
    return pars


def theory_data(q: Any, dq: float | None = None) -> Any:
    """An empty dataset on the Q grid *q*, smeared by ΔQ/Q = *dq* if given."""
    q_values = validate_q_grid(q)
    if dq is not None and (not np.isfinite(dq) or dq < 0):
        raise ValueError(f'dq must be non-negative and finite, got {dq}.')
    return empty_data1D(q_values, resolution=0.0 if dq is None else float(dq))


def evaluate_theory(data: Any, kernel: Any, pars: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate *kernel* over the fitted points of *data*.

    Returns the float64 curve and the boolean fit index. The dataset's own
    resolution columns are applied, exactly as during a fit.
    """
    if data.y is not None and data.dy is None:
        # DirectModel indexes dy whenever y is present. Evaluate on a shallow
        # copy so the caller's dataset keeps its missing dI.
        data = copy.copy(data)
        data.dy = np.zeros_like(np.asarray(data.y, dtype=float))

    calculator = DirectModel(data, kernel)
    curve = np.asarray(calculator(**pars), dtype=float)
    return curve, resolve_fit_index(extract_fit_index(calculator), len(data.x))


def scatter_to_full_length(curve: np.ndarray, fit_index: np.ndarray) -> np.ndarray:
    """Return *curve* on the full data grid, NaN where a point was not fitted."""
    full = np.full(len(fit_index), np.nan)
    full[fit_index] = curve
    return full


def preview_chisq(data: Any, curve: np.ndarray, fit_index: np.ndarray, n_free: int) -> float:
    """χ²/dof in bumps' convention, so it equals the "Initial χ²" a bumps fit prints.

    NaN when the data has no usable uncertainties: a preview does not invent
    a weight. dof is clamped to 1, diverging from bumps only when free
    parameters outnumber fitted points.
    """
    dy = None if data.dy is None else np.asarray(data.dy, dtype=float)[fit_index]
    if not has_real_data(dy) or np.any(dy == 0):
        return float('nan')

    y = np.asarray(data.y, dtype=float)[fit_index]
    return float(np.sum(((y - curve) / dy) ** 2) / max(len(y) - n_free, 1))

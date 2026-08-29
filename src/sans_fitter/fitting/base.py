from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..results import FitResultContract, ParameterStateSnapshot


def pd_is_active(pd_config: dict[str, Any]) -> bool:
    """Return whether a PD configuration should be included in model evaluation."""
    return pd_config['pd'] > 0 or pd_config.get('vary', False)


def extract_fit_index(source: Any) -> np.ndarray | None:
    """Return the boolean fit index from a sasmodels calculator/experiment.

    sasmodels stores the points it actually evaluates (inside [qmin, qmax],
    unmasked, finite) as ``source.index``. Returns None when unavailable so
    consumers fall back to treating the curve as full-length.
    """
    index = getattr(source, 'index', None)
    if index is None or isinstance(index, slice):
        return None
    return np.asarray(index, dtype=bool)


def link_radius_effective_model(model: Any, radius_effective_mode: str) -> None:
    """Link radius_effective to radius on mutable model objects when requested."""
    if (
        radius_effective_mode == 'link_radius'
        and hasattr(model, 'radius_effective')
        and hasattr(model, 'radius')
    ):
        model.radius_effective = model.radius


def link_radius_effective_dict(parameters: dict[str, Any], radius_effective_mode: str) -> None:
    """Link radius_effective to radius in parameter dictionaries when requested."""
    if (
        radius_effective_mode == 'link_radius'
        and 'radius' in parameters
        and 'radius_effective' in parameters
    ):
        parameters['radius_effective'] = parameters['radius']


def build_result_parameters(
    fit_state: ParameterStateSnapshot,
    varied: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Assemble the engine-independent ``parameters`` block of a fit result.

    Every engine reports the same set — one entry per model parameter, plus any
    polydispersity width the engine varied — so code written against one engine
    keeps working against the other. Every entry carries the same four fields:
    ``value``, ``stderr``, ``formatted``, a ``fixed`` flag separating the
    optimizer's dimensions from the rest, and ``linked_to`` naming the parameter
    it follows (``None`` when it follows none).

    *varied* holds the engine's own entries for the parameters it optimized,
    already formatted in that engine's uncertainty convention. Everything else
    in ``fit_state.params`` is appended here. A parameter that follows another
    one (``link_params`` or ``radius_effective_mode='link_radius'``) reports its
    target's fitted value rather than the pre-fit value the snapshot carries.

    Names here are canonical; ``linked_to`` is translated to the user-facing
    alias alongside the keys, in ``SANSFitter._finalize_fit``.
    """
    parameters = {
        name: {**info, 'fixed': False, 'linked_to': None} for name, info in varied.items()
    }

    followers = dict(fit_state.linked_params)
    if fit_state.radius_effective_mode == 'link_radius' and 'radius_effective' in fit_state.params:
        followers.setdefault('radius_effective', 'radius')

    for name, info in fit_state.params.items():
        if name in parameters:
            continue
        target = followers.get(name)
        if target is not None and target in parameters:
            # The target was fitted, so the snapshot's follower value is stale.
            # A fixed target needs no correction: its value never moved.
            value = parameters[target]['value']
        else:
            value = info['value']
        parameters[name] = {
            'value': value,
            'stderr': 0.0,
            'formatted': f'{value:.6g} ({"linked" if target else "fixed"})',
            'fixed': True,
            'linked_to': target,
        }
    return parameters


@dataclass(slots=True)
class EngineFitOutput:
    """Internal engine output used by SANSFitter to sync state after fitting."""

    contract: FitResultContract
    fitted_values: dict[str, float]
    runtime_model: Any


class FittingEngine(Protocol):
    """Protocol for extracted fitting engines."""

    def __call__(
        self,
        data: Any,
        kernel: Any,
        fit_state: ParameterStateSnapshot,
        method: str,
        **kwargs: Any,
    ) -> EngineFitOutput: ...

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

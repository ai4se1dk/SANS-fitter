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


def apply_parameter_links(parameters: dict[str, Any], linked_params: dict[str, str]) -> None:
    """Force every link follower to its target's value in a parameter dict.

    The dict-level counterpart of the bumps engine's parameter-object aliasing,
    used by evaluation paths that speak plain sasmodels kwargs (the scipy
    residual, the DREAM posterior evaluator). It must run on *every* evaluation:
    followers carry a stale value once the optimizer moves their target.

    The link graph has depth 1 (no target is itself a follower), so the order of
    assignment does not matter. ``ParameterManager`` guarantees that invariant.
    """
    for follower, target in linked_params.items():
        if follower in parameters and target in parameters:
            parameters[follower] = parameters[target]


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

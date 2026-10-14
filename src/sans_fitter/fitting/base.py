from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..results import FitResultContract, ParameterStateSnapshot

#: Absolute floor for the on-bound proximity test. Also what makes a parameter fitted to
#: exactly ``0`` against ``min=0`` register as on-bound.
BOUND_ATOL = 1e-12
#: Relative tolerance for the on-bound proximity test, against the larger of the bound and
#: the value. Deliberately *not* a fraction of the interval span: on ``scale in [1e-5, 1]``
#: a span fraction would flag every fitted scale below ~1e-3, which is a normal SANS value.
BOUND_RTOL = 1e-4


def pd_is_active(pd_config: dict[str, Any]) -> bool:
    """Return whether a PD configuration should be included in model evaluation."""
    return pd_config['pd'] > 0 or pd_config.get('vary', False)


def normalize_message(text: Any) -> str:
    """Collapse an optimizer message to a single line of single-spaced text.

    scipy's ``leastsq`` returns a wrapped multi-line string. A newline in the
    message would break the ``#``-comment block of the exported CSV and the row
    structure of a Markdown table, so it is removed where the message enters the
    package rather than at each place that renders it.
    """
    return ' '.join(str(text or '').split())


def reduced_chisq(chisq: float, dof: int) -> float:
    """Return ``chisq / dof``, or NaN when there are no degrees of freedom.

    The single place the division happens, so every engine and the theory preview
    agree — including in the degenerate case where the free parameters outnumber
    the fitted points.

    Note that bumps' own ``problem.chisq()`` does *not* degrade this way: its
    ``nllf_scale`` returns a scale of 1.0 when ``dof <= 0``, making the reported
    value half the raw sum of squares. Ours is NaN, which is why the equality test
    between the two is scoped to ``dof > 0``.
    """
    if dof <= 0 or not np.isfinite(chisq):
        return float('nan')
    return float(chisq) / dof


def correlation_matrix(cov: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix.

    ``R = D⁻¹ C D⁻¹`` written out, where D is the diagonal of standard errors.
    A zero-variance dimension yields NaN for its row and column: the correlation
    is undefined there, not infinite.

    Deliberately *not* ``bumps.lsqerror.corr``, whose implementation collapses to
    a scalar (its ``Dinv`` is 1-D, so the double ``np.dot`` contracts both axes)
    despite a correct docstring. Keeping it here also keeps the scipy engine free
    of a bumps import.
    """
    cov = np.asarray(cov, dtype=float)
    sigma = np.sqrt(np.diag(cov))
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = cov / np.outer(sigma, sigma)
    corr = np.asarray(corr, dtype=float)
    corr[~np.isfinite(corr)] = np.nan
    return corr


def at_bound(value: float, bound: float) -> bool:
    """Return whether *value* sits on *bound* within the on-bound tolerance.

    The tolerance is relative to the magnitude of the bound and the value, with an
    absolute floor, so a fitted value of exactly ``0`` registers against
    ``min=0`` while an ordinary small value well inside its range does not.
    An infinite bound is never hit.
    """
    if not np.isfinite(bound) or not np.isfinite(value):
        return False
    scale = max(abs(float(bound)), abs(float(value)), BOUND_ATOL)
    return abs(float(value) - float(bound)) <= max(BOUND_ATOL, BOUND_RTOL * scale)


def validate_covariance(cov: Any, labels: Sequence[str]) -> np.ndarray:
    """Check an engine's covariance matrix at the integration boundary.

    Returns the matrix as a float array. Raises ValueError when it is not
    two-dimensional, not square, not aligned with *labels*, or carries an entry
    that is neither finite nor NaN. A library guarantee is not a substitute for a
    contract check: a mislabelled matrix would be read as parameter correlations.
    """
    matrix = np.asarray(cov, dtype=float)
    expected = (len(labels), len(labels))
    if matrix.ndim != 2 or matrix.shape != expected:
        raise ValueError(
            f'Covariance matrix shape {matrix.shape} does not match the '
            f'{len(labels)} fitted parameter(s) {list(labels)} (expected {expected}).'
        )
    if np.isinf(matrix).any():
        raise ValueError('Covariance matrix contains infinite entries.')
    return matrix


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

    ``stderr`` is √diag(covariance) on every engine, deliberately **not** rescaled
    by √(χ²/dof): that is bumps' convention (``lsqerror.stderr``: "without any
    correction") and scipy's ``leastsq`` convention, so the two agree. See the
    "Judging a fit" section of ``docs/usage.md`` for when a caller should apply the
    scaling themselves.

    *varied* holds the engine's own entries for the parameters it optimized,
    already formatted in that engine's uncertainty convention. Everything else
    in ``fit_state.params`` is appended here. A parameter that follows another
    one (``link_params`` or ``radius_effective_mode='link_radius'``) reports its
    target's fitted value rather than the pre-fit value the snapshot carries.

    **A follower of a fitted target also reports that target's uncertainty.** An
    equality link makes the two parameters one quantity under two names, so the
    follower's error is the target's error exactly — not zero. ``fixed=True``
    stays as the flag separating optimizer coordinates from everything else, and
    no longer implies a zero uncertainty; read ``linked_to`` to tell a follower
    from a genuinely fixed parameter, whose error really is zero. A follower of a
    *fixed* target keeps the zero, because its target never moved.

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
            # The target was fitted, so both the snapshot's follower value and a
            # zero uncertainty would be wrong: the follower *is* the target.
            # Copying ``formatted`` keeps each engine's own convention, and the
            # two strings describe the same number.
            source = parameters[target]
            parameters[name] = {
                'value': source['value'],
                'stderr': source['stderr'],
                'formatted': source['formatted'],
                'fixed': True,
                'linked_to': target,
            }
            continue
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

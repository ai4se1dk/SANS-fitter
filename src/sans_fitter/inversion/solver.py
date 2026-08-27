"""Data preparation, matrix assembly, the SVD solve and the core invert()."""

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..data.loader import get_fit_index, has_real_data
from .basis import (
    _ortho_derived,
    _ortho_second_derivative,
    _ortho_transformed,
    _pr_curve_and_band,
    _sasview_reg_term,
)
from .result import InsufficientDataError, PrResult

DEFAULT_N_TERMS = 10
DEFAULT_R_POINTS = 101
REGULARIZERS = ('corrected', 'sasview')
SASVIEW_N_REG = 20
FABRICATED_SIGMA_RELATIVE = 0.05
FABRICATED_SIGMA_FLOOR_FRACTION = 0.01

# ---------------------------------------------------------------------------
# Data preparation (non-mutating)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _PreparedData:
    """Validated arrays plus the full-length accepted mask, built once per call."""

    q: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray
    accepted: np.ndarray
    uncertainties_fabricated: bool
    n_dropped_points: int


def _prepare_data(data: Any) -> _PreparedData:
    """Validate a dataset and build the accepted-point arrays without mutating it.

    Accepts a general ``Data1D`` — normalized (with ``qmin``/``qmax``/``mask``)
    or raw. For raw data no q-cut is invented: all finite, unmasked points are
    used. Points with non-finite q, I or sigma and ``q == 0`` points are
    dropped with a warning.
    """
    if getattr(data, 'qx_data', None) is not None:
        raise TypeError('P(r) inversion supports 1D data only, got 2D data.')
    if getattr(data, 'isSesans', False):
        raise TypeError('P(r) inversion does not support SESANS data.')

    x_raw = getattr(data, 'x', None)
    y_raw = getattr(data, 'y', None)
    if x_raw is None or y_raw is None:
        raise ValueError('Dataset has no data: x and y must be populated.')
    x = np.asarray(x_raw, dtype=float)
    y = np.asarray(y_raw, dtype=float)
    if x.size == 0 or y.size == 0:
        raise ValueError('Dataset has no data: x and y must be populated.')
    if x.size != y.size:
        raise ValueError(f'Length mismatch: x has {x.size} points but y has {y.size}.')

    dy_raw = getattr(data, 'dy', None)
    has_dy = has_real_data(dy_raw)
    if has_dy and np.asarray(dy_raw).size != x.size:
        raise ValueError(
            f'Length mismatch: x has {x.size} points but dy has {np.asarray(dy_raw).size}.'
        )

    if np.any(x[np.isfinite(x)] < 0):
        raise ValueError('Dataset contains negative q values; q must be non-negative.')

    for attr in ('dxl', 'dxw'):
        if has_real_data(getattr(data, attr, None)):
            warnings.warn(
                'Dataset carries slit-smearing resolution columns (dxl/dxw); '
                'P(r) inversion does not support smearing and will treat the '
                'data as unsmeared. Results for USANS-type data will be wrong.',
                stacklevel=3,
            )
            break

    if getattr(data, 'qmin', None) is not None and getattr(data, 'qmax', None) is not None:
        base = get_fit_index(data)
    else:
        base = ~np.isnan(y)
        mask = getattr(data, 'mask', None)
        if mask is not None and np.asarray(mask).size == y.size:
            base &= ~np.asarray(mask, dtype=bool)

    if has_dy:
        sigma = np.asarray(dy_raw, dtype=float)
        fabricated = False
    else:
        finite_i = np.abs(y[np.isfinite(y)])
        median_scale = float(np.median(finite_i)) if finite_i.size else 0.0
        sigma = np.maximum(
            FABRICATED_SIGMA_RELATIVE * np.abs(y),
            FABRICATED_SIGMA_FLOOR_FRACTION * median_scale,
        )
        fabricated = True
        warnings.warn(
            'Dataset has no intensity uncertainties (dI); fabricated '
            f'sigma = max({FABRICATED_SIGMA_RELATIVE}*|I|, '
            f'{FABRICATED_SIGMA_FLOOR_FRACTION}*median|I|) will be used. '
            'Chi-squared-based diagnostics are not interpretable.',
            stacklevel=3,
        )

    accepted = base & np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0) & (x > 0)
    n_dropped = int(base.sum()) - int(accepted.sum())
    if n_dropped > 0:
        warnings.warn(
            f'{n_dropped} point(s) dropped from the inversion (non-finite q/I/sigma, '
            'non-positive sigma, or q == 0).',
            stacklevel=3,
        )
    if not accepted.any():
        raise InsufficientDataError('No usable data points remain after validation.')

    return _PreparedData(
        q=x[accepted],
        intensity=y[accepted],
        sigma=sigma[accepted],
        accepted=accepted,
        uncertainties_fabricated=fabricated,
        n_dropped_points=n_dropped,
    )


# ---------------------------------------------------------------------------
# Matrix assembly and SVD solve
# ---------------------------------------------------------------------------


def _regularization_points(n_terms: int, regularizer: str) -> int:
    """Penalty-grid size: resolved for the corrected operator, SasView's 20 otherwise."""
    if regularizer == 'sasview':
        return SASVIEW_N_REG
    return max(101, 4 * n_terms + 1)


def _build_blocks(
    q: np.ndarray,
    intensity: np.ndarray,
    sigma: np.ndarray,
    d_max: float,
    n_terms: int,
    fit_background: bool,
    regularizer: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the whitened data block and the unit-alpha regularization block.

    Returns ``(a_data, b_data, reg_unit)``. Columns: the fitted background
    (when requested) comes first, then basis terms n = 1..n_terms. The
    background column is unregularized (zero in ``reg_unit``), matching
    SasView. ``reg_unit`` rows carry genuine quadrature weights ``sqrt(w_m)``
    for the corrected operator, or SasView's ``D/N_r`` prefactor on its exact
    20-point left-Riemann grid for the compatibility mode.
    """
    n_bg = 1 if fit_background else 0
    n_func = n_terms + n_bg

    a_data = np.zeros((q.size, n_func))
    if fit_background:
        a_data[:, 0] = 1.0 / sigma
    for j in range(n_terms):
        a_data[:, n_bg + j] = _ortho_transformed(d_max, j + 1, q) / sigma
    b_data = intensity / sigma

    n_reg = _regularization_points(n_terms, regularizer)
    reg_unit = np.zeros((n_reg, n_func))
    if regularizer == 'sasview':
        r_grid = np.arange(n_reg) * d_max / n_reg
        prefactor = d_max / n_reg
        for j in range(n_terms):
            reg_unit[:, n_bg + j] = prefactor * _sasview_reg_term(d_max, j + 1, r_grid)
    else:
        r_grid = np.linspace(0.0, d_max, n_reg)
        dr = d_max / (n_reg - 1)
        w = np.full(n_reg, dr)
        w[0] = w[-1] = dr / 2.0
        sqrt_w = np.sqrt(w)
        for j in range(n_terms):
            reg_unit[:, n_bg + j] = sqrt_w * _ortho_second_derivative(d_max, j + 1, r_grid)
    return a_data, b_data, reg_unit


@dataclass(slots=True)
class _Solution:
    """Raw output of the SVD solve on the augmented system."""

    coefficients: np.ndarray  # full n_func vector (background first when fitted)
    covariance: np.ndarray  # full n_func x n_func
    data_chisq: float
    effective_dof: float
    regularization_penalty: float
    rank: int
    condition_number: float


def _solve(a_data: np.ndarray, b_data: np.ndarray, reg_unit: np.ndarray, alpha: float) -> _Solution:
    """Solve the augmented system with one SVD; all diagnostics derive from it.

    ``C = P_d P_d^T`` with ``P_d = V S^-1 U_data^T`` (data rows whitened, penalty
    rows noiseless) and ``effective_dof = tr(A_data P_d) = ||U_data||_F^2`` over
    the kept singular directions — one coherent estimator pair. Truncation:
    ``s_i <= rcond * s_max`` with ``rcond = eps * max(M, N)`` (NumPy's
    ``lstsq(rcond=None)`` default). ``data_chisq`` is computed explicitly from
    the data block, never from residual output.
    """
    n_data = a_data.shape[0]
    if alpha > 0:
        a = np.vstack([a_data, math.sqrt(alpha) * reg_unit])
    else:
        a = a_data
    b = np.concatenate([b_data, np.zeros(a.shape[0] - n_data)])

    u, s, vt = np.linalg.svd(a, full_matrices=False)
    rcond = np.finfo(float).eps * max(a.shape)
    cutoff = rcond * (s[0] if s.size else 0.0)
    keep = s > cutoff
    rank = int(keep.sum())
    n_func = a.shape[1]
    if rank < n_func:
        warnings.warn(
            f'Rank-deficient system: rank {rank} of {n_func} coefficients; '
            'the smallest singular directions were truncated.',
            stacklevel=4,
        )

    s_inv = np.zeros_like(s)
    s_inv[keep] = 1.0 / s[keep]
    u_data = u[:n_data, :]
    p_d = (vt.T * s_inv) @ u_data.T
    coefficients = (vt.T * s_inv) @ (u.T @ b)
    covariance = p_d @ p_d.T
    effective_dof = float(np.sum(u_data[:, keep] ** 2))
    condition = float(s[0] / s[keep][-1]) if rank > 0 else float('inf')

    residual = a_data @ coefficients - b_data
    data_chisq = float(residual @ residual)
    penalty = float(alpha * np.sum((reg_unit @ coefficients) ** 2))

    return _Solution(
        coefficients=coefficients,
        covariance=covariance,
        data_chisq=data_chisq,
        effective_dof=effective_dof,
        regularization_penalty=penalty,
        rank=rank,
        condition_number=condition,
    )


# ---------------------------------------------------------------------------
# Derived outputs
# ---------------------------------------------------------------------------


def _derived_outputs(
    coefficients: np.ndarray,
    d_max: float,
    r: np.ndarray,
    pr: np.ndarray,
    pr_err: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Compute (rg, i0, oscillations, positive_fraction, sigma_positive_fraction).

    Trapezoidal quadrature on the shared r grid. Degenerate cases: Rg is NaN
    (with a warning) when the P(r) integral is non-positive; the oscillation
    and positive-fraction metrics are 0 for P identically zero.
    """
    integral_p = float(np.trapezoid(pr, r))
    i0 = 4.0 * np.pi * integral_p

    if integral_p > 0:
        integral_r2p = float(np.trapezoid(r**2 * pr, r))
        if integral_r2p > 0:
            rg = math.sqrt(integral_r2p / (2.0 * integral_p))
        else:
            rg = float('nan')
            warnings.warn(
                'The r^2-weighted P(r) integral is non-positive; Rg is undefined (NaN).',
                stacklevel=4,
            )
    else:
        rg = float('nan')
        warnings.warn(
            'The P(r) integral is non-positive; Rg is undefined (NaN).',
            stacklevel=4,
        )

    pr_square = float(np.trapezoid(pr**2, r))
    if pr_square > 0:
        pr_prime = np.zeros_like(r)
        for j, c in enumerate(coefficients):
            pr_prime += c * _ortho_derived(d_max, j + 1, r)
        prime_square = float(np.trapezoid(pr_prime**2, r))
        oscillations = (d_max / np.pi) * math.sqrt(prime_square / pr_square)
    else:
        oscillations = 0.0

    abs_sum = float(np.sum(np.abs(pr)))
    if abs_sum > 0:
        positive_fraction = float(np.sum(pr[pr > 0]) / abs_sum)
        sigma_positive_fraction = float(np.sum(pr[pr > pr_err]) / abs_sum)
    else:
        positive_fraction = 0.0
        sigma_positive_fraction = 0.0

    return rg, i0, oscillations, positive_fraction, sigma_positive_fraction


# ---------------------------------------------------------------------------
# Core inversion
# ---------------------------------------------------------------------------


def _require_integer(name: str, value: Any) -> None:
    """Reject booleans and non-integers before they reach range()/linspace()."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer, got {value!r}.')


def _validate_invert_args(
    d_max: float,
    n_terms: int,
    alpha: float,
    r_points: int,
    regularizer: str,
    background: float = 0.0,
) -> None:
    """Reject invalid scalar arguments (non-finite, non-integer, out of range)
    before any numerical work begins."""
    _require_integer('n_terms', n_terms)
    _require_integer('r_points', r_points)
    if not (np.isfinite(d_max) and d_max > 0):
        raise ValueError(f'd_max must be positive and finite, got {d_max}.')
    if not (np.isfinite(alpha) and alpha >= 0):
        raise ValueError(f'alpha must be non-negative and finite, got {alpha}.')
    if not np.isfinite(background):
        raise ValueError(f'background must be finite, got {background}.')
    if n_terms < 1:
        raise ValueError(f'n_terms must be at least 1, got {n_terms}.')
    if r_points < 2:
        raise ValueError(f'r_points must be at least 2, got {r_points}.')
    if regularizer not in REGULARIZERS:
        raise ValueError(f"regularizer must be one of {REGULARIZERS}, got '{regularizer}'.")


def _invert_prepared(
    prep: _PreparedData,
    d_max: float,
    n_terms: int,
    alpha: float,
    fit_background: bool,
    background: float,
    r_points: int,
    regularizer: str,
) -> PrResult:
    """Run the inversion on already-prepared data (shared by all entry points)."""
    n_bg = 1 if fit_background else 0
    n_func = n_terms + n_bg
    if prep.q.size < n_func + 1:
        raise InsufficientDataError(
            f'Only {prep.q.size} usable points for {n_func} coefficients; '
            f'at least {n_func + 1} are required.'
        )

    q_min = float(prep.q.min())
    q_max = float(prep.q.max())
    if d_max > np.pi / q_min:
        warnings.warn(
            f'd_max = {d_max:g} exceeds pi/q_min = {np.pi / q_min:g}; distances '
            'beyond pi/q_min are only weakly constrained by the lowest measured '
            'q (low-q support heuristic, not a hard limit).',
            stacklevel=3,
        )
    # Same ceiling as _admissible_n_range, so the warning can never fire on an
    # N the estimator itself selected.
    shannon_channels = math.ceil(q_max * d_max / np.pi)
    if n_terms > shannon_channels:
        warnings.warn(
            f'n_terms = {n_terms} exceeds the number of Shannon channels '
            f'(ceil(q_max*d_max/pi) = {shannon_channels}); the data cannot '
            'support that many terms.',
            stacklevel=3,
        )

    intensity = prep.intensity if fit_background else prep.intensity - background

    a_data, b_data, reg_unit = _build_blocks(
        prep.q, intensity, prep.sigma, d_max, n_terms, fit_background, regularizer
    )
    solution = _solve(a_data, b_data, reg_unit, alpha)

    if fit_background:
        coefficients = solution.coefficients[1:]
        background_value = float(solution.coefficients[0])
        background_err = float(np.sqrt(max(solution.covariance[0, 0], 0.0)))
    else:
        coefficients = solution.coefficients
        background_value = background
        background_err = float('nan')

    r = np.linspace(0.0, d_max, r_points)
    coefficient_cov = solution.covariance[n_bg:, n_bg:]
    pr, pr_err = _pr_curve_and_band(coefficients, coefficient_cov, d_max, r)

    rg, i0, oscillations, pos_frac, sigma_pos_frac = _derived_outputs(
        coefficients, d_max, r, pr, pr_err
    )

    iq_fit = np.full_like(prep.q, background_value)
    for j, c in enumerate(coefficients):
        iq_fit += c * _ortho_transformed(d_max, j + 1, prep.q)

    return PrResult(
        d_max=d_max,
        n_terms=n_terms,
        alpha=alpha,
        regularizer=regularizer,
        coefficients=coefficients,
        covariance=solution.covariance,
        background=background_value,
        background_fitted=fit_background,
        background_err=background_err,
        data_chisq=solution.data_chisq,
        effective_dof=solution.effective_dof,
        regularization_penalty=solution.regularization_penalty,
        n_points_used=int(prep.q.size),
        accepted=prep.accepted.copy(),
        condition_number=solution.condition_number,
        rank=solution.rank,
        uncertainties_fabricated=prep.uncertainties_fabricated,
        n_dropped_points=prep.n_dropped_points,
        rg=rg,
        i0=i0,
        oscillations=oscillations,
        positive_fraction=pos_frac,
        sigma_positive_fraction=sigma_pos_frac,
        r=r,
        pr=pr,
        pr_err=pr_err,
        q_fit=prep.q.copy(),
        iq_fit=iq_fit,
        sigma_fit=prep.sigma.copy(),
    )


def invert(
    data: Any,
    d_max: float,
    n_terms: int = DEFAULT_N_TERMS,
    alpha: float = 0.0,
    fit_background: bool = True,
    background: float = 0.0,
    r_points: int = DEFAULT_R_POINTS,
    regularizer: str = 'corrected',
) -> PrResult:
    """Invert I(q) into the pair distance distribution P(r) on [0, d_max].

    P(r) is expanded in Moore's sine basis and the coefficients come from a
    regularized linear least-squares fit solved by SVD. The fit honours the
    dataset's ``qmin``/``qmax``/``mask`` (as set by ``SANSFitter.set_q_range``)
    when present; raw datasets are used in full. The input dataset is never
    modified.

    Args:
        data: A 1D dataset (``Data1D``) with ``x`` (q), ``y`` (I) and
            optionally ``dy``. When ``dy`` is absent, uncertainties are
            fabricated (with a warning and a flag on the result).
        d_max: Maximum particle dimension in Angstrom; P(d_max) = 0 by
            construction.
        n_terms: Number of sine-basis terms.
        alpha: Regularization constant; 0 means an unregularized fit (a
            warning is emitted — use :func:`estimate_alpha` or
            :func:`auto_invert` for a data-driven value).
        fit_background: Fit a flat background as an extra (unregularized)
            column. Use ``False`` for buffer-subtracted data — the usual
            protein workflow — together with ``background`` for any known
            residual level.
        background: Constant background subtracted from the data when
            ``fit_background`` is False. Ignored otherwise.
        r_points: Number of points of the r grid for P(r) evaluation and the
            derived integrals.
        regularizer: ``'corrected'`` (true second-derivative penalty with a
            resolved quadrature) or ``'sasview'`` (SasView's exact operator,
            for compatibility/comparison).

    Returns:
        A :class:`PrResult` with the solution, uncertainties, quality
        diagnostics and derived scalars (Rg, I(0), oscillations, positive
        fractions).

    Raises:
        TypeError: If the dataset is not 1D (2D or SESANS data).
        ValueError: For invalid arguments or unusable datasets.
        InsufficientDataError: When too few usable points remain.
    """
    _validate_invert_args(d_max, n_terms, alpha, r_points, regularizer, background)
    if alpha == 0.0:
        warnings.warn(
            'alpha = 0 gives an unregularized fit, which is usually noise-dominated; '
            'consider estimate_alpha() or auto_invert() for a data-driven value.',
            stacklevel=2,
        )
    prep = _prepare_data(data)
    return _invert_prepared(
        prep, d_max, n_terms, alpha, fit_background, background, r_points, regularizer
    )

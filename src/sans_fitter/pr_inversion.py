"""
P(r) inversion — indirect Fourier transform of I(q) (issue #57).

Model-free analysis recovering the real-space pair distance distribution
function P(r) from measured I(q), using Moore's sine-basis expansion
(J. Appl. Cryst. 13 (1980) 168). This is a Moore-style IFT *inspired by*
SasView's Inversion perspective, not a numeric port: the regularization
operator, uncertainty semantics and heuristics are re-derived, with SasView's
exact operator available as ``regularizer='sasview'`` for comparison.

Units and conventions: q in 1/Angstrom, r and d_max in Angstrom; intensity in
whatever units the file uses. P(r) is defined by ``I(0) = 4*pi * integral(P dr)``
in those same intensity units. P(0) = P(d_max) = 0 by construction of the
basis; the fit is unconstrained, so P(r) can go negative (unlike GNOM/ATSAS) —
the ``positive_fraction`` diagnostics quantify this.

Example:
    >>> from sans_fitter import data_ops, pr_inversion
    >>> data = data_ops.load('protein.csv')
    >>> scan = pr_inversion.explore_dmax(data, d_max=120.0, fit_background=False)
    >>> scan.plot()
    >>> result = pr_inversion.auto_invert(data, d_max=120.0, fit_background=False)
    >>> print(result.format_summary())
    >>> result.plot_pr()
    >>> result.plot_fit(data)

Limitations:
    - Slit smearing (USANS) and pinhole dQ resolution are not supported; a
      warning is emitted when slit-smearing columns carry real data.
    - Buffer-subtracted data (the usual protein case) should use
      ``fit_background=False`` — the fitted flat background of the default can
      absorb I(0) and bias Rg on already-subtracted data.
"""

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from .data.loader import get_fit_index, has_real_data

logger = logging.getLogger(__name__)

__all__ = [
    'invert',
    'auto_invert',
    'estimate_alpha',
    'estimate_n_terms',
    'explore_dmax',
    'PrResult',
    'DmaxScan',
    'AlphaEstimate',
    'NTermsEstimate',
    'InsufficientDataError',
    'PrEstimationError',
]

DEFAULT_N_TERMS = 10
DEFAULT_R_POINTS = 101
REGULARIZERS = ('corrected', 'sasview')
SASVIEW_N_REG = 20
N_TERMS_SCAN_MIN = 10
N_TERMS_SCAN_MIN_SMALL = 4
N_TERMS_SCAN_MAX = 50
RESIDUAL_DOF_RESERVE = 3
OSCILLATION_SCAN_LIMIT = 10.0
N_TERMS_CHI_FACTOR = 1.5
ALPHA_DESCENT_FACTOR = 0.33
ALPHA_DESCENT_STEPS = 30
DISCREPANCY_TARGET_FACTOR = 1.0
PEAK_FLATNESS_RTOL = 1e-3
FABRICATED_SIGMA_RELATIVE = 0.05
FABRICATED_SIGMA_FLOOR_FRACTION = 0.01
POSITIVE_FRACTION_BUCKETS = ((0.9, 1.0 + 1e-12), (0.8, 0.9), (0.7, 0.8))
POLE_RELATIVE_TOLERANCE = 1e-8
DMAX_SCAN_POINTS = 25
DMAX_SCAN_LOW_FACTOR = 0.9
DMAX_SCAN_HIGH_FACTOR = 1.1


class InsufficientDataError(ValueError):
    """The dataset does not carry enough usable points for the request."""


class PrEstimationError(RuntimeError):
    """A parameter estimation scan found no acceptable candidate."""


# ---------------------------------------------------------------------------
# Basis functions (exact mathematics layer)
# ---------------------------------------------------------------------------


def _ortho(d_max: float, n: int, r: np.ndarray) -> np.ndarray:
    """Basis function Phi_n(r) = 2 r sin(pi n r / d_max) on [0, d_max]."""
    r = np.asarray(r, dtype=float)
    return 2.0 * r * np.sin(np.pi * n * r / d_max)


def _ortho_derived(d_max: float, n: int, r: np.ndarray) -> np.ndarray:
    """First derivative Phi'_n(r) = 2[sin(kr) + kr cos(kr)], k = pi n / d_max."""
    r = np.asarray(r, dtype=float)
    kr = np.pi * n * r / d_max
    return 2.0 * (np.sin(kr) + kr * np.cos(kr))


def _ortho_second_derivative(d_max: float, n: int, r: np.ndarray) -> np.ndarray:
    """True second derivative Phi''_n(r) = 2k[2cos(kr) - kr sin(kr)]."""
    r = np.asarray(r, dtype=float)
    k = np.pi * n / d_max
    kr = k * r
    return 2.0 * k * (2.0 * np.cos(kr) - kr * np.sin(kr))


def _sasview_reg_term(d_max: float, n: int, r: np.ndarray) -> np.ndarray:
    """SasView's smoothing operator 2k[2cos(kr) + kr sin(kr)] (sign quirk kept)."""
    r = np.asarray(r, dtype=float)
    k = np.pi * n / d_max
    kr = k * r
    return 2.0 * k * (2.0 * np.cos(kr) + kr * np.sin(kr))


def _ortho_transformed(d_max: float, n: int, q: np.ndarray) -> np.ndarray:
    """Fourier transform of Phi_n evaluated at q (vectorized over q).

    ``Phi~_n(q) = 8 pi^2 D n (-1)^(n+1) sin(qD) / (q [(pi n)^2 - (qD)^2])``

    The removable singularity at ``qD = pi n`` is handled analytically with
    the limit ``Phi~_n(pi n / D) = 4 D^2 / n`` inside a floating-point-scale
    tolerance band; the ``q = 0`` limit is ``8 D^2 (-1)^(n+1) / n``.
    """
    q = np.asarray(q, dtype=float)
    u = q * d_max
    pin = np.pi * n
    sign = -1.0 if n % 2 == 0 else 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        value = 8.0 * np.pi**2 * d_max * n * sign * np.sin(u) / (q * (pin**2 - u**2))
    at_zero = q == 0.0
    at_pole = np.abs(u - pin) <= POLE_RELATIVE_TOLERANCE * pin
    value = np.where(at_zero, 8.0 * d_max**2 * sign / n, value)
    value = np.where(at_pole & ~at_zero, 4.0 * d_max**2 / n, value)
    return value


def _pr_curve_and_band(
    coefficients: np.ndarray, coefficient_cov: np.ndarray, d_max: float, r: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate P(r) and its 1-sigma band (full quadratic form) on a grid."""
    phi = np.column_stack([_ortho(d_max, j + 1, r) for j in range(len(coefficients))])
    pr = phi @ coefficients
    variance = np.einsum('ij,jk,ik->i', phi, coefficient_cov, phi)
    return pr, np.sqrt(np.maximum(variance, 0.0))


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
# Result object
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class PrResult:
    """Result of a P(r) inversion.

    ``coefficients[k]`` is ``c_(k+1)`` (basis index n = 1..n_terms); the
    background is never a coefficient entry. ``covariance`` is the full
    matrix — background row/column *first* when the background was fitted —
    and is a conditional linearized uncertainty: valid at the chosen alpha
    and d_max, assuming known independent Gaussian errors, biased by the
    regularization. ``q_fit``/``iq_fit``/``sigma_fit`` are the accepted data
    q-points, the fit evaluated there, and the sigma actually used (needed
    for residuals when uncertainties were fabricated); smooth curves come
    from :meth:`evaluate_iq` on a dense grid.

    ``regularization_penalty`` is ``alpha * ||L c||^2`` in the active
    regularizer's own scaling: approximately ``alpha * integral(P''(r)^2 dr)``
    for ``'corrected'``, SasView's native ``(D/N_r)^2`` row scaling for
    ``'sasview'`` — the values are not comparable across modes.
    """

    d_max: float
    n_terms: int
    alpha: float
    regularizer: str
    coefficients: np.ndarray
    covariance: np.ndarray
    background: float
    background_fitted: bool
    background_err: float
    data_chisq: float
    effective_dof: float
    regularization_penalty: float
    n_points_used: int
    accepted: np.ndarray
    condition_number: float
    rank: int
    uncertainties_fabricated: bool
    n_dropped_points: int
    rg: float
    i0: float
    oscillations: float
    positive_fraction: float
    sigma_positive_fraction: float
    r: np.ndarray
    pr: np.ndarray
    pr_err: np.ndarray
    q_fit: np.ndarray
    iq_fit: np.ndarray
    sigma_fit: np.ndarray

    @property
    def coefficient_covariance(self) -> np.ndarray:
        """Coefficient block of the covariance (background row/column removed)."""
        if self.background_fitted:
            return self.covariance[1:, 1:]
        return self.covariance

    def evaluate_pr(self, r: np.ndarray) -> np.ndarray:
        """Evaluate P(r) on an arbitrary r grid."""
        r = np.asarray(r, dtype=float)
        total = np.zeros_like(r)
        for j, c in enumerate(self.coefficients):
            total += c * _ortho(self.d_max, j + 1, r)
        return total

    def evaluate_pr_err(self, r: np.ndarray) -> np.ndarray:
        """Evaluate the P(r) uncertainty band via the full quadratic form."""
        r = np.asarray(r, dtype=float)
        _, band = _pr_curve_and_band(self.coefficients, self.coefficient_covariance, self.d_max, r)
        return band

    def evaluate_iq(self, q: np.ndarray) -> np.ndarray:
        """Evaluate the fitted I(q) (including the background) on an arbitrary q grid."""
        q = np.asarray(q, dtype=float)
        total = np.full_like(q, self.background)
        for j, c in enumerate(self.coefficients):
            total += c * _ortho_transformed(self.d_max, j + 1, q)
        return total

    def format_summary(self) -> str:
        """Return an ASCII table of inputs, quality diagnostics and derived outputs.

        The goodness-of-fit line is labelled "approx. chi2 per residual dof":
        ``data_chisq / (n_points_used - effective_dof)``, where
        ``effective_dof`` counts the *fitted* dimensions of data space
        (``tr(H)``), not the number of parameters. It is an approximate
        diagnostic for a regularized fit, and not interpretable at all when
        uncertainties were fabricated.
        """
        residual_dof = self.n_points_used - self.effective_dof
        approx_chi2 = self.data_chisq / residual_dof if residual_dof > 0 else float('nan')
        background_note = 'fitted' if self.background_fitted else 'fixed'
        lines = [
            'P(r) inversion summary',
            '----------------------',
            f'{"D_max (Ang)":<28} {self.d_max:.6g}',
            f'{"Number of terms":<28} {self.n_terms}',
            f'{"Alpha":<28} {self.alpha:.6g}',
            f'{"Regularizer":<28} {self.regularizer}',
            f'{f"Background ({background_note})":<28} {self.background:.6g}'
            + (f' +/- {self.background_err:.3g}' if self.background_fitted else ''),
            f'{"Rg (Ang)":<28} {self.rg:.6g}',
            f'{"I(0)":<28} {self.i0:.6g}',
            f'{"Oscillations":<28} {self.oscillations:.4g}',
            f'{"Positive fraction":<28} {self.positive_fraction:.4g}',
            f'{"1-sigma positive fraction":<28} {self.sigma_positive_fraction:.4g}',
            f'{"Data chi-squared":<28} {self.data_chisq:.6g}',
            f'{"Effective dof (tr H)":<28} {self.effective_dof:.4g}',
            f'{"Approx. chi2 per residual dof":<28} {approx_chi2:.4g}',
            f'{"Points used":<28} {self.n_points_used} (of {self.accepted.size})',
            f'{"Condition number":<28} {self.condition_number:.4g}',
        ]
        if self.uncertainties_fabricated:
            lines.append(
                'WARNING: intensity uncertainties were fabricated; '
                'chi-squared-based diagnostics are not interpretable.'
            )
        if self.n_dropped_points:
            lines.append(f'NOTE: {self.n_dropped_points} point(s) dropped during preparation.')
        return '\n'.join(lines)

    def save_csv(self, filename: str) -> None:
        """Save inputs, diagnostics and the P(r) curve (r, P, dP columns) to CSV."""
        with open(filename, 'w') as f:
            f.write('# P(r) Inversion Results\n')
            f.write(f'# D_max (Ang): {self.d_max:.6g}\n')
            f.write(f'# Number of terms: {self.n_terms}\n')
            f.write(f'# Alpha: {self.alpha:.6g}\n')
            f.write(f'# Regularizer: {self.regularizer}\n')
            f.write(f'# Background: {self.background:.6g}\n')
            f.write(f'# Background fitted: {self.background_fitted}\n')
            if self.background_fitted:
                f.write(f'# Background uncertainty: {self.background_err:.6g}\n')
            f.write(f'# Rg (Ang): {self.rg:.6g}\n')
            f.write(f'# I(0): {self.i0:.6g}\n')
            f.write(f'# Oscillations: {self.oscillations:.6g}\n')
            f.write(f'# Positive fraction: {self.positive_fraction:.6g}\n')
            f.write(f'# 1-sigma positive fraction: {self.sigma_positive_fraction:.6g}\n')
            f.write(f'# Data chi-squared: {self.data_chisq:.6g}\n')
            f.write(f'# Effective dof: {self.effective_dof:.6g}\n')
            f.write(f'# Points used: {self.n_points_used} of {self.accepted.size}\n')
            f.write(f'# Uncertainties fabricated: {self.uncertainties_fabricated}\n')
            f.write(f'# Points dropped in preparation: {self.n_dropped_points}\n')
            f.write('#\n')
            f.write('r,P(r),dP(r)\n')
            for r_value, p_value, dp_value in zip(self.r, self.pr, self.pr_err, strict=True):
                f.write(f'{r_value:.6e},{p_value:.6e},{dp_value:.6e}\n')

    def plot_pr(self, show: bool | None = None):
        """Plot P(r) with its uncertainty band. Same display convention as plot_results()."""
        from .plotting import plot_pr_distribution

        return plot_pr_distribution(self, show=show)

    def plot_fit(self, data: Any, show: bool | None = None, log_scale: bool = True):
        """Plot data vs the fitted I(q) with residuals.

        The dataset is passed explicitly — the result stores the model and the
        accepted mask, not the observed intensities. Same display convention
        as plot_results(). Pass ``log_scale=False`` when intensities include
        zero or negative values (a log axis silently omits such points).
        """
        from .plotting import plot_pr_fit

        return plot_pr_fit(data, self, show=show, log_scale=log_scale)


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


def _count_peaks(values: np.ndarray) -> int:
    """Count local maxima of a curve, ignoring near-flat slopes.

    A slope with magnitude below ``PEAK_FLATNESS_RTOL * max|P|`` is treated as
    flat (SasView's raw sign-change count is fragile on near-flat P(r)).
    Grid endpoints are excluded by construction.
    """
    values = np.asarray(values, dtype=float)
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    if scale == 0.0:
        return 0
    slopes = np.diff(values)
    tol = PEAK_FLATNESS_RTOL * scale
    signs = np.where(slopes > tol, 1, np.where(slopes < -tol, -1, 0))
    signs = signs[signs != 0]
    if signs.size < 2:
        return 0
    return int(np.sum((signs[:-1] == 1) & (signs[1:] == -1)))


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


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlphaEstimate:
    """Result of :func:`estimate_alpha`."""

    alpha: float
    message: str


@dataclass(frozen=True)
class NTermsEstimate:
    """Result of :func:`estimate_n_terms`. ``alpha`` was evaluated at the
    chosen ``n_terms`` during the scan and is the authoritative companion
    value — use it directly rather than re-estimating."""

    n_terms: int
    alpha: float
    message: str


def _alpha_suggestion(
    prep: _PreparedData,
    d_max: float,
    n_terms: int,
    fit_background: bool,
    regularizer: str,
) -> float:
    """Frobenius-balance suggestion: ``||A_data||_F^2 / ||L(alpha=1)||_F^2``."""
    a_data, _, reg_unit = _build_blocks(
        prep.q, prep.intensity, prep.sigma, d_max, n_terms, fit_background, regularizer
    )
    reg_norm = float(np.sum(reg_unit**2))
    if reg_norm == 0.0:
        raise PrEstimationError('The regularization block is empty; cannot suggest alpha.')
    return float(np.sum(a_data**2)) / reg_norm


def _estimate_alpha_prepared(
    prep: _PreparedData,
    d_max: float,
    n_terms: int,
    fit_background: bool,
    regularizer: str,
    background: float = 0.0,
) -> AlphaEstimate:
    """Alpha heuristic on prepared data (shared by the public estimators).

    Geometric descent from the Frobenius-balance suggestion, stopped by
    whichever criterion fires first:

    - **Spurious structure** (a second peak in P(r)) — return the previous,
      larger alpha. This is SasView's criterion; it protects noisy data.
    - **Discrepancy principle** — return the first alpha whose data block
      chi-squared drops to ``N_acc`` (Morozov: with known uncertainties, a fit
      should not do better than the noise). This is what actually selects
      alpha for smooth single-peak shapes, where a second peak may never
      appear and the peak criterion alone would keep the massively
      over-smoothed starting suggestion (found in Step-5 harness tuning).
    """
    alpha_suggested = _alpha_suggestion(prep, d_max, n_terms, fit_background, regularizer)
    chisq_target = DISCREPANCY_TARGET_FACTOR * prep.q.size
    failures: list[str] = []
    previous_alpha: float | None = None

    for step in range(ALPHA_DESCENT_STEPS + 1):
        alpha_step = alpha_suggested * ALPHA_DESCENT_FACTOR**step
        try:
            result = _invert_prepared(
                prep,
                d_max,
                n_terms,
                alpha_step,
                fit_background,
                background,
                DEFAULT_R_POINTS,
                regularizer,
            )
        except (ValueError, np.linalg.LinAlgError) as e:
            failures.append(f'alpha={alpha_step:g}: {e}')
            continue
        n_peaks = _count_peaks(result.pr)
        if n_peaks > 1:
            chosen = previous_alpha if previous_alpha is not None else alpha_step
            return AlphaEstimate(
                alpha=chosen,
                message=f'Largest alpha before spurious structure ({n_peaks} peaks '
                f'at alpha={alpha_step:g}).',
            )
        if result.data_chisq <= chisq_target:
            return AlphaEstimate(
                alpha=alpha_step,
                message='Largest alpha satisfying the discrepancy principle '
                f'(data chi-squared {result.data_chisq:.4g} <= {chisq_target:.4g}).',
            )
        previous_alpha = alpha_step

    if previous_alpha is None:
        raise PrEstimationError(
            'No alpha in the scan produced a solvable inversion: ' + '; '.join(failures)
        )
    return AlphaEstimate(
        alpha=previous_alpha,
        message='Discrepancy target not reached; returning the smallest alpha '
        'scanned without spurious structure.',
    )


def estimate_alpha(
    data: Any,
    d_max: float,
    n_terms: int,
    fit_background: bool = True,
    regularizer: str = 'corrected',
    background: float = 0.0,
) -> AlphaEstimate:
    """Estimate the regularization constant alpha for a given number of terms.

    Starts from the Frobenius-balance suggestion (the alpha that balances the
    data and unit-alpha penalty blocks) and descends geometrically, returning
    the largest alpha just before spurious structure (a second peak) appears
    in P(r), or — for smooth single-peak shapes where that never happens —
    the largest alpha satisfying the discrepancy principle (data chi-squared
    down to the number of points).

    ``background`` is the fixed level subtracted when ``fit_background`` is
    False (ignored otherwise) — pass the same value you will pass to
    :func:`invert`, so the estimate is made on the problem actually solved.
    The heuristic evaluates candidates on the default 101-point r grid
    regardless of the ``r_points`` used for the final inversion.

    Raises:
        TypeError: If the dataset is not 1D (2D or SESANS data).
        PrEstimationError: When no alpha in the scan yields a solvable
            inversion.
    """
    _validate_invert_args(d_max, n_terms, 0.0, DEFAULT_R_POINTS, regularizer, background)
    prep = _prepare_data(data)
    return _estimate_alpha_prepared(prep, d_max, n_terms, fit_background, regularizer, background)


def _admissible_n_range(prep: _PreparedData, d_max: float, fit_background: bool) -> tuple[int, int]:
    """Return the (n_min, n_max) scan bounds; raise when no N is admissible."""
    n_bg = 1 if fit_background else 0
    n_acc = int(prep.q.size)
    shannon = math.ceil(float(prep.q.max()) * d_max / np.pi)
    n_max = min(N_TERMS_SCAN_MAX, n_acc - n_bg - RESIDUAL_DOF_RESERVE, shannon)
    n_min = N_TERMS_SCAN_MIN
    if n_max < n_min:
        n_min = N_TERMS_SCAN_MIN_SMALL
    if n_max < n_min:
        raise InsufficientDataError(
            f'Only {n_acc} usable points (Shannon channels: {shannon}); '
            f'cannot scan any admissible number of terms (minimum {n_min}).'
        )
    return n_min, n_max


def _estimate_n_terms_prepared(
    prep: _PreparedData,
    d_max: float,
    fit_background: bool,
    regularizer: str,
    background: float = 0.0,
) -> NTermsEstimate:
    """N-terms heuristic on prepared data (shared by the public entry points)."""
    n_min, n_max = _admissible_n_range(prep, d_max, fit_background)

    # candidate tuples: (n, alpha, oscillations, pos_1sigma, data_chisq)
    candidates: list[tuple[int, float, float, float, float]] = []
    failures: list[str] = []
    for n in range(n_min, n_max + 1):
        try:
            alpha_est = _estimate_alpha_prepared(
                prep, d_max, n, fit_background, regularizer, background
            )
            result = _invert_prepared(
                prep,
                d_max,
                n,
                alpha_est.alpha,
                fit_background,
                background,
                DEFAULT_R_POINTS,
                regularizer,
            )
        except (ValueError, np.linalg.LinAlgError, PrEstimationError) as e:
            failures.append(f'n={n}: {e}')
            continue
        candidates.append(
            (
                n,
                alpha_est.alpha,
                result.oscillations,
                result.sigma_positive_fraction,
                result.data_chisq,
            )
        )
        if result.oscillations > OSCILLATION_SCAN_LIMIT:
            logger.debug('n-terms scan stopped at n=%d (oscillations %.3g)', n, result.oscillations)
            break

    if not candidates:
        raise PrEstimationError(
            'No number of terms in the scan produced a solvable inversion: ' + '; '.join(failures)
        )

    bucket: list[tuple[int, float, float, float, float]] = []
    for low, high in POSITIVE_FRACTION_BUCKETS:
        bucket = [c for c in candidates if low <= c[3] < high]
        if bucket:
            break
    if not bucket:
        best = max(c[3] for c in candidates)
        raise PrEstimationError(
            f'No candidate reached a 1-sigma positive fraction of 0.7 '
            f'(best achieved: {best:.3f}). The data may not support a stable '
            'P(r); set n_terms manually and inspect the result.'
        )

    # Prefer the smallest N that actually fits the data (discrepancy within a
    # factor of the point count) — the median-oscillation pick alone selects
    # under-fitting N for elongated particles, whose I(q) needs the full
    # Shannon channel count (found in Step-5 harness tuning). Fall back to the
    # most typical oscillation level when no candidate fits.
    chisq_limit = N_TERMS_CHI_FACTOR * prep.q.size
    fit_ok = [c for c in bucket if c[4] <= chisq_limit]
    if fit_ok:
        chosen = min(fit_ok, key=lambda c: c[0])
        criterion = 'smallest N fitting the data'
    else:
        median_osc = float(np.median([c[2] for c in bucket]))
        chosen = min(bucket, key=lambda c: (abs(c[2] - median_osc), c[0]))
        criterion = 'most typical oscillation level (no N fit the data)'
        warnings.warn(
            'No scanned number of terms fit the data '
            f'(chi-squared <= {N_TERMS_CHI_FACTOR} per point). The data may not '
            'support a smooth single-population P(r) at this D_max (for example '
            'multimodal size distributions); the returned P(r) may fit the data '
            'poorly. Inspect result.format_summary() and plot_fit() before use.',
            stacklevel=3,
        )
    n_chosen, alpha_chosen, osc_chosen, pos_chosen, chisq_chosen = chosen
    return NTermsEstimate(
        n_terms=n_chosen,
        alpha=alpha_chosen,
        message=(
            f'Scanned n = {n_min}..{candidates[-1][0]}; chose n = {n_chosen} by {criterion} '
            f'(1-sigma positive fraction {pos_chosen:.3f}, oscillations {osc_chosen:.3g}, '
            f'data chi-squared/point {chisq_chosen / prep.q.size:.3g}).'
        ),
    )


def estimate_n_terms(
    data: Any,
    d_max: float,
    fit_background: bool = True,
    regularizer: str = 'corrected',
    background: float = 0.0,
) -> NTermsEstimate:
    """Estimate the number of basis terms (and the matching alpha).

    Scans admissible N, preferring the smallest N that fits the data with a
    significantly positive P(r) (1-sigma positive fraction >= 0.9, with 0.8
    and 0.7 fallback buckets). The scan stops early once P(r) becomes wildly
    oscillatory.

    ``background`` is the fixed level subtracted when ``fit_background`` is
    False (ignored otherwise) — pass the same value you will pass to
    :func:`invert`, so the selection is made on the problem actually solved.
    The heuristic evaluates candidates on the default 101-point r grid
    regardless of the ``r_points`` used for the final inversion.

    Raises:
        TypeError: If the dataset is not 1D (2D or SESANS data).
        InsufficientDataError: When no N is admissible for the dataset.
        PrEstimationError: When the scan has no acceptable candidate.
    """
    _validate_invert_args(d_max, 1, 0.0, DEFAULT_R_POINTS, regularizer, background)
    prep = _prepare_data(data)
    return _estimate_n_terms_prepared(prep, d_max, fit_background, regularizer, background)


def auto_invert(
    data: Any,
    d_max: float,
    fit_background: bool = True,
    background: float = 0.0,
    r_points: int = DEFAULT_R_POINTS,
    regularizer: str = 'corrected',
) -> PrResult:
    """One-shot inversion with automatic selection of n_terms and alpha.

    Runs the :func:`estimate_n_terms` scan and inverts with the estimate's
    ``(n_terms, alpha)`` pair. The ``background`` (used when ``fit_background``
    is False) is applied during the selection scan as well, so the parameters
    are chosen on the same problem the final inversion solves. The chosen
    values are recorded on the result (``result.n_terms``, ``result.alpha``).
    """
    _validate_invert_args(d_max, 1, 0.0, r_points, regularizer, background)
    prep = _prepare_data(data)
    estimate = _estimate_n_terms_prepared(prep, d_max, fit_background, regularizer, background)
    logger.debug('auto_invert: %s', estimate.message)
    return _invert_prepared(
        prep,
        d_max,
        estimate.n_terms,
        estimate.alpha,
        fit_background,
        background,
        r_points,
        regularizer,
    )


# ---------------------------------------------------------------------------
# D_max exploration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DmaxScan:
    """Result of :func:`explore_dmax`.

    Arrays are success-only: every quantity array has the same length as
    ``d_max_values``; D_max points whose inversion failed are omitted and
    recorded in ``failures`` as ``(d_max, message)`` pairs.
    """

    d_max_values: np.ndarray
    data_chisq: np.ndarray
    rg: np.ndarray
    i0: np.ndarray
    oscillations: np.ndarray
    positive_fraction: np.ndarray
    sigma_positive_fraction: np.ndarray
    background: np.ndarray
    alpha: np.ndarray
    n_terms: int
    failures: list[tuple[float, str]]

    def format_summary(self) -> str:
        """Return an ASCII table of the scanned quantities per D_max."""
        header = (
            f'{"D_max":>10} {"data_chisq":>12} {"Rg":>10} {"I(0)":>12} '
            f'{"Osc":>8} {"P+":>7} {"P+1s":>7} {"Bkg":>10}'
        )
        lines = [f'D_max scan ({self.n_terms} terms):', header, '-' * len(header)]
        for i, d in enumerate(self.d_max_values):
            lines.append(
                f'{d:>10.4g} {self.data_chisq[i]:>12.6g} {self.rg[i]:>10.4g} '
                f'{self.i0[i]:>12.6g} {self.oscillations[i]:>8.3g} '
                f'{self.positive_fraction[i]:>7.3f} {self.sigma_positive_fraction[i]:>7.3f} '
                f'{self.background[i]:>10.4g}'
            )
        if self.failures:
            lines.append('')
            lines.append('Failed points:')
            for d, message in self.failures:
                lines.append(f'  D_max = {d:.4g}: {message}')
        return '\n'.join(lines)

    def plot(self, quantity: str = 'rg', show: bool | None = None):
        """Plot a scanned quantity (or 'all') vs D_max. Same display convention
        as plot_results()."""
        from .plotting import plot_dmax_scan

        return plot_dmax_scan(self, quantity=quantity, show=show)


def explore_dmax(
    data: Any,
    d_max: float,
    n_terms: int | None = None,
    alpha: float | None = None,
    dmin: float | None = None,
    dmax: float | None = None,
    n_points: int = DMAX_SCAN_POINTS,
    refit_alpha: bool = False,
    fit_background: bool = True,
    regularizer: str = 'corrected',
    background: float = 0.0,
) -> DmaxScan:
    """Re-invert over a range of D_max values to locate a stable choice.

    A good D_max shows a plateau in Rg and I(0) and a minimum in the data
    chi-squared. Defaults: scan ``0.9*d_max .. 1.1*d_max`` in ``n_points``
    steps, with ``n_terms``/``alpha`` estimated once at the central ``d_max``
    and held fixed across the scan (comparable across D_max thanks to the
    corrected operator's resolved quadrature). ``refit_alpha=True`` recomputes
    the alpha suggestion at each D_max instead.

    ``background`` is the fixed level subtracted when ``fit_background`` is
    False (ignored otherwise) — pass the same value you use with
    :func:`invert`/:func:`auto_invert`, so the scan explores the same problem
    the final inversion solves.

    Note:
        The scan suppresses the per-point Shannon-support warnings via the
        process-global ``warnings`` filter, so it is not thread-safe:
        inversions running concurrently in other threads may have those
        warnings swallowed while a scan is in flight.

    Raises:
        TypeError: If the dataset is not 1D (2D or SESANS data).
        ValueError: For an invalid scan range.
        InsufficientDataError / PrEstimationError: From the central estimation
            when ``n_terms``/``alpha`` are not supplied, or when every scan
            point fails.
    """
    _validate_invert_args(
        d_max,
        1 if n_terms is None else n_terms,
        0.0 if alpha is None else alpha,
        DEFAULT_R_POINTS,
        regularizer,
        background,
    )
    _require_integer('n_points', n_points)
    low = DMAX_SCAN_LOW_FACTOR * d_max if dmin is None else dmin
    high = DMAX_SCAN_HIGH_FACTOR * d_max if dmax is None else dmax
    if not (np.isfinite(low) and np.isfinite(high) and 0 < low < high):
        raise ValueError(f'Invalid D_max scan range: [{low}, {high}].')
    if n_points < 2:
        raise ValueError(f'n_points must be at least 2, got {n_points}.')

    prep = _prepare_data(data)

    # One scan-level advisory instead of a per-point repeat (the per-point
    # copies are suppressed inside the loop below).
    support = np.pi / float(prep.q.min())
    if high > support:
        warnings.warn(
            f'Part of the D_max scan range ([{low:g}, {high:g}]) exceeds '
            f'pi/q_min = {support:g}; distances beyond it are only weakly '
            'constrained by the lowest measured q (low-q support heuristic).',
            stacklevel=2,
        )

    if n_terms is None:
        estimate = _estimate_n_terms_prepared(prep, d_max, fit_background, regularizer, background)
        n_terms = estimate.n_terms
        if alpha is None:
            alpha = estimate.alpha
    elif alpha is None:
        alpha = _estimate_alpha_prepared(
            prep, d_max, n_terms, fit_background, regularizer, background
        ).alpha

    scan_values = np.linspace(low, high, n_points)
    collected: dict[str, list[float]] = {
        key: []
        for key in (
            'd_max',
            'data_chisq',
            'rg',
            'i0',
            'oscillations',
            'positive_fraction',
            'sigma_positive_fraction',
            'background',
            'alpha',
        )
    }
    failures: list[tuple[float, str]] = []
    for d in scan_values:
        try:
            # The scan deliberately varies D_max at fixed N, so both per-point
            # support warnings (channel count and pi/q_min) are noise by
            # construction here — the scan-level advisory above covers them.
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*Shannon channels.*')
                warnings.filterwarnings('ignore', message='.*pi/q_min.*')
                alpha_d = (
                    _estimate_alpha_prepared(
                        prep, float(d), n_terms, fit_background, regularizer, background
                    ).alpha
                    if refit_alpha
                    else alpha
                )
                result = _invert_prepared(
                    prep,
                    float(d),
                    n_terms,
                    alpha_d,
                    fit_background,
                    background,
                    DEFAULT_R_POINTS,
                    regularizer,
                )
        except (ValueError, np.linalg.LinAlgError, PrEstimationError) as e:
            failures.append((float(d), str(e)))
            continue
        collected['d_max'].append(float(d))
        collected['data_chisq'].append(result.data_chisq)
        collected['rg'].append(result.rg)
        collected['i0'].append(result.i0)
        collected['oscillations'].append(result.oscillations)
        collected['positive_fraction'].append(result.positive_fraction)
        collected['sigma_positive_fraction'].append(result.sigma_positive_fraction)
        collected['background'].append(result.background)
        collected['alpha'].append(alpha_d)

    if not collected['d_max']:
        details = '; '.join(f'D_max={d:g}: {message}' for d, message in failures)
        raise PrEstimationError(f'Every D_max scan point failed: {details}')

    return DmaxScan(
        d_max_values=np.asarray(collected['d_max']),
        data_chisq=np.asarray(collected['data_chisq']),
        rg=np.asarray(collected['rg']),
        i0=np.asarray(collected['i0']),
        oscillations=np.asarray(collected['oscillations']),
        positive_fraction=np.asarray(collected['positive_fraction']),
        sigma_positive_fraction=np.asarray(collected['sigma_positive_fraction']),
        background=np.asarray(collected['background']),
        alpha=np.asarray(collected['alpha']),
        n_terms=int(n_terms),
        failures=failures,
    )

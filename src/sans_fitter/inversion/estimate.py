"""Heuristics: alpha and n_terms estimation, auto_invert() and D_max scans."""

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from .result import InsufficientDataError, PrEstimationError, PrResult
from .solver import (
    DEFAULT_R_POINTS,
    _build_blocks,
    _invert_prepared,
    _prepare_data,
    _PreparedData,
    _require_integer,
    _validate_invert_args,
)

logger = logging.getLogger(__name__)

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
POSITIVE_FRACTION_BUCKETS = ((0.9, 1.0 + 1e-12), (0.8, 0.9), (0.7, 0.8))
DMAX_SCAN_POINTS = 25
DMAX_SCAN_LOW_FACTOR = 0.9
DMAX_SCAN_HIGH_FACTOR = 1.1


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
            if previous_alpha is None:
                # Even the first solvable (largest) alpha shows structure, so
                # there is no structure-free alpha to fall back on.
                return AlphaEstimate(
                    alpha=alpha_step,
                    message='No alpha in the scan was free of spurious structure; '
                    f'returning alpha={alpha_step:g} despite {n_peaks} peaks.',
                )
            return AlphaEstimate(
                alpha=previous_alpha,
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
        from ..plotting import plot_dmax_scan

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

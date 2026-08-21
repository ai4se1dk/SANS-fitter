"""Result objects and errors for P(r) inversion."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .basis import _ortho, _ortho_transformed, _pr_curve_and_band


class InsufficientDataError(ValueError):
    """The dataset does not carry enough usable points for the request."""


class PrEstimationError(RuntimeError):
    """A parameter estimation scan found no acceptable candidate."""


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
        from ..plotting import plot_pr_distribution

        return plot_pr_distribution(self, show=show)

    def plot_fit(self, data: Any, show: bool | None = None, log_scale: bool = True):
        """Plot data vs the fitted I(q) with residuals.

        The dataset is passed explicitly — the result stores the model and the
        accepted mask, not the observed intensities. Same display convention
        as plot_results(). Pass ``log_scale=False`` when intensities include
        zero or negative values (a log axis silently omits such points).
        """
        from ..plotting import plot_pr_fit

        return plot_pr_fit(data, self, show=show, log_scale=log_scale)

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

from .estimate import (
    AlphaEstimate,
    DmaxScan,
    NTermsEstimate,
    auto_invert,
    estimate_alpha,
    estimate_n_terms,
    explore_dmax,
)
from .result import InsufficientDataError, PrEstimationError, PrResult
from .solver import DEFAULT_N_TERMS, DEFAULT_R_POINTS, REGULARIZERS, SASVIEW_N_REG, invert

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
    'DEFAULT_N_TERMS',
    'DEFAULT_R_POINTS',
    'REGULARIZERS',
    'SASVIEW_N_REG',
]

"""Sine-basis functions and their transforms (exact mathematics layer)."""

import numpy as np

POLE_RELATIVE_TOLERANCE = 1e-8


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

"""Exact Debye sums for a set of point scatterers.

Both functions return the same quantity::

    D(q) = sum_j sum_k w_j w_k sin(q r_jk) / (q r_jk)

for weights ``w`` (scattering lengths, Å) at positions in Å. ``debye_pairs``
enumerates pairs and works for any point set. ``debye_lattice`` regroups the
same sum through the autocorrelation of a weight grid, which is exact when
every point sits on a lattice and much faster for large grids.
"""

from collections.abc import Callable

import numpy as np

#: Pairs per block. Small blocks stay in cache; larger ones were measured slower.
PAIR_BLOCK_ELEMENTS = 50_000

#: Largest zero-padded grid the lattice path will allocate, in cells.
LATTICE_MAX_CELLS = 8_000_000


class CalculationInterrupted(Exception):
    """Raised when a progress callback asks a running calculation to stop."""


def _sinc_sum(q: np.ndarray, r: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Return ``sum(weight * sin(q r) / (q r))`` for every q, with sinc(0) = 1.

    Pairs at r = 0 and q values of 0 contribute their full weight. The two
    buffers are reused across q, so memory does not grow with len(q).
    """
    out = np.empty(q.size)
    at_zero = r == 0
    flat = weight[at_zero].sum()
    r, weight = r[~at_zero], weight[~at_zero]
    total = weight.sum()
    qr = np.empty_like(r)
    s = np.empty_like(r)
    for i, qi in enumerate(q):
        if qi == 0:
            out[i] = flat + total
            continue
        np.multiply(r, qi, out=qr)
        np.sin(qr, out=s)
        s /= qr
        out[i] = flat + np.dot(weight, s)
    return out


def debye_pairs(
    q,
    xyz,
    w,
    *,
    block: int | None = None,
    progress: Callable[[float], bool | None] | None = None,
) -> np.ndarray:
    """Exact Debye sum by enumerating the upper triangle of pairs.

    Args:
        q: Momentum transfers, 1/Å, any values >= 0.
        xyz: Positions, shape (N, 3), Å.
        w: Weights, shape (N,).
        block: Rows per block. Defaults to a size that keeps each pair buffer
            near ``PAIR_BLOCK_ELEMENTS``.
        progress: Called with the completed fraction after each block. A
            return value of True stops the sum with ``CalculationInterrupted``.
    """
    q = np.asarray(q, dtype=float)
    xyz = np.asarray(xyz, dtype=float)
    w = np.asarray(w, dtype=float)
    n = w.size
    result = np.full(q.size, float(np.dot(w, w)))  # self terms, r = 0
    if n < 2:
        return result
    block = max(1, block or PAIR_BLOCK_ELEMENTS // n)
    total_pairs = n * (n - 1) / 2
    done = 0
    for start in range(0, n - 1, block):
        stop = min(start + block, n - 1)
        rows = np.arange(start, stop)
        cols = np.arange(start + 1, n)
        upper = cols[None, :] > rows[:, None]
        delta = xyz[rows][:, None, :] - xyz[cols][None, :, :]
        r = np.sqrt(np.einsum('ijk,ijk->ij', delta, delta))[upper]
        weight = (w[rows][:, None] * w[cols][None, :])[upper]
        result += 2.0 * _sinc_sum(q, r, weight)
        done += weight.size
        if progress is not None and progress(done / total_pairs):
            raise CalculationInterrupted('Debye sum stopped by the progress callback.')
    return result


def lattice_cells(shape) -> int:
    """Cells in the zero-padded grid ``debye_lattice`` allocates for *shape*."""
    return int(np.prod([2 * n - 1 for n in shape]))


def debye_lattice(q, grid_w, step) -> np.ndarray:
    """Exact Debye sum for weights on a rectangular lattice.

    On a lattice every pair displacement is an integer vector Δ, so::

        C(Δ) = sum_x w(x) w(x + Δ)          (autocorrelation of the grid)
        D(q) = sum_Δ C(Δ) sinc(q |Δ|)

    Lags are grouped by the exact integer key (|i|, |j|, |k|), so no
    floating-point rounding enters the grouping.

    Args:
        q: Momentum transfers, 1/Å.
        grid_w: Weight grid, shape (nx, ny, nz); empty sites hold 0.
        step: Lattice steps along x, y and z, Å.
    """
    q = np.asarray(q, dtype=float)
    grid_w = np.asarray(grid_w, dtype=float)
    shape = grid_w.shape
    padded = [2 * n - 1 for n in shape]
    spectrum = np.fft.rfftn(grid_w, s=padded, axes=(0, 1, 2))
    corr = np.fft.irfftn(spectrum * spectrum.conj(), s=padded, axes=(0, 1, 2))

    # Index i of a padded axis of length m = 2n - 1 holds lag i for i < n and
    # lag i - m above that; only |lag| matters for the distance.
    lags = [np.minimum(np.arange(m), m - np.arange(m)) for m in padded]
    key = (
        lags[0][:, None, None] * (shape[1] * shape[2])
        + lags[1][None, :, None] * shape[2]
        + lags[2][None, None, :]
    )
    summed = np.bincount(key.ravel(), weights=corr.ravel(), minlength=int(np.prod(shape)))
    ix, iy, iz = np.unravel_index(np.arange(summed.size), shape)
    step = np.asarray(step, dtype=float)
    r = np.sqrt((ix * step[0]) ** 2 + (iy * step[1]) ** 2 + (iz * step[2]) ** 2)
    return _sinc_sum(q, r, summed)

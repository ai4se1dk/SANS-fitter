from typing import Any

import numpy as np
from sasdata.dataloader.loader import Loader


def _has_real_data(arr) -> bool:
    """Return True when *arr* contains at least one finite, non-zero value.

    sasdata zero-fills optional columns (dI, dQ) that are absent from the input
    file, so an all-zero (or all-NaN) array means the column was not provided.
    """
    if arr is None or getattr(arr, 'size', 0) == 0:
        return False
    return bool(np.any(np.nan_to_num(np.asarray(arr, dtype=float)) != 0))


def get_fit_index(data: Any) -> np.ndarray:
    """Return the boolean index of points included in the fit.

    Mirrors how sasmodels interprets 1D data: a point is fitted when it lies
    inside [qmin, qmax], is not masked, and has a finite intensity.
    """
    x = np.asarray(data.x)
    index = (x >= data.qmin) & (x <= data.qmax)
    mask = getattr(data, 'mask', None)
    if mask is not None:
        index &= ~np.asarray(mask, dtype=bool)
    index &= ~np.isnan(np.asarray(data.y))
    return index


def normalize_sans_data(data: Any) -> Any:
    """Ensure *data* carries the qmin/qmax/mask attributes the fit engines need.

    Both fit engines rely on ``qmin``, ``qmax`` and ``mask`` being present on
    the dataset. Loaded files get them here via :func:`load_sans_data`;
    datasets built in memory (arithmetic results, simulations) get them via
    :meth:`SANSFitter.set_data` or :mod:`sans_fitter.data_ops`.

    ``qmin``/``qmax`` are only computed when absent (explicit ``is None``
    check, so a legitimate limit of ``0.0`` is preserved). The mask is the
    union of any existing mask and the NaN positions in x, y and — when the
    columns carry real data — dy and dx.
    """
    qmin = getattr(data, 'qmin', None)
    qmax = getattr(data, 'qmax', None)
    data.qmin = data.x.min() if qmin is None else qmin
    data.qmax = data.x.max() if qmax is None else qmax

    existing_mask = getattr(data, 'mask', None)
    if existing_mask is None or np.asarray(existing_mask).size != np.asarray(data.y).size:
        existing_mask = np.zeros_like(data.y, dtype=bool)
    existing_mask = np.asarray(existing_mask, dtype=bool)
    nan_mask = np.isnan(data.x) | np.isnan(data.y)
    if _has_real_data(data.dy):
        nan_mask |= np.isnan(data.dy)
    if _has_real_data(data.dx):
        nan_mask |= np.isnan(data.dx)
    data.mask = existing_mask | nan_mask
    return data


def load_sans_data(filename: str) -> Any:
    """Load SANS data and normalize required fields for downstream fitting."""
    loader = Loader()

    try:
        data_list = loader.load(filename)
        if not data_list:
            raise ValueError(f'No data loaded from {filename}')

        return normalize_sans_data(data_list[0])
    except Exception as e:
        raise ValueError(f'Failed to load data from {filename}: {str(e)}') from e

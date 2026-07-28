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


def load_sans_data(filename: str) -> Any:
    """Load SANS data and normalize required fields for downstream fitting."""
    loader = Loader()

    try:
        data_list = loader.load(filename)
        if not data_list:
            raise ValueError(f'No data loaded from {filename}')

        data = data_list[0]
        qmin = getattr(data, 'qmin', None)
        qmax = getattr(data, 'qmax', None)
        data.qmin = data.x.min() if qmin is None else qmin
        data.qmax = data.x.max() if qmax is None else qmax

        existing_mask = np.asarray(
            getattr(data, 'mask', np.zeros_like(data.y, dtype=bool)),
            dtype=bool,
        )
        nan_mask = np.isnan(data.x) | np.isnan(data.y)
        if _has_real_data(data.dy):
            nan_mask |= np.isnan(data.dy)
        if _has_real_data(data.dx):
            nan_mask |= np.isnan(data.dx)
        data.mask = existing_mask | nan_mask
        return data
    except Exception as e:
        raise ValueError(f'Failed to load data from {filename}: {str(e)}') from e

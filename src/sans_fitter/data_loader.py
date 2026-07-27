from typing import Any

import numpy as np
from sasdata.dataloader.loader import Loader


def _has_real_data(arr) -> bool:
    """Return True when *arr* is a non-empty array containing at least one non-NaN value."""
    return arr is not None and getattr(arr, 'size', 0) > 0 and not np.all(np.isnan(arr))


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

from typing import Any

import numpy as np
from sasdata.dataloader.loader import Loader


def load_sans_data(filename: str) -> Any:
    """Load SANS data and normalize required fields for downstream fitting."""
    loader = Loader()

    try:
        data_list = loader.load(filename)
        if not data_list:
            raise ValueError(f'No data loaded from {filename}')

        data = data_list[0]
        data.qmin = getattr(data, 'qmin', None) or data.x.min()
        data.qmax = getattr(data, 'qmax', None) or data.x.max()
        data.mask = np.isnan(data.y)
        return data
    except Exception as e:
        raise ValueError(f'Failed to load data from {filename}: {str(e)}') from e

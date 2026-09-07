import os
import warnings
from typing import Any

import numpy as np
from sasdata.dataloader.loader import Loader


def has_real_data(arr) -> bool:
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
    :meth:`SANSFitter.set_data` or :mod:`sans_fitter.data.ops`.

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
    if has_real_data(data.dy):
        nan_mask |= np.isnan(data.dy)
    if has_real_data(data.dx):
        nan_mask |= np.isnan(data.dx)
    data.mask = existing_mask | nan_mask
    return data


def load_sans_data(filename: str, dataset: int | str = 0) -> Any:
    """Load a single SANS dataset from *filename* and make it fit-ready.

    SANS files may hold several datasets (e.g. a CanSAS XML with multiple
    ``SASentry`` blocks, or an NXcanSAS file with several entries). By default
    the first dataset is returned; pass *dataset* to choose another one by
    0-based index or by name (title, run id or filename). When the file
    contains more than one dataset, a warning lists them all so the caller
    knows a different selection was possible.

    Args:
        filename: Path to the data file.
        dataset: Which dataset to return — a 0-based index or a name (title,
            run id or filename). Defaults to the first dataset.

    Returns:
        A fit-ready dataset with ``qmin``/``qmax``/``mask`` set.

    Raises:
        ValueError: If the file cannot be loaded, contains no data, or the
            requested dataset does not exist or is ambiguous.
    """
    data_list = _load_all(filename)
    index = _select_dataset(data_list, filename, dataset)

    if len(data_list) > 1:
        listing = ', '.join(f'{i}: {_dataset_label(d, i)}' for i, d in enumerate(data_list))
        warnings.warn(
            f'{os.path.basename(filename)} contains {len(data_list)} datasets '
            f'({listing}); returning dataset {_dataset_label(data_list[index], index)!r}. '
            'Pass dataset=<index or name> to load a different one.',
            stacklevel=3,
        )

    return normalize_sans_data(data_list[index])


def _load_all(filename: str) -> list[Any]:
    """Return every dataset found in *filename* via sasdata, or raise."""
    loader = Loader()
    try:
        data_list = loader.load(filename)
        if not data_list:
            raise ValueError(f'No data loaded from {filename}')
        return data_list
    except Exception as e:
        raise ValueError(f'Failed to load data from {filename}: {str(e)}') from e


def _dataset_names(data: Any) -> list[str]:
    """Names a dataset can be labelled or selected by (title, run id, filename)."""
    names: list[str] = []
    for attr in ('title', 'name'):
        value = getattr(data, attr, None)
        if value:
            names.append(str(value))
    run = getattr(data, 'run', None)
    names.extend(str(v) for v in (run if isinstance(run, (list, tuple)) else [run]) if v)
    filename = getattr(data, 'filename', None)
    if filename:
        basename = os.path.basename(filename)
        names += [basename, os.path.splitext(basename)[0]]
    return [name for name in dict.fromkeys(names) if name]


def _dataset_label(data: Any, index: int) -> str:
    """Short human-readable name for a loaded dataset, for messages."""
    names = _dataset_names(data)
    return names[0] if names else f'dataset #{index}'


def _select_dataset(data_list: list[Any], filename: str, dataset: int | str) -> int:
    """Resolve *dataset* (int index or str name) to a position in *data_list*."""
    if isinstance(dataset, int) and not isinstance(dataset, bool):  # bool subclasses int
        if not 0 <= dataset < len(data_list):
            raise ValueError(
                f'Dataset index {dataset} is out of range for {filename}: '
                f'file contains {len(data_list)} dataset(s) (0–{len(data_list) - 1}).'
            )
        return dataset
    if isinstance(dataset, str):
        matches = [i for i, d in enumerate(data_list) if dataset in _dataset_names(d)]
        if not matches:
            available = ', '.join(f'{i}: {_dataset_label(d, i)}' for i, d in enumerate(data_list))
            raise ValueError(
                f'No dataset named {dataset!r} in {filename}. Available datasets: {available}.'
            )
        if len(matches) > 1:
            raise ValueError(
                f'Dataset name {dataset!r} is ambiguous in {filename}: matches datasets '
                f'{matches} ({", ".join(_dataset_label(data_list[i], i) for i in matches)}). '
                'Select by 0-based index instead.'
            )
        return matches[0]
    raise TypeError(f'dataset must be an int index or a str name, got {type(dataset).__name__}.')

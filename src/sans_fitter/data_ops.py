"""
Dataset arithmetic for SANS data (issue #45).

Notebook-friendly functions to add, subtract, multiply and divide datasets —
similar to SasView's *Data Operation* utility. Typical uses:

- ``subtract(sample, background)`` — empty-cell / solvent subtraction
- ``multiply(data, 2.0)`` or ``divide(data, transmission)`` — rescaling
- ``subtract(data, 0.05)`` — flat background subtraction

The arithmetic itself (including error propagation, e.g.
``dy = sqrt(dy_a² + dy_b²)`` for addition/subtraction) is delegated to
sasdata's ``Data1D`` operators; this module wraps them so the results are
fit-ready (``qmin``/``qmax``/``mask`` set), carry provenance metadata, and
fail with actionable messages.

Example:
    >>> from sans_fitter import SANSFitter, data_ops
    >>> sample = data_ops.load('sample.csv')
    >>> background = data_ops.load('empty_cell.csv')
    >>> result = data_ops.subtract(sample, background)
    >>> result = data_ops.multiply(result, 0.85)
    >>> fitter = SANSFitter()
    >>> fitter.set_data(result)
    >>> fitter.set_model('sphere')
    >>> fitter.fit()

Limitations:
    - Both datasets must share the same Q grid (sasdata requires x-values to
      match within 1% relative tolerance). Interpolation onto a common grid is
      not yet supported.
    - Resolution (``dx``/``dxl``/``dxw``) propagation through arithmetic is
      **not validated** — sasdata combines ``dx`` in an RMS-like way and skips
      slit-smearing widths. A warning is emitted when resolution data is
      present on any operand; treat resolution on results with care,
      especially for slit-smeared data.
"""

import numbers
import operator
import warnings
from typing import Any

import numpy as np
from sasdata.dataloader.data_info import Data1D, Data2D, Process

from .data_loader import _has_real_data, load_sans_data, normalize_sans_data

__all__ = ['load', 'add', 'subtract', 'multiply', 'divide']

Operand = Data1D | numbers.Number

_OPERATORS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
}


def load(filename: str) -> Data1D:
    """Load a SANS dataset from a file and return a fit-ready ``Data1D``.

    This is the canonical standalone loader: ``SANSFitter.load_data()`` uses
    the same implementation, so datasets loaded here behave identically to
    those loaded through the fitter. Supports CSV, XML and HDF5 formats via
    sasdata (columnar text files are read in the order Q, I, dI, dQ).

    Args:
        filename: Path to the data file.

    Returns:
        A ``Data1D`` object with ``qmin``/``qmax``/``mask`` set.

    Raises:
        ValueError: If the file cannot be loaded or contains no data.
    """
    return load_sans_data(filename)


def add(a: Data1D, b: Operand) -> Data1D:
    """Return ``a + b`` as a new fit-ready dataset.

    ``b`` may be a ``Data1D`` on the same Q grid as ``a`` (intensities are
    added point-wise, uncertainties combine as ``dy = sqrt(dy_a² + dy_b²)``)
    or a scalar (``y' = y + b``; ``dy`` and the Q grid are unchanged).
    """
    return _operation('+', a, b)


def subtract(a: Data1D, b: Operand) -> Data1D:
    """Return ``a − b`` as a new fit-ready dataset. Order matters.

    Typical use: ``subtract(sample, background)`` for empty-cell / solvent
    subtraction. ``b`` may be a ``Data1D`` on the same Q grid as ``a``
    (uncertainties combine as ``dy = sqrt(dy_a² + dy_b²)``) or a scalar for
    flat background subtraction (``y' = y − b``; ``dy`` and the Q grid are
    unchanged).
    """
    return _operation('-', a, b)


def multiply(a: Data1D, b: Operand) -> Data1D:
    """Return ``a × b`` as a new fit-ready dataset.

    ``b`` may be a ``Data1D`` on the same Q grid as ``a`` (relative
    uncertainties combine in quadrature) or a scalar (``y' = b·y``,
    ``dy' = b·dy``; the Q grid is unchanged). Typical scalar use: rescaling
    to absolute units.
    """
    return _operation('*', a, b)


def divide(a: Data1D, b: Operand) -> Data1D:
    """Return ``a / b`` as a new fit-ready dataset. Order matters.

    ``b`` may be a ``Data1D`` on the same Q grid as ``a`` (relative
    uncertainties combine in quadrature) or a scalar (``y' = y/b``,
    ``dy' = dy/b``; the Q grid is unchanged). Typical scalar use: dividing by
    a transmission factor.
    """
    return _operation('/', a, b)


def _dataset_label(data: Any) -> str:
    """Short human-readable name for a dataset or scalar, for messages."""
    if isinstance(data, numbers.Number):
        return f'{data:g}'
    for attr in ('filename', 'title', 'name'):
        value = getattr(data, attr, None)
        if value:
            return str(value)
    return 'dataset'


def _validate_operand(obj: Any, position: str, allow_scalar: bool) -> None:
    """Check that an operand is a usable Data1D (or scalar, if allowed)."""
    if isinstance(obj, numbers.Number) and not isinstance(obj, bool):
        if allow_scalar:
            return
        raise TypeError(f'The {position} operand must be a Data1D dataset, got a scalar.')
    if isinstance(obj, Data2D) or getattr(obj, 'qx_data', None) is not None:
        raise TypeError(
            f'The {position} operand is 2D data; dataset arithmetic supports 1D data only.'
        )
    if not isinstance(obj, Data1D):
        raise TypeError(
            f'The {position} operand must be a Data1D dataset or a number, '
            f'got {type(obj).__name__}.'
        )
    x = getattr(obj, 'x', None)
    y = getattr(obj, 'y', None)
    if x is None or y is None or np.asarray(x).size == 0 or np.asarray(y).size == 0:
        raise ValueError(
            f'The {position} operand ({_dataset_label(obj)}) has no data: '
            'x and y must be populated.'
        )
    if not _has_real_data(getattr(obj, 'dy', None)):
        warnings.warn(
            f'The {position} operand ({_dataset_label(obj)}) has no intensity '
            'uncertainties (dI); they are treated as zero in error propagation.',
            stacklevel=4,
        )


def _has_resolution(data: Any) -> bool:
    """True when a dataset carries any resolution information."""
    if isinstance(data, numbers.Number):
        return False
    return any(_has_real_data(getattr(data, attr, None)) for attr in ('dx', 'dxl', 'dxw'))


def _masked_count(data: Any) -> int:
    """Number of NaN intensity points in a dataset (0 for scalars)."""
    if isinstance(data, numbers.Number):
        return 0
    return int(np.sum(np.isnan(np.asarray(data.y, dtype=float))))


def _operation(op_symbol: str, a: Data1D, b: Operand) -> Data1D:
    """Validate operands, run the sasdata operator and finalize the result."""
    _validate_operand(a, 'first', allow_scalar=False)
    _validate_operand(b, 'second', allow_scalar=True)

    try:
        result = _OPERATORS[op_symbol](a, b)
    except ValueError as e:
        if isinstance(b, Data1D):
            raise ValueError(
                f'Cannot compute {_dataset_label(a)} {op_symbol} {_dataset_label(b)}: '
                f'{e}. The datasets must share the same Q grid '
                f'(x-values matching within 1%). '
                f'{_dataset_label(a)}: {np.asarray(a.x).size} points, '
                f'Q = [{np.min(a.x):g}, {np.max(a.x):g}]; '
                f'{_dataset_label(b)}: {np.asarray(b.x).size} points, '
                f'Q = [{np.min(b.x):g}, {np.max(b.x):g}]. '
                'Interpolation onto a common grid is not yet supported — '
                'rebin the data onto matching Q values first.'
            ) from e
        raise

    operation = f'{_dataset_label(a)} {op_symbol} {_dataset_label(b)}'
    return _finalize(result, operation, a, b)


def _finalize(result: Data1D, operation: str, a: Data1D, b: Operand) -> Data1D:
    """Make an arithmetic result fit-ready and record provenance."""
    # The upstream operators clone metadata from the left operand only and do
    # not set qmin/qmax/mask, which both fit engines require.
    result.qmin = None
    result.qmax = None
    result.mask = None
    normalize_sans_data(result)

    result.title = operation
    result.filename = operation
    process = Process()
    process.name = 'sans_fitter.data_ops'
    process.description = f'Dataset arithmetic: {operation}'
    if getattr(result, 'process', None) is None:
        result.process = []
    result.process.append(process)

    if not _has_real_data(getattr(result, 'dy', None)):
        warnings.warn(
            f"Result of '{operation}' has no intensity uncertainties (dI); "
            'fits will be unweighted.',
            stacklevel=3,
        )

    if _has_resolution(a) or _has_resolution(b):
        warnings.warn(
            'Resolution data (dQ) is present: propagation of resolution through '
            'dataset arithmetic is not validated (sasdata combines dx in an '
            'RMS-like way and ignores slit-smearing widths). Check the result '
            'before relying on smeared fits.',
            stacklevel=3,
        )
        for attr in ('dx', 'dxl', 'dxw'):
            values = getattr(result, attr, None)
            if _has_real_data(values) and np.any(np.nan_to_num(np.asarray(values)) < 0):
                warnings.warn(
                    f"Result of '{operation}' has negative {attr} values — the "
                    'propagated resolution is not physical.',
                    stacklevel=3,
                )

    masked_after = _masked_count(result)
    if masked_after > _masked_count(a):
        warnings.warn(
            f"Result of '{operation}' has {masked_after} masked (NaN) intensity "
            f'points (first operand had {_masked_count(a)}); they are excluded '
            'from fits.',
            stacklevel=3,
        )

    return result

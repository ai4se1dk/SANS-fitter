"""Explicit data weighting for the least-squares objective (issue #72, item 3).

Every engine minimises ``sum(((I - I_model) / sigma) ** 2)`` over the fitted
points. Which ``sigma`` divides each residual used to be implicit: the file's
``dI`` column, with bumps refusing data that had none and the scipy engine
silently unit-weighting it. This module makes it a stated choice —
:class:`WeightingSetting` records the choice, :func:`validate_weighting`
checks it, and :func:`apply_weighting` writes it onto the ``dy`` column of
the evaluation copy that sasmodels is handed.

Why ``dy``: sasmodels reads ``sigma`` from ``data.dy[index]`` in
``direct_model.DataMixin._interpret_data``, which both ``DirectModel`` and
``bumps_model.Experiment`` inherit. ``Experiment.residuals()`` is
``(theory - Iq) / dIq`` and the scipy engine reads ``calculator.dIq``, so
writing the column covers both engines, DREAM and the theory preview with no
engine changes — the same argument that put resolution in columns (#75).

The four modes and their formulas are SasView's Fit Page options, taken from
``sas.qtgui.Perspectives.Fitting.FittingUtilities.getWeight``::

    'none' -> ones_like(I)       "No Weighting"
    'dI'   -> dI                 "Use dI Data"
    'sqrt' -> sqrt(abs(I))       "Use |sqrt(I Data)|"
    'abs'  -> abs(I)             "Use |I Data|"

Unlike SasView, a zero sigma is not divided by: the fit guard in
``SANSFitter`` catches it before the engine runs. This module never raises on
zeros, because preview paths share it and must still draw a curve.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..console import SIGMA, SQRT

#: The four weighting modes, mirroring SasView's Fit Page.
WEIGHTING_MODES = ('dI', 'sqrt', 'abs', 'none')

#: Mode used when nothing has been stated: the dataset's own dI column.
DEFAULT_WEIGHTING_MODE = 'dI'


@dataclass(frozen=True, slots=True)
class WeightingSetting:
    """An immutable record of the active weighting choice.

    Attributes:
        mode: One of :data:`WEIGHTING_MODES`.
    """

    mode: str = DEFAULT_WEIGHTING_MODE

    def as_dict(self) -> dict[str, Any]:
        """Return the setting as a plain dict.

        A new dict on every call, so a caller cannot reach back into fitter
        state through the value returned by ``get_weighting()``. A dict rather
        than the bare mode, for symmetry with ``get_resolution()`` and so that
        a future mode with extra fields does not change the return type.
        """
        return {'mode': self.mode}

    def describe(self, *, ascii_only: bool = False) -> str:
        """Return a one-line human-readable description.

        Args:
            ascii_only: Force plain ASCII. Pass ``True`` for file content (CSV
                header, HTML report), which must not depend on what the
                console can encode; leave it ``False`` for log messages.
        """
        sigma = 'sigma' if ascii_only else SIGMA
        if self.mode == 'dI':
            return "dI (the dataset's own dI column)"
        if self.mode == 'sqrt':
            root = 'sqrt(|I|)' if ascii_only else f'{SQRT}|I|'
            return f'sqrt ({sigma} = {root})'
        if self.mode == 'abs':
            return f'abs ({sigma} = |I|)'
        return f'none (unweighted, {sigma} = 1)'

    def note(self, *, ascii_only: bool = False) -> str:
        """Return the short label an engine records as ``weighting_note``.

        ``'unweighted'`` rather than ``'1'`` for mode ``'none'``: the note is
        user-facing in the fit report, the CSV header and the saved result,
        where a bare ``1`` reads like a typo.

        Args:
            ascii_only: Force plain ASCII. The note stored on a fit result
                ends up in files, so the fitter always passes ``True``.
        """
        if self.mode == 'sqrt':
            return 'sqrt(|I|)' if ascii_only else f'{SQRT}|I|'
        if self.mode == 'abs':
            return '|I|'
        if self.mode == 'none':
            return 'unweighted'
        return 'dI'


def validate_weighting(mode: Any) -> WeightingSetting:
    """Validate a weighting choice and return it as a :class:`WeightingSetting`.

    Matching is exact and case-sensitive, like every other mode string in the
    package; the error lists the accepted spellings.

    Args:
        mode: One of ``'dI'``, ``'sqrt'``, ``'abs'``, ``'none'``.

    Returns:
        The validated, immutable setting.

    Raises:
        ValueError: If the mode is not one of the four (including ``''``,
            ``None`` and case variants such as ``'di'``).
    """
    if not isinstance(mode, str) or mode not in WEIGHTING_MODES:
        raise ValueError(
            f'Unknown weighting mode {mode!r}. Use one of: '
            f'{", ".join(repr(name) for name in WEIGHTING_MODES)}.'
        )
    return WeightingSetting(mode=mode)


def weighting_sigma(intensity: Any, mode: str) -> np.ndarray:
    """Return the per-point sigma *mode* assigns to *intensity*.

    Full length and element-wise, so a NaN intensity gives a NaN sigma — at
    points the fit index already excludes. Not defined for ``'dI'``, whose
    sigma is the data's own column.
    """
    values = np.asarray(intensity, dtype=float)
    if mode == 'sqrt':
        return np.sqrt(np.abs(values))
    if mode == 'abs':
        return np.abs(values)
    if mode == 'none':
        return np.ones_like(values)
    raise ValueError(f'Weighting mode {mode!r} does not derive sigma from the intensity.')


def apply_weighting(data: Any, setting: WeightingSetting, *, copy: bool = True) -> Any:
    """Return *data* with the ``dy`` column *setting* implies.

    Args:
        data: The dataset to weight.
        setting: The validated weighting choice.
        copy: Deep-copy *data* first (the default), so the caller's dataset is
            never modified. ``SANSFitter._evaluation_data`` passes ``False``
            because it hands over the fresh copy ``apply_resolution`` just
            made, which nobody else holds.

    Returns:
        A ``Data1D`` whose ``dy`` is the file's own column under ``'dI'``, or
        the full-length sigma of the mode otherwise. SESANS data is returned
        unchanged under every mode: its ``y`` is a polarisation, not an
        intensity, so ``sqrt(|y|)`` or ``|y|`` has no meaning as a sigma (the
        fit guard refuses that combination).
    """
    weighted = deepcopy(data) if copy else data
    if setting.mode == 'dI' or getattr(weighted, 'isSesans', False):
        return weighted
    weighted.dy = weighting_sigma(weighted.y, setting.mode)
    return weighted

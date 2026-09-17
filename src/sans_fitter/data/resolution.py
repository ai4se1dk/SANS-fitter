"""Explicit resolution (smearing) control for model evaluation (issue #75).

Resolution smearing used to be *implicit*: whatever ``dx`` the loaded file
happened to carry is what every engine smeared with. This module makes it a
stated choice — :class:`ResolutionSetting` records the choice,
:func:`validate_resolution` checks it, and :func:`apply_resolution` writes it
onto a **copy** of the dataset that is handed to sasmodels.

Why a copy, and why columns rather than a resolution object: sasmodels picks
its resolution class from the dataset's columns in
``direct_model.DataMixin._interpret_data``, which both ``DirectModel`` and
``bumps_model.Experiment`` inherit. Setting the columns is therefore the one
mechanism that covers both fitting engines and DREAM with no engine changes.

The dispatch it performs on 1D (Iq) data is::

    if data.dx is not None:
        if (dx[index] > 0).any():   -> Pinhole1D(q, dx)
        else:                       -> Perfect1D(q)
    elif data.dxl is not None or data.dxw is not None:
        -> Slit1D(q, q_length=dxl, q_width=dxw)
    else:
        -> Perfect1D(q)

Two consequences drive the rules in :func:`apply_resolution`:

1. The predicate is ``dx is not None``, **not** "dx has data". sasdata
   zero-fills an absent ``dQ`` column, so almost every loaded dataset has a
   non-None ``dx``. Slit smearing is therefore unreachable unless ``dx`` is
   explicitly set to ``None`` first.
2. ``dx`` has absolute priority over the slit columns. A file carrying only
   ``dxl``/``dxw`` (the USANS case) is silently fitted *unsmeared* unless
   ``dx`` is cleared.

Widths follow sasmodels' own convention: ``dx`` is the Gaussian 1-sigma width
sigma_q (not FWHM), and ``dxl``/``dxw`` are absolute slit lengths/widths in
1/Angstrom.
"""

import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..console import INVERSE_ANGSTROM, SIGMA
from .loader import get_fit_index, has_real_data

#: The four resolution modes, mirroring SasView's Fit Page
#: (*None* / *Use dQ Data* / *Custom Pinhole* / *Custom Slit*).
RESOLUTION_MODES = ('data', 'none', 'pinhole', 'slit')

#: Mode used when nothing has been stated: the dataset's own columns.
DEFAULT_RESOLUTION_MODE = 'data'


@dataclass(frozen=True, slots=True)
class ResolutionSetting:
    """An immutable record of the active resolution choice.

    Attributes:
        mode: One of :data:`RESOLUTION_MODES`.
        dq_over_q: Relative pinhole width sigma_q/q (Gaussian 1-sigma), set
            only for ``'pinhole'``.
        slit_length: Slit length along q in 1/Angstrom (``dxl``), set only for
            ``'slit'``.
        slit_width: Slit width perpendicular to q in 1/Angstrom (``dxw``), set
            only for ``'slit'``.
    """

    mode: str = DEFAULT_RESOLUTION_MODE
    dq_over_q: float | None = None
    slit_length: float | None = None
    slit_width: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the setting as a plain dict with all four keys always present.

        A new dict on every call, so a caller cannot reach back into fitter
        state through the value returned by ``get_resolution()``.
        """
        return {
            'mode': self.mode,
            'dq_over_q': self.dq_over_q,
            'slit_length': self.slit_length,
            'slit_width': self.slit_width,
        }

    def describe(self, *, ascii_only: bool = False) -> str:
        """Return a one-line human-readable description.

        Args:
            ascii_only: Force plain ASCII. The glyph constants pick their
                spelling from what ``sys.stdout`` can encode, which is the
                wrong authority for anything written to a file — a UTF-8
                console on a cp1252 filesystem would produce a sigma the
                export cannot store. Pass ``True`` for file content, leave it
                ``False`` for log messages.
        """
        sigma = 'sigma' if ascii_only else SIGMA
        inverse_angstrom = '1/Ang' if ascii_only else INVERSE_ANGSTROM

        if self.mode == 'data':
            return "data (the dataset's own dQ columns)"
        if self.mode == 'none':
            return 'none (perfect resolution)'
        if self.mode == 'pinhole':
            return f'pinhole ({sigma}_q/q = {self.dq_over_q:g})'
        parts = []
        if self.slit_length:
            parts.append(f'length = {self.slit_length:g}')
        if self.slit_width:
            parts.append(f'width = {self.slit_width:g}')
        return f'slit ({", ".join(parts)} {inverse_angstrom})'


def _reject_unused(mode: str, **values: float | None) -> None:
    """Raise when arguments that do not belong to *mode* were supplied."""
    supplied = [name for name, value in values.items() if value is not None]
    if supplied:
        raise ValueError(
            f'{", ".join(sorted(supplied))} '
            f'{"is" if len(supplied) == 1 else "are"} not accepted by '
            f"resolution mode '{mode}'."
        )


def _require_finite(name: str, value: float, *, positive: bool) -> float:
    """Validate a single width argument and return it as a float."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be a number, got {value!r}.') from None
    if not np.isfinite(number):
        raise ValueError(f'{name} must be finite, got {value!r}.')
    if positive and number <= 0:
        raise ValueError(
            f'{name} must be greater than zero, got {number!r}. '
            "A width of zero is set_resolution('none'), not a degenerate pinhole."
        )
    if not positive and number < 0:
        raise ValueError(f'{name} must not be negative, got {number!r}.')
    return number


def validate_resolution(
    mode: str,
    dq_over_q: float | None = None,
    slit_length: float | None = None,
    slit_width: float | None = None,
) -> ResolutionSetting:
    """Validate a resolution choice and return it as a :class:`ResolutionSetting`.

    Every check runs before anything is constructed, so a caller can treat a
    raised ``ValueError`` as "nothing changed" — the same discipline as
    :func:`~sans_fitter.modeling.structure_factor.validate_radius_effective_mode`.

    Args:
        mode: One of ``'data'``, ``'none'``, ``'pinhole'``, ``'slit'``.
        dq_over_q: Relative pinhole width sigma_q/q; required by ``'pinhole'``
            and rejected by every other mode.
        slit_length: Slit length along q (1/Angstrom); required by ``'slit'``
            and rejected by every other mode.
        slit_width: Slit width perpendicular to q (1/Angstrom); optional under
            ``'slit'``, rejected elsewhere.

    Returns:
        The validated, immutable setting.

    Raises:
        ValueError: If the mode is unknown, an argument does not belong to the
            mode, a required argument is missing, or a width is non-finite,
            negative, or degenerate (a pinhole or a slit length of zero).
    """
    if mode not in RESOLUTION_MODES:
        raise ValueError(
            f"Unknown resolution mode '{mode}'. Use one of: "
            f'{", ".join(repr(name) for name in RESOLUTION_MODES)}.'
        )

    if mode in ('data', 'none'):
        _reject_unused(mode, dq_over_q=dq_over_q, slit_length=slit_length, slit_width=slit_width)
        return ResolutionSetting(mode=mode)

    if mode == 'pinhole':
        _reject_unused(mode, slit_length=slit_length, slit_width=slit_width)
        if dq_over_q is None:
            raise ValueError(
                "Resolution mode 'pinhole' requires dq_over_q, the relative "
                f'pinhole width {SIGMA}_q/q (Gaussian 1-sigma, not FWHM), '
                "e.g. set_resolution('pinhole', dq_over_q=0.10)."
            )
        return ResolutionSetting(
            mode='pinhole',
            dq_over_q=_require_finite('dq_over_q', dq_over_q, positive=True),
        )

    # mode == 'slit'
    _reject_unused('slit', dq_over_q=dq_over_q)
    if slit_length is None:
        raise ValueError(
            "Resolution mode 'slit' requires slit_length, the slit dimension "
            f'along q in {INVERSE_ANGSTROM} '
            "(e.g. set_resolution('slit', slit_length=0.05)). "
            'sasmodels does not implement slit smearing from a width alone: '
            'a slit with no length raises NotImplementedError inside its '
            'weight-matrix construction, so slit_width is an optional '
            'refinement of a real slit rather than a slit on its own.'
        )
    length = _require_finite('slit_length', slit_length, positive=True)
    width = (
        None if slit_width is None else _require_finite('slit_width', slit_width, positive=False)
    )
    return ResolutionSetting(mode='slit', slit_length=length, slit_width=width)


def _clear_columns(data: Any) -> None:
    """Null every resolution column so sasmodels falls through to Perfect1D.

    Nulling rather than zeroing: ``_interpret_data`` branches on
    ``dx is not None``, so an all-zero ``dx`` still takes the pinhole branch
    and only *happens* to degrade to ``Perfect1D`` when every fitted point is
    zero. Zeroing also leaves the copy claiming a dQ column the file does not
    have.
    """
    data.dx = None
    data.dxl = None
    data.dxw = None


def _constant_column(value: float | None, n_points: int) -> np.ndarray | None:
    """Broadcast a constant slit dimension over every q point, or None if unset."""
    if not value:
        return None
    return np.full(n_points, float(value))


def _check_slit_columns(data: Any, source: str) -> None:
    """Guard the two slit geometries sasmodels cannot evaluate.

    Both would otherwise surface from deep inside ``sasmodels.resolution`` as
    errors that name nothing the caller recognises, and both are checked over
    the fitted points only — the same subset ``_interpret_data`` passes on.

    1. A slit **width with no length** hits an explicit
       ``NotImplementedError("We do not yet handle the q_length=0.0 case for
       all q_width.")`` in ``slit_resolution``.
    2. A width exactly equal to the smallest fitted Q puts the floor of
       ``Slit1D``'s extended grid at zero, and the extrapolation takes
       ``log(0)`` — an ``OverflowError`` about converting infinity to an
       integer. A larger width is fine (the floor is clamped positive) and a
       smaller one is ordinary; only exact equality breaks.
    """
    index = get_fit_index(data)
    q_values = np.asarray(data.x, dtype=float)[index]
    if q_values.size == 0:
        return

    def column(name: str) -> np.ndarray:
        values = getattr(data, name, None)
        if values is None:
            return np.zeros_like(q_values)
        return np.nan_to_num(np.asarray(values, dtype=float))[index]

    length, width = column('dxl'), column('dxw')

    if np.any((length == 0) & (width != 0)):
        raise ValueError(
            f'{source} describes a slit width with no slit length, which '
            'sasmodels does not implement (it smears along the slit length, '
            'and a width alone has nothing to integrate over). Supply a slit '
            "length, or use set_resolution('none') to fit unsmeared."
        )

    if np.any(width) and float(np.min(q_values - width)) == 0.0:
        raise ValueError(
            f'{source} has a slit width exactly equal to the smallest fitted '
            f'Q ({float(np.min(q_values)):g}), which sasmodels cannot build a '
            'resolution grid for. Use a slightly different width, or trim the '
            'lowest Q point with set_q_range().'
        )


def apply_resolution(data: Any, setting: ResolutionSetting, *, warn: bool = True) -> Any:
    """Return a copy of *data* carrying the resolution columns *setting* implies.

    The input dataset is never modified: the copy is a ``deepcopy``, because
    ``Data1D`` holds its columns as numpy arrays by reference and a shallow
    copy would rewrite the caller's ``dx`` in place. ``qmin``/``qmax``/``mask``
    survive the copy, so ``set_q_range()`` and the NaN mask still select the
    same points — ``_interpret_data`` builds its fit index from exactly those.

    Args:
        data: The user's dataset. Returned untouched (as a copy) when it is
            SESANS data, which sasmodels interprets through a different branch.
        setting: The validated resolution choice.
        warn: Whether to emit the advisory warnings for mode ``'data'``. Set
            ``False`` on evaluation paths that follow a fit which already
            warned, so one fit does not warn repeatedly.

    Returns:
        A ``Data1D`` copy whose ``dx``/``dxl``/``dxw`` select the intended
        sasmodels resolution class.
    """
    evaluation_data = deepcopy(data)

    # SESANS data is interpreted through a different branch of
    # _interpret_data, where these columns mean something else entirely.
    if getattr(evaluation_data, 'isSesans', False):
        return evaluation_data

    q_values = np.asarray(evaluation_data.x)

    if setting.mode == 'none':
        _clear_columns(evaluation_data)
        return evaluation_data

    if setting.mode == 'pinhole':
        # dx = sigma_q = (sigma_q/q) * q, the same convention as the file's own
        # dQ column and as examples.simulate(dq=...).
        evaluation_data.dx = q_values * float(setting.dq_over_q)
        evaluation_data.dxl = None
        evaluation_data.dxw = None
        return evaluation_data

    if setting.mode == 'slit':
        # dx must go first: while it is non-None the slit branch is unreachable.
        evaluation_data.dx = None
        evaluation_data.dxl = _constant_column(setting.slit_length, q_values.size)
        evaluation_data.dxw = _constant_column(setting.slit_width, q_values.size)
        _check_slit_columns(evaluation_data, 'The requested slit resolution')
        return evaluation_data

    # mode == 'data': use whatever the dataset actually carries.
    has_pinhole = has_real_data(getattr(evaluation_data, 'dx', None))
    has_slit = has_real_data(getattr(evaluation_data, 'dxl', None)) or has_real_data(
        getattr(evaluation_data, 'dxw', None)
    )

    if has_pinhole:
        if has_slit and warn:
            warnings.warn(
                'Dataset carries both a dQ (pinhole) column and slit columns '
                '(dxl/dxw). sasmodels gives dQ priority, so the slit columns '
                "will be ignored. Use set_resolution('slit', ...) to smear "
                'with slit geometry instead.',
                stacklevel=2,
            )
        return evaluation_data

    if has_slit:
        # Clearing the zero-filled dx is what makes the slit branch reachable;
        # without it a slit-only dataset fits unsmeared and says nothing.
        evaluation_data.dx = None
        _check_slit_columns(evaluation_data, "The dataset's slit columns (dxl/dxw)")
        return evaluation_data

    if warn:
        warnings.warn(
            "Resolution mode 'data' was requested but the dataset carries no "
            'resolution columns (dQ, dxl, dxw). The model will be evaluated '
            "unsmeared. Use set_resolution('pinhole', dq_over_q=...) to state "
            "a width, or set_resolution('none') to silence this.",
            stacklevel=2,
        )
    _clear_columns(evaluation_data)
    return evaluation_data

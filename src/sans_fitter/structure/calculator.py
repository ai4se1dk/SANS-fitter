"""The Generic Scattering Calculator: I(q) of a real-space structure."""

import html
import os
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sasdata.dataloader.data_info import Data1D, Process

from ..console import ANGSTROM, CUBIC_ANGSTROM, INVERSE_ANGSTROM, OK, SLD_UNIT, logger
from ..data.loader import normalize_sans_data, validate_q_grid
from ..persistence import _package_version
from .debye import LATTICE_MAX_CELLS, debye_lattice, debye_pairs, lattice_cells
from .model import AVOGADRO, PROTEIN_SPECIFIC_VOLUME, Structure
from .readers import load

#: SasView's default 1D grid: 30 log-spaced points up to 0.3 1/Å.
DEFAULT_Q_GRID = {'qmin': 3e-4, 'qmax': 0.3, 'npoints': 30, 'spacing': 'log'}

#: Above this many active points the pair loop warns about run time.
LARGE_PAIR_LOOP = 20_000

#: A population is "near-matched" when its contrast is this small relative to the largest.
NEAR_MATCH = 1e-6

#: The point-per-cell approximation is flagged above q * cell size = 1.
CELL_SIZE_LIMIT = 1.0

_UNSET = object()


def _q_grid(qmin: float, qmax: float, npoints: int, spacing: str) -> np.ndarray:
    if spacing not in ('log', 'linear'):
        raise ValueError(f"spacing must be 'log' or 'linear', got {spacing!r}.")
    if int(npoints) != npoints or npoints < 2:
        raise ValueError(f'npoints must be an integer of at least 2, got {npoints!r}.')
    if not 0 < qmin < qmax:
        raise ValueError(f'Need 0 < qmin < qmax, got qmin={qmin}, qmax={qmax}.')
    if spacing == 'log':
        return validate_q_grid(np.geomspace(qmin, qmax, int(npoints)))
    return validate_q_grid(np.linspace(qmin, qmax, int(npoints)))


@dataclass(frozen=True, eq=False)
class ScatteringProfile:
    """A computed I(q) with everything needed to know how it was made.

    Attributes:
        q: Momentum transfer, 1/Å.
        intensity: I(q), 1/cm.
        normalisation_volume: The volume V_norm the Debye sum was divided by,
            Å³; nan when nothing scatters and no volume was set.
        normalisation: ``'active-volume'`` (the default) or ``'explicit'``.
        n_zero_contrast: Points whose SLD equals the solvent's.
        method: Kernel used: ``'lattice'``, ``'pairs'`` or ``'none'``.
        notes: Warnings raised while reading and calculating.
        elapsed: Calculation time, s.
    """

    q: np.ndarray
    intensity: np.ndarray
    normalisation_volume: float
    normalisation: str
    n_zero_contrast: int
    method: str
    notes: tuple[str, ...]
    elapsed: float
    _params: dict = field(repr=False)
    _structure_summary: dict = field(repr=False)

    def __post_init__(self) -> None:
        for name in ('q', 'intensity'):
            array = np.array(getattr(self, name), dtype=float, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def params(self) -> dict[str, Any]:
        """scale, background, sld_solvent (10⁻⁶ Å⁻²) and normalisation_volume, as set."""
        return dict(self._params)

    @property
    def structure_summary(self) -> dict[str, Any]:
        """Source, format, kind, size and reader notes of the structure."""
        return dict(self._structure_summary)

    def _process_text(self) -> str:
        """Provenance in ASCII, for file content."""
        s, p = self._structure_summary, self._params
        text = (
            f'Generic Scattering Calculator: {s["basename"]} ({s["format"]}, {s["kind"]}, '
            f'{s["n_points"]} points, sha256 {str(s["source_sha256"])[:12]}); '
            f'scale={p["scale"]:g}, background={p["background"]:g} 1/cm, '
            f'sld_solvent={p["sld_solvent"]:g} 1e-6/Ang^2, '
            f'V_norm={self.normalisation_volume:.6g} Ang^3 ({self.normalisation}), '
            f'kernel={self.method}'
        )
        if self.notes:
            text += '; notes: ' + ' | '.join(self.notes)
        return text.encode('ascii', 'replace').decode('ascii')

    def to_data1d(self, *, dI=None) -> Data1D:
        """Return the curve as a fit-ready sasdata ``Data1D``.

        The curve is theory, so it carries no dQ and, unless *dI* is given, no
        dI. Fitting it then needs an explicit ``set_weighting`` (for example
        ``'abs'``) and ``set_resolution('none')``.

        Args:
            dI: Optional uncertainties: finite, non-negative, one per q.
        """
        dy = None
        if dI is not None:
            dy = np.array(dI, dtype=float, copy=True)
            if dy.shape != self.q.shape or not np.all(np.isfinite(dy)) or np.any(dy < 0):
                raise ValueError('dI must be finite, non-negative, and have one value per q.')
        data = Data1D(x=self.q.copy(), y=self.intensity.copy(), dy=dy)
        data.title = f'GSC {self._structure_summary["basename"]}'
        process = Process()
        process.name = 'sans_fitter.structure'
        process.description = self._process_text()
        data.process.append(process)
        return normalize_sans_data(data)

    def save_csv(self, path) -> None:
        """Write q and I(q) as CSV, with the settings in an ASCII header."""
        header = (
            f'{self._process_text()}\n'
            f'source: {self._structure_summary["source"]}\n'
            'q (1/Ang), I (1/cm)'
        )
        np.savetxt(path, np.column_stack([self.q, self.intensity]), delimiter=',', header=header)

    def _lines(self) -> list[tuple[str, str]]:
        s, p = self._structure_summary, self._params
        rows = [
            ('Structure', f'{s["basename"]}: {s["n_points"]} {s["kind"]}'),
            ('q', f'{self.q.size} points, {self.q[0]:.4g} to {self.q[-1]:.4g} {INVERSE_ANGSTROM}'),
            ('I(q)', f'{np.nanmin(self.intensity):.4g} to {np.nanmax(self.intensity):.4g} 1/cm'),
            ('Solvent SLD', f'{p["sld_solvent"]:g} {SLD_UNIT}'),
            ('Scale, background', f'{p["scale"]:g}, {p["background"]:g} 1/cm'),
            (
                'V_norm',
                f'{self.normalisation_volume:.6g} {CUBIC_ANGSTROM} ({self.normalisation}); '
                f'{self.n_zero_contrast} zero-contrast points',
            ),
            ('Kernel', f'{self.method}, {self.elapsed:.3g} s'),
        ]
        rows.extend(('Note', note) for note in self.notes)
        return rows

    def describe(self) -> None:
        """Print how the profile was computed."""
        for label, value in self._lines():
            print(f'  {label:<18} {value}')

    def _repr_html_(self) -> str:
        cells = ''.join(
            f'<tr><th style="text-align:left">{html.escape(k)}</th><td>{html.escape(v)}</td></tr>'
            for k, v in self._lines()
        )
        return f'<table>{cells}</table>'

    def plot(self, *, data=None, log_scale: bool = True, show: bool | None = None):
        """Plot I(q); see :func:`sans_fitter.structure.plotting.plot_profile`."""
        from .plotting import plot_profile

        return plot_profile(self, data=data, log_scale=log_scale, show=show)


class GenericScatteringCalculator:
    """Orientationally averaged I(q) of a structure, by the exact Debye formula.

    The nuclear, 1D part of SasView's Generic Scattering Calculator::

        w_j  = (sld_j - sld_solvent * 1e-6) * volume_j
        I(q) = scale * 1e8 * D(q) / V_norm + background
        D(q) = sum_jk w_j w_k sin(q r_jk) / (q r_jk)

    ``V_norm`` is the volume of the points that scatter (``w != 0``) unless
    ``normalisation_volume`` is set. For space-filling pixels and cells,
    ``scale`` is then a volume fraction, as in sasmodels.

    Example:
        >>> gsc = GenericScatteringCalculator('particle.sld')
        >>> gsc.set_params(sld_solvent=6.34)
        >>> profile = gsc.calculate()
        >>> gsc.plot_profile()
    """

    def __init__(self, structure: Structure | str | os.PathLike | None = None) -> None:
        self._structure: Structure | None = None
        self._profile: ScatteringProfile | None = None
        self._q_args: dict[str, Any] | None = dict(DEFAULT_Q_GRID)
        self._q = _q_grid(**DEFAULT_Q_GRID)
        self._params: dict[str, Any] = {
            'scale': 1.0,
            'background': 0.0,
            'sld_solvent': 0.0,
            'normalisation_volume': 'auto',
        }
        self._reader_options: dict[str, Any] = {}
        if isinstance(structure, Structure):
            self.set_structure(structure)
        elif structure is not None:
            self.load_structure(structure)

    # ------------------------------------------------------------- properties

    @property
    def structure(self) -> Structure | None:
        return self._structure

    @property
    def profile(self) -> ScatteringProfile | None:
        """The result of the last ``calculate()``, cleared by every setter."""
        return self._profile

    @property
    def q(self) -> np.ndarray:
        return self._q.copy()

    # ---------------------------------------------------------------- setters

    def load_structure(self, path, *, include_hetatm: bool = False) -> Structure:
        """Read a PDB, SLD or VTK file and make it the current structure."""
        structure = load(path, include_hetatm=include_hetatm)
        self._install(structure, {'include_hetatm': include_hetatm})
        return structure

    def set_structure(self, structure: Structure) -> None:
        """Use an existing :class:`Structure`, for example one from ``Structure.from_arrays``."""
        if not isinstance(structure, Structure):
            raise TypeError(f'Expected a Structure, got {type(structure).__name__}.')
        self._install(structure, {})

    def _install(self, structure: Structure, options: dict[str, Any]) -> None:
        self._structure = structure
        self._reader_options = options
        self._params['normalisation_volume'] = 'auto'
        self._profile = None
        for note in structure.notes:
            warnings.warn(f'{os.path.basename(structure.source)}: {note}', stacklevel=3)
        rg = structure.radius_of_gyration('sld', sld_solvent=self._params['sld_solvent'])
        detail = (
            f'Rg {rg.value:.3g} {ANGSTROM}' if np.isfinite(rg.value) else f'Rg n/a ({rg.reason})'
        )
        if structure.lattice is not None:
            detail += (
                f', lattice step {[round(s, 6) for s in structure.lattice.step]} {ANGSTROM}, '
                f'pixel volume {structure.volume[0]:.6g} {CUBIC_ANGSTROM}'
            )
        logger.info(
            f'{OK} Loaded {os.path.basename(structure.source)}: {structure.n_points} '
            f'{structure.kind}, {detail}; normalisation volume reset to auto'
        )

    def set_q_grid(
        self, qmin: float = 3e-4, qmax: float = 0.3, npoints: int = 30, *, spacing: str = 'log'
    ) -> None:
        """Build the q grid: *npoints* from *qmin* to *qmax* (1/Å), log or linear.

        Unlike SasView, qmin does not follow qmax; ``set_q_grid(qmax=0.5)``
        keeps qmin = 3e-4.
        """
        args = {'qmin': qmin, 'qmax': qmax, 'npoints': npoints, 'spacing': spacing}
        self._q = _q_grid(**args)
        self._q_args = args
        self._profile = None
        logger.info(
            f'{OK} q grid: {npoints} {spacing} points, {qmin:g} to {qmax:g} {INVERSE_ANGSTROM}'
        )

    def set_q(self, q) -> None:
        """Use explicit q values (1/Å): positive, finite and increasing."""
        self._q = validate_q_grid(q).copy()
        self._q_args = None
        self._profile = None
        logger.info(f'{OK} q grid: {self._q.size} explicit values')

    def set_params(
        self,
        *,
        scale=_UNSET,
        background=_UNSET,
        sld_solvent=_UNSET,
        normalisation_volume=_UNSET,
    ) -> None:
        """Set any of the calculation parameters; the others are kept.

        Args:
            scale: Multiplies I(q); >= 0.
            background: Flat background, 1/cm; >= 0, because a computed curve
                has no subtraction residue to correct.
            sld_solvent: Solvent SLD in 10⁻⁶ Å⁻², as in every sasmodels model
                (D2O is 6.34). SasView's calculator takes Å⁻² instead.
            normalisation_volume: V_norm in Å³, or ``'auto'`` for the volume of
                the points that scatter. SasView calls this field total_volume.
        """
        new = dict(self._params)
        for name, value, lower in (('scale', scale, 0.0), ('background', background, 0.0)):
            if value is not _UNSET:
                value = float(value)
                if not np.isfinite(value) or value < lower:
                    raise ValueError(f'{name} must be finite and >= {lower:g}, got {value}.')
                new[name] = value
        if sld_solvent is not _UNSET:
            value = float(sld_solvent)
            if not np.isfinite(value):
                raise ValueError(f'sld_solvent must be finite, got {value}.')
            if 0 < abs(value) < 1e-3:
                warnings.warn(
                    f'sld_solvent={value:g} is in 10^-6 Ang^-2 and looks like a value in Ang^-2. '
                    f'Did you mean {value * 1e6:g}?',
                    stacklevel=2,
                )
            new['sld_solvent'] = value
        if normalisation_volume is not _UNSET:
            if normalisation_volume != 'auto':
                value = float(normalisation_volume)
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f"normalisation_volume must be > 0 or 'auto', got {value}.")
                normalisation_volume = value
            new['normalisation_volume'] = normalisation_volume
        self._params = new
        self._profile = None
        logger.info(f'{OK} Parameters: {self._describe_params()}')

    def get_params(self) -> dict[str, Any]:
        return dict(self._params)

    def _describe_params(self) -> str:
        p = self._params
        volume = p['normalisation_volume']
        volume = volume if volume == 'auto' else f'{volume:.6g} {CUBIC_ANGSTROM}'
        return (
            f'scale {p["scale"]:g}, background {p["background"]:g} 1/cm, '
            f'sld_solvent {p["sld_solvent"]:g} {SLD_UNIT}, normalisation volume {volume}'
        )

    # ------------------------------------------------------------ calculation

    def _require_structure(self) -> Structure:
        if self._structure is None:
            raise ValueError('No structure loaded. Use load_structure() first.')
        return self._structure

    def calculate(self, *, progress: Callable[[float], bool | None] | None = None):
        """Compute I(q) on the current grid and return a :class:`ScatteringProfile`.

        Args:
            progress: Optional callback for long pair loops. It receives the
                completed fraction; returning True stops the calculation with
                ``CalculationInterrupted``, leaving the previous state intact.
        """
        structure = self._require_structure()
        p = self._params
        q = self._q
        notes = list(structure.notes)
        rho = p['sld_solvent'] * 1e-6
        contrast = structure.sld - rho
        weight = contrast * structure.volume
        active = weight != 0
        n_zero = int(np.sum(~active))
        explicit = p['normalisation_volume'] != 'auto'
        start = time.perf_counter()

        if not active.any():
            self._warn(
                notes,
                f'Every point matches the solvent SLD ({p["sld_solvent"]:g}); I(q) is the background.',
            )
            intensity = np.full(q.size, p['background'])
            v_norm = p['normalisation_volume'] if explicit else float('nan')
            method = 'none'
        else:
            v_norm = (
                p['normalisation_volume'] if explicit else float(structure.volume[active].sum())
            )
            self._check(structure, contrast, active, notes)
            method, debye = self._debye(structure, weight, active, progress, notes)
            intensity = p['scale'] * 1e8 * debye / v_norm + p['background']

        summary = {
            'source': structure.source,
            'basename': os.path.basename(structure.source),
            'source_sha256': structure.source_sha256,
            'format': structure.format,
            'kind': structure.kind,
            'n_points': structure.n_points,
            'reader_options': dict(self._reader_options),
            'notes': list(structure.notes),
        }
        profile = ScatteringProfile(
            q=q,
            intensity=intensity,
            normalisation_volume=float(v_norm),
            normalisation='explicit' if explicit else 'active-volume',
            n_zero_contrast=n_zero,
            method=method,
            notes=tuple(notes),
            elapsed=time.perf_counter() - start,
            _params=dict(p),
            _structure_summary=summary,
        )
        self._profile = profile
        logger.info(
            f'{OK} I(q) at {q.size} q values: {int(active.sum())} scattering points, '
            f'{n_zero} at zero contrast, V_norm {v_norm:.6g} {CUBIC_ANGSTROM} '
            f'({profile.normalisation}), {method} kernel, {profile.elapsed:.3g} s'
        )
        return profile

    @staticmethod
    def _warn(notes: list[str], text: str) -> None:
        warnings.warn(text, stacklevel=4)
        notes.append(text)

    def _check(self, structure: Structure, contrast, active, notes: list[str]) -> None:
        """Warnings that change how the curve should be read."""
        largest = np.abs(contrast).max()
        near = active & (np.abs(contrast) <= NEAR_MATCH * largest)
        if near.any():
            self._warn(
                notes,
                f'{int(near.sum())} points almost match the solvent (|contrast| <= '
                f'{NEAR_MATCH:g} of the largest) but still count towards V_norm. Set '
                'normalisation_volume explicitly for contrast-variation series.',
            )

        size = None
        if structure.kind == 'pixels':
            if structure.lattice is not None:
                size = max(structure.lattice.step)
            else:
                size = float(np.cbrt(structure.volume.max()))
        elif structure.kind == 'cells':
            size = float(structure.cell_diameter.max() / np.sqrt(3))
        if size and self._q[-1] * size > CELL_SIZE_LIMIT:
            self._warn(
                notes,
                f'q_max * cell size = {self._q[-1] * size:.3g} > {CELL_SIZE_LIMIT:g} (cell size '
                f'{size:.4g} Ang): treating each {structure.kind[:-1]} as a point degrades above '
                f'q = {CELL_SIZE_LIMIT / size:.4g} 1/Ang.',
            )

        if structure.kind == 'atoms' and self._params['sld_solvent'] != 0:
            elemental = float(structure.volume[active].sum())
            protein = float(
                structure.masses[active].sum() * PROTEIN_SPECIFIC_VOLUME / AVOGADRO * 1e24
            )
            self._warn(
                notes,
                f'Atomic volumes come from elemental densities: {elemental:.4g} Ang^3 in total, '
                f'{elemental / protein:.2f} times a protein-like estimate ({protein:.4g} Ang^3 at '
                f'{PROTEIN_SPECIFIC_VOLUME} cm^3/g). They are not molecular excluded volumes, so '
                'in a solvent I(q) is only qualitative; there is no hydration shell or H/D '
                'exchange either.',
            )

    def _debye(self, structure, weight, active, progress, notes) -> tuple[str, np.ndarray]:
        lattice = structure.lattice
        if lattice is not None:
            index = lattice.indices[active]
            index = index - index.min(axis=0)
            shape = tuple(index.max(axis=0) + 1)
            if lattice_cells(shape) <= LATTICE_MAX_CELLS:
                grid = np.zeros(shape)
                np.add.at(grid, tuple(index.T), weight[active])
                return 'lattice', debye_lattice(self._q, grid, lattice.step)
        n = int(active.sum())
        if n > LARGE_PAIR_LOOP:
            self._warn(
                notes,
                f'{n} scattering points: the exact pair sum scales as N^2 and may take many '
                'minutes.',
            )
        last = [time.monotonic()]

        def report(fraction: float) -> bool | None:
            if time.monotonic() - last[0] > 5:
                logger.info(f'  Debye sum {fraction:.0%} done')
                last[0] = time.monotonic()
            return progress(fraction) if progress is not None else None

        return 'pairs', debye_pairs(self._q, structure.xyz[active], weight[active], progress=report)

    # ----------------------------------------------------------------- output

    def describe(self) -> None:
        """Print the structure, the q grid and the parameters."""
        structure = self._require_structure()
        structure.describe()
        contrast = structure.sld - self._params['sld_solvent'] * 1e-6
        print(f'  {"Zero contrast":<18} {int(np.sum(contrast == 0))} points at the current solvent')
        print(
            f'  {"q grid":<18} {self._q.size} points, {self._q[0]:.4g} to {self._q[-1]:.4g} {INVERSE_ANGSTROM}'
        )
        print(f'  {"Parameters":<18} {self._describe_params()}')

    def plot_points(self, **kwargs: Any):
        """Draw the structure's points; zero-contrast points use the current solvent."""
        kwargs.setdefault('sld_solvent', self._params['sld_solvent'])
        return self._require_structure().plot_points(**kwargs)

    def plot_profile(self, *, data=None, log_scale: bool = True, show: bool | None = None):
        """Plot the current profile, optionally over measured *data* (a ``Data1D``)."""
        if self._profile is None:
            raise ValueError('No current profile. Call calculate() first.')
        return self._profile.plot(data=data, log_scale=log_scale, show=show)

    # ------------------------------------------------------------ settings

    def get_settings(self) -> dict[str, Any]:
        """A JSON-serialisable record of the source, grid, parameters and versions."""
        structure = self._require_structure()
        return {
            'source': structure.source,
            'basename': os.path.basename(structure.source),
            'source_sha256': structure.source_sha256,
            'reader_options': dict(self._reader_options),
            'q': self._q.tolist(),
            'q_grid': dict(self._q_args) if self._q_args else None,
            'params': dict(self._params),
            'versions': {
                name: _package_version(name) for name in ('sans-fitter', 'numpy', 'periodictable')
            },
        }

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> 'GenericScatteringCalculator':
        """Rebuild a calculator from ``get_settings()``, warning about what changed."""
        if settings['source'] == 'in-memory':
            raise ValueError('These settings describe an in-memory structure; use set_structure().')
        calculator = cls()
        structure = calculator.load_structure(settings['source'], **settings['reader_options'])
        if structure.source_sha256 != settings['source_sha256']:
            warnings.warn(
                f'{settings["source"]} has changed since the settings were saved.', stacklevel=2
            )
        saved = settings['versions'].get('periodictable')
        if saved and saved != _package_version('periodictable'):
            warnings.warn(
                f'periodictable {saved} was used for the saved settings; scattering lengths '
                'and densities may differ in this version.',
                stacklevel=2,
            )
        if settings.get('q_grid'):
            calculator.set_q_grid(**settings['q_grid'])
        else:
            calculator.set_q(settings['q'])
        calculator.set_params(**settings['params'])
        return calculator

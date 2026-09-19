"""
SANS Model Fitter - A flexible template for fitting SANS data with SasModels

This module provides a unified interface for fitting SANS data using different
optimization engines (BUMPS, LMFit) with any model from the SasModels library.
"""

import difflib
import functools
import os
import re
import warnings
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from plotly.graph_objects import Figure

# SasModels and SasData imports
from sasmodels import core
from sasmodels.core import load_model
from sasmodels.direct_model import DirectModel

from . import plotting
from .console import ARROW, CHI_SQUARED, INVERSE_ANGSTROM, OK, logger
from .data.loader import (
    get_fit_index,
    has_real_data,
    load_sans_dataset,
    normalize_sans_data,
)
from .data.provenance import DataSource, fingerprint_arrays
from .data.resolution import ResolutionSetting, apply_resolution, validate_resolution
from .fitting import (
    DEFAULT_DREAM_BURN,
    DEFAULT_DREAM_POP,
    DEFAULT_DREAM_SAMPLES,
    DEFAULT_DREAM_THIN,
    SCIPY_AVAILABLE,
    fit_bumps,
    fit_bumps_dream,
    fit_scipy,
)
from .fitting.base import at_bound, extract_fit_index, pd_is_active, reduced_chisq
from .fitting.theory import (
    build_model_parameters,
    evaluate_theory,
    preview_quality,
    scatter_to_full_length,
    theory_data,
)
from .modeling.parameters import ParameterManager
from .modeling.structure_factor import validate_radius_effective_mode
from .persistence import build_fit_context, read_analysis, write_analysis
from .plotting import DEFAULT_POSTERIOR_PREDICTIVE_DRAWS, format_reduced_chisq, plot_fit
from .report import FitReport
from .reporting import Report, format_for, warn_if_no_image
from .reporting import render as render_report
from .results import (
    PREVIEW_ENGINE,
    FitArtifacts,
    FitResultContract,
    PosteriorSummary,
    resolve_fit_index,
    save_fit_result,
)


def get_all_models() -> list[str]:
    """
    Fetch all available models from sasmodels.

    Returns:
        List of model names

    Raises:
        Exception: Whatever ``sasmodels.core.list_models()`` raises. A broken
            sasmodels installation surfaces as an error rather than as an
            empty model list.
    """
    return sorted(core.list_models())


@functools.lru_cache(maxsize=1)
def get_structure_factors() -> tuple[str, ...]:
    """Return the structure-factor model names available in sasmodels.

    Derived from sasmodels rather than a hardcoded whitelist: every built-in
    model whose ``ModelInfo.structure_factor`` flag is set (via
    ``sasmodels.core.load_model_info``). New structure factors added upstream
    (e.g. ``two_yukawa``) are picked up automatically, so this list cannot go
    stale. The result is cached; the immutable tuple prevents accidental
    mutation of the cached value.

    Returns:
        Sorted tuple of structure-factor names.
    """
    return tuple(
        sorted(name for name in core.list_models() if core.load_model_info(name).structure_factor)
    )


def _validate_model_expression(model_name: str) -> None:
    """Validate atomic model names in a (possibly composite) expression.

    Splits on top-level ``+``/``*`` and on ``@`` (product parts). Only parts
    that look like plain model identifiers are checked against
    ``sasmodels.core.list_models()``; unknown names raise a ``ValueError``
    with a nearest-match suggestion. Anything else — custom plugin-model
    paths, parenthesized or scaled expressions — is passed through for
    ``sasmodels.core.load_model`` to accept or reject, since it is the
    authority on those forms.
    """
    available = set(core.list_models())
    for part in re.split(r'[+*]', model_name):
        part = part.strip()
        if not part:
            raise ValueError(f"Invalid model expression '{model_name}': empty component.")
        for atomic in part.split('@'):
            atomic = atomic.strip()
            if not atomic:
                raise ValueError(f"Invalid model expression '{model_name}': empty component.")
            if not re.fullmatch(r'[A-Za-z_]\w*', atomic):
                continue  # custom path or expression form — load_model decides
            if atomic not in available:
                suggestions = difflib.get_close_matches(atomic, available, n=1)
                hint = f" Did you mean '{suggestions[0]}'?" if suggestions else ''
                raise ValueError(f"Unknown model '{atomic}' in '{model_name}'.{hint}")


LMFIT_AVAILABLE = SCIPY_AVAILABLE
if not LMFIT_AVAILABLE:
    warnings.warn('scipy not available. Only bumps engine will work.', stacklevel=2)


class SANSFitter:
    """
    A flexible SANS model fitter that works with any SasModels model.

    Features:
    - Loads data from various file formats (CSV, XML, HDF5)
    - Model-agnostic: works with any model from SasModels library
    - Supports multiple fitting engines (BUMPS, LMFit)
    - User-friendly parameter management

    For model-free P(r) inversion (pair distance distribution analysis), see
    :mod:`sans_fitter.inversion` — it operates directly on datasets
    (``fitter.data`` or ``data_ops`` results) and needs no model setup.

    Example:
        >>> fitter = SANSFitter()
        >>> fitter.load_data('my_sans_data.csv')
        >>> fitter.set_model('cylinder')
        >>> fitter.set_param('radius', value=20, min=1, max=100)
        >>> fitter.set_param('length', value=400, min=10, max=1000)
        >>> result = fitter.fit(engine='bumps')
        >>> fitter.plot_results()
    """

    def __init__(self):
        """Initialize the SANS fitter."""
        self.data = None
        self.kernel = None
        self.fit_result = None
        # Where self.data came from, recorded at ingestion by load_data() and
        # set_data(); None until one of them runs. See data/provenance.py.
        self._data_source: DataSource | None = None
        self._fit_contract: FitResultContract | None = None
        self._fitted_model = None
        self._full_q_range: tuple[float, float] | None = None

        # Resolution is an analysis choice, like the model and the parameters:
        # it lives on the fitter and survives load_data()/set_data(). The
        # default 'data' means "whatever the loaded dataset carries", so it
        # keeps doing the right thing when the dataset is swapped.
        self._resolution = ResolutionSetting()

        # Parameter management delegated to ParameterManager
        self._param_manager = ParameterManager()

    def load_data(self, filename: str, dataset: int | str = 0) -> None:
        """
        Load SANS data from a file.

        Supports CSV, XML, and HDF5 formats through sasdata. Columnar text/CSV
        files are interpreted in the order Q, I, dI, dQ (per the sasdata ASCII
        convention) — a file whose third column is dQ rather than dI will have
        its uncertainties and resolution swapped. Check the column summary
        printed after loading.

        Files may hold several datasets (e.g. CanSAS XML with multiple
        ``SASentry`` blocks). Pass *dataset* to select one by 0-based index or
        by name (title, run id or filename); the first dataset is used by
        default. If the file contains more than one dataset, a warning lists
        them all.

        Args:
            filename: Path to the data file
            dataset: Which dataset to load — a 0-based index or a name (title,
                run id or filename). Defaults to the first dataset.

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the data cannot be loaded or is invalid
        """
        loaded = load_sans_dataset(filename, dataset=dataset)
        self.data = loaded.data
        # Recorded here because nothing downstream can recover it: the selector
        # is resolved to a position and dropped, and the dataset's own
        # .filename metadata is not an authoritative path.
        self._data_source = DataSource.from_file(
            filename,
            self.data,
            requested=dataset,
            index=loaded.index,
            n_datasets=loaded.n_datasets,
        )
        self._full_q_range = (self.data.qmin, self.data.qmax)

        has_dy = has_real_data(self.data.dy)
        has_dx = has_real_data(self.data.dx)

        logger.info(
            f'{OK} Loaded data from {filename}\n'
            f'  Q range: {self.data.qmin:.4f} to {self.data.qmax:.4f} {INVERSE_ANGSTROM}\n'
            f'  Data points: {len(self.data.x)}\n'
            f'  Error (dI) column: {"yes" if has_dy else "no"}\n'
            f'  Resolution (dQ) column: {"yes" if has_dx else "no"}\n'
            f'  Resolution mode: {self._resolution.describe()}'
        )

    def set_data(self, data: Any) -> None:
        """
        Use an in-memory dataset for fitting.

        This is the injection point for datasets that were not loaded from a
        file: results of dataset arithmetic (see :mod:`sans_fitter.data.ops`),
        simulated data, or any sasdata ``Data1D`` built programmatically. The
        dataset is validated and normalized (``qmin``/``qmax``/``mask`` are
        recomputed as needed) so it is fit-ready.

        Args:
            data: A sasdata ``Data1D`` object with populated ``x`` and ``y``
                arrays. 2D data is not supported.

        Raises:
            TypeError: If the object is 2D data or lacks ``x``/``y`` arrays.
            ValueError: If ``x``/``y`` are empty, have mismatched lengths, or
                contain non-positive Q values.
        """
        if getattr(data, 'qx_data', None) is not None:
            raise TypeError('2D data is not supported. Provide a Data1D object.')
        x = getattr(data, 'x', None)
        y = getattr(data, 'y', None)
        if x is None or y is None:
            raise TypeError('Dataset must have populated x and y arrays.')
        x = np.asarray(x)
        y = np.asarray(y)
        if x.size == 0 or y.size == 0:
            raise ValueError('Dataset is empty: x and y must contain data points.')
        if x.size != y.size:
            raise ValueError(f'x and y have different lengths ({x.size} vs {y.size}).')
        if np.any(x[np.isfinite(x)] <= 0):
            raise ValueError('Q values must be positive.')
        if x.size < 5:
            warnings.warn(
                f'Dataset has only {x.size} points; fits may be unreliable.',
                stacklevel=2,
            )

        self.data = normalize_sans_data(data)
        # Replaces any file provenance: this dataset did not come from a path,
        # and inferring one from its metadata would be wrong (data_ops results
        # carry an operation string in .filename, not a file).
        self._data_source = DataSource.from_memory(self.data)
        self._full_q_range = (self.data.qmin, self.data.qmax)

        has_dy = has_real_data(self.data.dy)
        has_dx = has_real_data(self.data.dx)
        label = getattr(data, 'title', '') or getattr(data, 'filename', '') or 'in-memory dataset'

        logger.info(
            f'{OK} Data set: {label}\n'
            f'  Q range: {self.data.qmin:.4f} to {self.data.qmax:.4f} {INVERSE_ANGSTROM}\n'
            f'  Data points: {len(self.data.x)}\n'
            f'  Error (dI) column: {"yes" if has_dy else "no"}\n'
            f'  Resolution (dQ) column: {"yes" if has_dx else "no"}\n'
            f'  Resolution mode: {self._resolution.describe()}'
        )

    def set_q_range(self, qmin: float | None = None, qmax: float | None = None) -> None:
        """
        Restrict the Q range used for fitting.

        Data points outside [qmin, qmax] are excluded from the fit (and from
        the exported fit curve/residuals) but remain visible in plots. Typical
        uses: trimming beam-stop spillover at low Q or background-dominated
        high-Q points.

        Args:
            qmin: Lower Q limit in Å⁻¹. If omitted, the current lower limit
                is reset to the full data range.
            qmax: Upper Q limit in Å⁻¹. If omitted, the current upper limit
                is reset to the full data range.

        Raises:
            ValueError: If no data is loaded, if qmin >= qmax, or if no data
                points remain in the requested range (the previous range is
                kept in that case).
        """
        if self.data is None:
            raise ValueError('No data loaded. Use load_data() first.')
        if qmin is None and qmax is None:
            raise ValueError('Provide qmin, qmax, or both.')

        full_min, full_max = self._full_q_range
        new_qmin = full_min if qmin is None else float(qmin)
        new_qmax = full_max if qmax is None else float(qmax)
        if new_qmin >= new_qmax:
            raise ValueError(f'qmin ({new_qmin:g}) must be smaller than qmax ({new_qmax:g}).')

        previous = (self.data.qmin, self.data.qmax)
        self.data.qmin = new_qmin
        self.data.qmax = new_qmax

        index = get_fit_index(self.data)
        n_points = int(index.sum())
        if n_points == 0:
            self.data.qmin, self.data.qmax = previous
            raise ValueError(
                f'No data points in Q range [{new_qmin:g}, {new_qmax:g}]. Range unchanged.'
            )

        logger.info(
            f'{OK} Q range for fitting: {new_qmin:.6g} to {new_qmax:.6g} {INVERSE_ANGSTROM}\n'
            f'  Points in fit: {n_points} of {len(index)}'
        )

    def reset_q_range(self) -> None:
        """
        Reset the fitting Q range to the full range of the loaded data.

        Raises:
            ValueError: If no data is loaded.
        """
        if self.data is None:
            raise ValueError('No data loaded. Use load_data() first.')

        self.data.qmin, self.data.qmax = self._full_q_range
        n_points = int(get_fit_index(self.data).sum())
        logger.info(
            f'{OK} Q range reset to {self.data.qmin:.6g} to {self.data.qmax:.6g} '
            f'{INVERSE_ANGSTROM}\n  Points in fit: {n_points}'
        )

    def get_q_range(self) -> tuple[float, float] | None:
        """
        Get the Q range currently used for fitting.

        Returns:
            Tuple (qmin, qmax) in Å⁻¹, or None if no data is loaded.
        """
        if self.data is None:
            return None
        return (self.data.qmin, self.data.qmax)

    def set_resolution(
        self,
        mode: str = 'data',
        dq_over_q: float | None = None,
        slit_length: float | None = None,
        slit_width: float | None = None,
    ) -> None:
        """
        State how instrument resolution is applied when the model is evaluated.

        Resolution smearing changes the fitted parameters, so it is a stated
        choice rather than a property of the input file. The four modes mirror
        SasView's Fit Page (*None* / *Use dQ Data* / *Custom Pinhole* /
        *Custom Slit*)::

            fitter.set_resolution('data')                     # default
            fitter.set_resolution('none')
            fitter.set_resolution('pinhole', dq_over_q=0.10)
            fitter.set_resolution('slit', slit_length=0.05)

        The setting is applied to a **copy** of the dataset made for
        evaluation; ``fitter.data`` is never modified. It reaches
        ``fit(engine='bumps')``, ``fit(engine='lmfit')``, ``fit_bayesian()``
        and the post-fit curve displays alike.

        The mode is fitter state, not data state: it **persists** across
        ``load_data()`` / ``set_data()``, exactly as the model, the parameters
        and the links do. Mode ``'data'`` already means "use *this* dataset's
        columns", so swapping datasets under the default needs no reset; the
        load summary reports the active mode so a custom width cannot be
        applied to a new file unnoticed.

        Args:
            mode: Which resolution to apply.

                - ``'data'`` (default): the dataset's own resolution columns.
                  A dQ (pinhole) column wins over slit columns, as in
                  sasmodels. A dataset carrying only ``dxl``/``dxw`` is smeared
                  with slit geometry. A dataset with no resolution columns
                  warns and is evaluated unsmeared.
                - ``'none'``: perfect resolution; any columns in the file are
                  ignored.
                - ``'pinhole'``: constant relative width, ``dx = dq_over_q·q``.
                - ``'slit'``: constant slit geometry.
            dq_over_q: Relative pinhole width **σ_q/q, a Gaussian 1-σ** — the
                same quantity as the file's dQ column and as
                :func:`sans_fitter.examples.simulate`'s ``dq``. **It is not
                FWHM.** Required by ``'pinhole'``, rejected by other modes.
            slit_length: Slit length along q, an absolute width in Å⁻¹
                (sasmodels' ``dxl``). **Required** by ``'slit'`` — sasmodels
                does not implement smearing from a slit width alone.
            slit_width: Slit width perpendicular to q, an absolute width in
                Å⁻¹ (sasmodels' ``dxw``). Optional under ``'slit'``; omit it
                for the usual long-slit (USANS) geometry.

        Raises:
            ValueError: If the mode is unknown, an argument does not belong to
                the mode, a required argument is missing, or a width is
                non-finite, negative or degenerate (a pinhole or slit length of
                zero). Every check runs before any state is touched, so a
                rejected call leaves the fitter exactly as it was.

        Note:
            Per-point custom widths, constant *absolute* σ_q, 2D/oriented
            resolution and fittable resolution parameters are out of scope.
            P(r) inversion is unaffected: it reads ``fitter.data``, which this
            setting deliberately leaves alone.
        """
        self._resolution = validate_resolution(
            mode, dq_over_q=dq_over_q, slit_length=slit_length, slit_width=slit_width
        )
        logger.info(f'{OK} Resolution: {self._resolution.describe()}')

    def get_resolution(self) -> dict[str, Any]:
        """
        Return the active resolution setting.

        Works before any data is loaded — the mode is fitter state, and the
        evaluation copy it describes is only built at fit time.

        Returns:
            A fresh dict with all four keys always present, e.g.
            ``{'mode': 'pinhole', 'dq_over_q': 0.1, 'slit_length': None,
            'slit_width': None}``. Mutating it does not affect the fitter.
        """
        return self._resolution.as_dict()

    def _evaluation_data(self, data: Any = None, *, warn: bool = True) -> Any:
        """Return the dataset copy that sasmodels should be handed.

        The single seam through which every calculator built from fitter state
        gets its data. Keeping it single is what lets the engines stay
        untouched, and it is where the ``dy`` rewriting of the data-weighting
        work will go too: copy once, apply resolution, then weighting.

        Args:
            data: Dataset to base the copy on. Defaults to ``self.data``.
            warn: Whether mode ``'data'`` may warn about missing or shadowed
                resolution columns. Post-fit evaluation paths pass ``False``
                because the fit they follow has already warned.
        """
        return apply_resolution(self.data if data is None else data, self._resolution, warn=warn)

    def set_model(self, model_name: str, platform: str = 'cpu') -> None:
        """
        Set the SANS model to use for fitting.

        Accepts both single models and composite expressions understood by
        sasmodels: ``'dab+peak_lorentz'`` (sum mixture), ``'modelA*modelB'``
        (product mixture), and ``'sphere@hardsphere'`` (form factor with
        structure factor). Every atomic model name in the expression is
        validated against the sasmodels model list before loading, with a
        nearest-match suggestion for unknown names.

        This resets any active structure factor to ensure a clean state.

        Args:
            model_name: Name of the model from SasModels (e.g., 'cylinder',
                'sphere', 'dab+peak_lorentz')
            platform: Computation platform ('cpu' or 'opencl')

        Raises:
            ValueError: If the model name is not valid
        """
        _validate_model_expression(model_name)

        try:
            # Force CPU platform to avoid OpenCL issues
            self.kernel = load_model(model_name, dtype='single', platform='dll')

            # Initialize parameters via ParameterManager. Components are
            # derived from the kernel's composition tree, not the expression
            # string (robust against nested mixture plugins).
            self._param_manager.initialize_from_kernel(self.kernel, model_name)

            logger.info(
                f"{OK} Model '{model_name}' loaded successfully\n"
                f'  Available parameters: {len(self._param_manager.params)}'
            )

        except Exception as e:
            raise ValueError(f"Failed to load model '{model_name}': {str(e)}") from e

    def set_models(
        self,
        *model_names: str,
        operation: str = '+',
        shared: Sequence[str] = (),
        **monikers: str,
    ) -> None:
        """
        Combine multiple models against the current dataset.

        The friendly-name entry point for composite models. Parameters are
        exposed with model-name (or moniker) prefixes instead of sasmodels'
        ``A_``/``B_`` prefixes, e.g. ``dab_cor_length`` instead of
        ``A_cor_length``.

        Args:
            *model_names: Model names, positionally. Each may itself contain
                ``@`` to apply a structure factor to one part (e.g.
                ``'sphere@hardsphere'``).
            operation: How to combine the models: ``'+'`` (sum mixture, the
                default) or ``'*'`` (product mixture).
            shared: Unprefixed parameter names that must exist in at least 2
                components. Each becomes a single unprefixed parameter driving
                every component that has it (e.g. ``shared=['sld']``).
                Polydispersity configuration stays per-component under the
                prefixed names.
            **monikers: Components given as ``moniker=model_name`` keyword
                arguments, for long model names, duplicates, or physics
                labels (e.g. ``small='sphere', large='sphere'``).

        Example:
            >>> fitter.set_models('dab', 'peak_lorentz')
            >>> fitter.set_param('dab_cor_length', value=50, vary=True)
            >>> fitter.set_models(small='sphere', large='sphere', shared=['sld'])

        Raises:
            ValueError: If fewer than 2 models are given, the operation is
                invalid, a moniker is invalid, a shared name is missing from
                enough components or names a global parameter
                (``'scale'``/``'background'``), the generated alias names
                collide or shadow a canonical name, or an
                entry expands to more than one kernel component (e.g. a
                nested ``'+'``/``'*'`` expression) — each entry must be a
                single component so monikers map 1:1; use the raw
                ``set_model('a+b')`` string path for nested expressions.
        """
        if operation not in ('+', '*'):
            raise ValueError(f"Invalid operation '{operation}'. Use '+' or '*'.")

        components: list[tuple[str, str]] = []  # (moniker, model_name)
        for name in model_names:
            # Positional moniker defaults to the model name; for product
            # entries ('sphere@hardsphere') use the form-factor part so the
            # moniker stays a valid identifier.
            moniker = name if name.isidentifier() else name.split('@')[0]
            components.append((moniker, name))
        for moniker, name in monikers.items():
            components.append((moniker, name))

        if len(components) < 2:
            raise ValueError(
                "set_models() requires at least 2 models. For a single model use set_model('name')."
            )

        # The global scale/background are shared by every component natively;
        # letting them through shared= would collapse the per-component
        # scales onto the global entry and silently drop it from the fit.
        conflicting = {'scale', 'background'} & set(shared)
        if conflicting:
            raise ValueError(
                f'Cannot share the global parameter(s) {", ".join(sorted(conflicting))}: '
                "'scale' and 'background' are already shared by every component."
            )

        # Validate monikers: valid identifiers and not reserved names.
        # Positional model names may repeat (they get auto-suffixed below);
        # keyword monikers must be unique among themselves.
        reserved = {'scale', 'background'} | set(shared)
        for moniker, _name in components:
            if not moniker.isidentifier():
                raise ValueError(
                    f"Component name '{moniker}' is not a valid identifier. "
                    'Use keyword monikers for non-identifier model names.'
                )
            if moniker in reserved:
                raise ValueError(
                    f"Component name '{moniker}' is reserved "
                    "(collides with 'scale', 'background', or a shared name)."
                )
        keyword_monikers = [moniker for moniker, _name in components[len(model_names) :]]
        if len(set(keyword_monikers)) != len(keyword_monikers):
            raise ValueError('Duplicate keyword monikers are not allowed.')

        # Duplicate positional model names auto-suffix their monikers
        # (sphere1_, sphere2_); keyword monikers are the recommended spelling
        # for that case.
        positional_counts: dict[str, int] = {}
        for name in model_names:
            positional_counts[name] = positional_counts.get(name, 0) + 1
        duplicate_names = {name for name, count in positional_counts.items() if count > 1}

        resolved: list[tuple[str, str]] = []
        dup_counters: dict[str, int] = {}
        for moniker, name in components:
            if moniker == name and name in duplicate_names:
                dup_counters[name] = dup_counters.get(name, 0) + 1
                resolved.append((f'{name}{dup_counters[name]}', name))
            else:
                resolved.append((moniker, name))
        components = resolved

        # Re-check uniqueness after auto-suffixing (a generated suffix could
        # collide with an explicit moniker).
        all_monikers = [moniker for moniker, _name in components]
        if len(set(all_monikers)) != len(all_monikers):
            raise ValueError(
                'Component names collide after auto-suffixing duplicates: '
                f'{all_monikers}. Use distinct keyword monikers.'
            )

        # Delegate loading/validation to set_model using canonical syntax.
        expression = operation.join(name for _moniker, name in components)
        self.set_model(expression)

        # Register the friendly-name alias layer. register_aliases raises on
        # shared-name or alias-collision problems (detected by building the
        # full alias map, not by ad-hoc string rules).
        self._param_manager.register_aliases(components, list(shared))

        lines = [
            f'{OK} Combined {len(components)} models: {expression}',
            f'  Components: {", ".join(m for m, _n in components)}',
        ]
        if shared:
            lines.append(f'  Shared parameters: {", ".join(shared)}')
        lines.append(f'  Available parameters: {len(self._param_manager.params)}')
        logger.info('\n'.join(lines))

    def link_params(self, name: str, to: str) -> None:
        """
        Create an equality link between two parameters.

        The follower (*name*) is forced to ``vary=False`` and mirrors the
        target's (*to*) value at all times — before, during, and after the
        fit. Links are equality-only; no expressions. Works for any pair of
        parameters, including cross-component ones (``'large_sld'`` following
        ``'small_sld'``) and differently named ones.

        This is the same mechanism as
        ``set_structure_factor(..., radius_effective_mode='link_radius')``,
        which links ``'radius_effective'`` to ``'radius'``; ``get_links()``
        reports both alike.

        Args:
            name: The follower parameter name.
            to: The target parameter name.

        Raises:
            KeyError: If either name does not exist.
            ValueError: On self-links, link chains, or conflicting links.
        """
        self._param_manager.link_params(name, to)
        logger.info(f'{OK} Linked {name} {ARROW} {to}')

    def unlink_params(self, name: str) -> None:
        """
        Remove an equality link, restoring the follower's independence.

        Args:
            name: The follower parameter name.

        Raises:
            KeyError: If the name does not exist.
            ValueError: If the parameter is not linked.
        """
        self._param_manager.unlink_params(name)
        logger.info(f'{OK} Unlinked {name}')

    def get_links(self) -> dict[str, str]:
        """Return the active parameter equality links (follower -> target)."""
        return self._param_manager.get_links()

    def get_components(self) -> list[tuple[str, str, str]]:
        """
        Return the composite-model components.

        Returns:
            List of ``(prefix, moniker, part_model_name)`` triples, e.g.
            ``[('A', 'dab', 'dab'), ('B', 'peak_lorentz', 'peak_lorentz')]``.
            Empty for atomic models.
        """
        return self._param_manager.get_components()

    # =========================================================================
    # Property accessors for backward compatibility
    # =========================================================================

    @property
    def model_name(self) -> str | None:
        """Get the current model name."""
        return self._param_manager.model_name

    @model_name.setter
    def model_name(self, value: str | None) -> None:
        """Set the model name (used internally)."""
        self._param_manager.model_name = value

    @property
    def params(self) -> dict[str, dict[str, Any]]:
        """Get the parameter dictionary."""
        return self._param_manager.params

    @params.setter
    def params(self, value: dict[str, dict[str, Any]]) -> None:
        """Set the parameter dictionary (used internally)."""
        self._param_manager.params = value

    @property
    def _structure_factor_name(self) -> str | None:
        """Get the structure factor name."""
        return self._param_manager.get_structure_factor()

    @property
    def _radius_effective_mode(self) -> str:
        """Get the radius effective mode."""
        return self._param_manager.get_radius_effective_mode()

    def get_params(self) -> None:
        """Display current parameter values and settings in a readable format."""
        self._param_manager.display_params()

    def set_param(
        self,
        name: str,
        value: float | None = None,
        min: float | None = None,
        max: float | None = None,
        vary: bool | None = None,
    ) -> None:
        """
        Configure a model parameter for fitting.

        Args:
            name: Parameter name
            value: Initial value (optional)
            min: Minimum bound (optional)
            max: Maximum bound (optional)
            vary: Whether to vary during fit (optional)

        Raises:
            KeyError: If parameter name doesn't exist for the current model
        """
        self._param_manager.set_param(name, value=value, min=min, max=max, vary=vary)

    def set_structure_factor(
        self, structure_factor_name: str, radius_effective_mode: str = 'unconstrained'
    ) -> None:
        """
        Apply a structure factor to the current model.

        This creates a product model (form_factor * structure_factor) to account
        for inter-particle interactions in concentrated systems.

        Available structure factors are queried from sasmodels at runtime —
        see :func:`get_structure_factors` for the full, up-to-date list
        (e.g. 'hardsphere', 'hayter_msa', 'squarewell', 'stickyhardsphere').

        Args:
            structure_factor_name: Name of the structure factor (e.g., 'hardsphere')
            radius_effective_mode: How to handle the effective radius.
                - 'unconstrained': 'radius_effective' is a separate fitting parameter.
                - 'link_radius': 'radius_effective' is constrained to the form factor's 'radius'.

        Raises:
            ValueError: If no form factor model is set, or if the structure
                factor name or radius_effective_mode is invalid. Every check
                runs before any state is touched, so a rejected call leaves the
                fitter exactly as it was.
        """
        if self.kernel is None or self.model_name is None:
            raise ValueError('No form factor model loaded. Use set_model() first.')

        if self._param_manager.get_components():
            raise ValueError(
                'Cannot apply a structure factor to a composite model. '
                "The expression '(modelA+modelB)@sf' cannot be expressed in "
                'sasmodels, and naive concatenation would be parsed as '
                "'modelA + (modelB@sf)'. Apply the structure factor to one "
                "part instead, e.g. set_models('sphere@hardsphere', 'peak_lorentz')."
            )

        # Validate structure factor name against sasmodels (cached query)
        supported_sf = get_structure_factors()
        if structure_factor_name not in supported_sf:
            raise ValueError(
                f"Unsupported structure factor '{structure_factor_name}'. "
                f'Supported: {", ".join(supported_sf)}'
            )

        # Validate the mode here, before anything is swapped. Leaving it to
        # update_for_product_model would abort *after* self.kernel had become
        # the product model while params still described the form factor — a
        # desynchronized state in which the next fit silently evaluates the
        # product kernel with default structure-factor parameters.
        validate_radius_effective_mode(radius_effective_mode)

        # Create product model name
        full_model_name = f'{self.model_name}@{structure_factor_name}'

        try:
            # Load the product model. Held locally until the parameter manager
            # has accepted it, so a failure cannot desynchronize the two.
            product_kernel = load_model(full_model_name, dtype='single', platform='dll')

            # Delegate parameter management to ParameterManager
            self._param_manager.update_for_product_model(
                product_kernel, structure_factor_name, radius_effective_mode
            )
            self.kernel = product_kernel

            lines = []
            if radius_effective_mode == 'link_radius':
                lines.append("  Note: 'radius_effective' linked to 'radius' value")
            lines.append(
                f"{OK} Structure factor '{structure_factor_name}' applied to '{self.model_name}'"
            )
            lines.append(f'  Product model: {full_model_name}')
            lines.append(f'  Total parameters: {len(self.params)}')
            logger.info('\n'.join(lines))

        except Exception as e:
            raise ValueError(f"Failed to load model '{full_model_name}': {str(e)}") from e

    def remove_structure_factor(self) -> None:
        """
        Remove the current structure factor and revert to the form factor only.

        Raises:
            ValueError: If no structure factor is currently set
        """
        if self._structure_factor_name is None:
            raise ValueError('No structure factor is currently set.')

        # Reload the original form factor model. Held locally until the
        # parameter manager has restored its side, for the same reason as in
        # set_structure_factor: kernel and params must never disagree.
        try:
            form_factor_kernel = load_model(self.model_name, dtype='single', platform='dll')

            # Delegate to ParameterManager - this restores params and PD state
            sf_name = self._param_manager.remove_structure_factor()
            self.kernel = form_factor_kernel

            logger.info(
                f"{OK} Structure factor '{sf_name}' removed\n"
                f'  Reverted to form factor: {self.model_name}'
            )

        except Exception as e:
            raise ValueError(f'Failed to reload form factor model: {str(e)}') from e

    def get_structure_factor(self) -> str | None:
        """
        Get the name of the currently applied structure factor.

        Returns:
            Name of the structure factor, or None if no structure factor is set
        """
        return self._structure_factor_name

    # =========================================================================
    # Polydispersity Methods
    # =========================================================================

    def supports_polydispersity(self) -> bool:
        """
        Check if current model has polydisperse parameters.

        Returns:
            True if model supports polydispersity, False otherwise
        """
        return self._param_manager.has_polydisperse_parameters()

    def get_polydisperse_parameters(self) -> list[str]:
        """
        Get list of polydisperse parameter names.

        Returns:
            List of parameter names that support polydispersity
        """
        return self._param_manager.get_polydisperse_parameters()

    def set_pd_param(
        self,
        param_name: str,
        pd_width: float | None = None,
        pd_n: int | None = None,
        pd_nsigma: float | None = None,
        pd_type: str | None = None,
        vary: bool | None = None,
    ) -> None:
        """
        Configure polydispersity for a parameter.

        Args:
            param_name: Name of the base parameter (e.g., 'radius')
            pd_width: Polydispersity width (relative, 0.0 = monodisperse)
            pd_n: Number of Gaussian quadrature points (default: 35)
            pd_nsigma: Number of sigmas to include (default: 3.0)
            pd_type: Distribution type ('gaussian', 'rectangle', 'lognormal', 'schulz', 'boltzmann')
            vary: Whether to vary the pd_width during fitting

        Raises:
            KeyError: If param_name is not a polydisperse parameter
            ValueError: If pd_type is not a valid distribution type
        """
        self._param_manager.set_pd_param(
            param_name,
            pd_width=pd_width,
            pd_n=pd_n,
            pd_nsigma=pd_nsigma,
            pd_type=pd_type,
            vary=vary,
        )

    def get_pd_param(self, param_name: str) -> dict[str, Any]:
        """
        Get polydispersity configuration for a parameter.

        Args:
            param_name: Name of the base parameter (e.g., 'radius')

        Returns:
            Dictionary with pd, pd_n, pd_nsigma, pd_type, vary, and active values.
            'active' indicates whether polydispersity is active for this parameter (pd > 0).

        Raises:
            KeyError: If param_name is not a polydisperse parameter
        """
        return self._param_manager.get_pd_param(param_name)

    def enable_polydispersity(self, enabled: bool = True) -> None:
        """
        Enable or disable polydispersity globally.

        When disabled, polydispersity parameters are excluded from fitting
        but their values are preserved for when PD is re-enabled.

        Args:
            enabled: Whether to enable polydispersity (default: True)
        """
        self._param_manager.toggle_pd_visibility(enabled)

    def is_polydispersity_enabled(self) -> bool:
        """
        Check if polydispersity is enabled.

        Returns:
            True if polydispersity is globally enabled, False otherwise
        """
        return self._param_manager.is_pd_enabled()

    def get_pd_params(self) -> None:
        """Display polydispersity parameter values and settings."""
        self._param_manager.display_pd_params()

    def get_varying_pd_params(self) -> list[str]:
        """
        Get list of polydispersity parameters that are set to vary.

        Returns:
            List of parameter names (e.g., ['radius_pd']) that will vary during fitting
        """
        # ParameterManager returns base param names, we need to add _pd suffix
        # and translate to user-facing aliases on the set_models path.
        varying_base = self._param_manager.get_varying_pd_params()
        return [
            self._param_manager.to_display_name(f'{param_name}_pd') for param_name in varying_base
        ]

    def _finalize_fit(self, engine_output) -> dict[str, Any]:
        """Apply engine output to fitter state and return legacy-compatible results."""
        self._param_manager.apply_fitted_values(engine_output.fitted_values)
        self._fit_contract = engine_output.contract
        # Record what smeared this fit: the parameter values only mean
        # something alongside the resolution that produced them, and χ² is not
        # comparable across modes.
        self._fit_contract.resolution = self._resolution.as_dict()

        # Translate engine result names (canonical) back to user-facing names
        # so saved results and displays never expose A_/B_ on the set_models
        # path (Boundary 2 of the alias layer).
        to_display = self._param_manager.to_display_name
        self._fit_contract.parameters = {
            to_display(name): dict(info)
            | {'linked_to': to_display(info['linked_to']) if info['linked_to'] else None}
            for name, info in self._fit_contract.parameters.items()
        }

        # Covariance labels are canonical too, and land next to the parameter
        # names in the same translation (Boundary 2 of the alias layer).
        self._fit_contract.cov_labels = [to_display(name) for name in self._fit_contract.cov_labels]

        # Record the configuration and data this result belongs to. No setter
        # clears a fit result, so without this a later save_analysis() could
        # pair the current settings with a chi-squared that a different set
        # produced. Taken after apply_fitted_values, so it describes the fit.
        self._fit_contract.fit_context = build_fit_context(
            self._param_manager.export_config(),
            self._resolution.as_dict(),
            self.get_q_range(),
            self._fit_contract.n_points,
            None if self.data is None else fingerprint_arrays(self.data),
        )

        # Attach per-component curves for '+' mixture models (no-op otherwise).
        component_curves = self._compute_component_curves()
        if component_curves:
            self._fit_contract.artifacts.component_curves = component_curves

        self._fit_contract.on_bounds = self._find_parameters_on_bounds()

        self.fit_result = self._fit_contract.to_legacy_dict()
        self._fitted_model = engine_output.runtime_model

        logger.info(f'\n{OK} Fit completed!\n{self.get_fit_report()}')

        self._warn_about_bounds(self._fit_contract.on_bounds)
        if self._fit_contract.converged is False:
            warnings.warn(
                f'The optimizer did not report convergence: {self._fit_contract.message}. '
                'The reported parameters are wherever it stopped; increase the '
                'iteration budget or revisit the starting values.',
                stacklevel=3,
            )

        return self.fit_result

    def _find_parameters_on_bounds(self) -> list[tuple[str, str]]:
        """Varied parameters whose fitted value sits on one of their bounds.

        Only the optimizer's own dimensions can hit a wall, so fixed and linked
        parameters are skipped even when their value happens to equal a bound.
        Polydispersity widths are bounded [0, 1] by both engines rather than by
        ``ParameterManager``, so their limits are supplied here.

        Names are user-facing: this runs after the display-name translation.
        """
        hits: list[tuple[str, str]] = []
        for name, info in self._fit_contract.parameters.items():
            if info.get('fixed', False) or info.get('linked_to') is not None:
                continue
            value = info.get('value')
            if value is None:
                continue

            bounds = self._param_manager.params.get(name)
            if bounds is not None:
                lo, hi = bounds.get('min'), bounds.get('max')
            elif name.endswith('_pd'):
                # PD widths are not in the parameter table; both engines bound
                # them to [0, 1] themselves.
                lo, hi = 0.0, 1.0
            else:
                continue
            if lo is not None and at_bound(value, lo):
                hits.append((name, 'min'))
            if hi is not None and at_bound(value, hi):
                hits.append((name, 'max'))
        return hits

    def _warn_about_bounds(self, on_bounds: list[tuple[str, str]]) -> None:
        """Warn once about every fitted parameter resting on a bound.

        A warning rather than a log line, so it survives
        ``set_verbosity('quiet')`` and shows up in a notebook. The wording is
        deliberately neutral: an optimum at a bound can be physically correct (a
        non-negative background at zero), so the claim is that the estimate is
        constrained by the configured domain, not that a better one lies outside it.
        """
        if not on_bounds:
            return
        hits = ', '.join(
            f'{name} = {self._fit_contract.parameters[name]["value"]:.6g} ({side})'
            for name, side in on_bounds
        )
        warnings.warn(
            f'Fitted parameter(s) at a bound: {hits}. The estimate may be constrained '
            'by these limits; review or widen the bounds if the boundary was not '
            'intentional. Uncertainties from a symmetric covariance estimate are '
            'unreliable at an active bound.',
            stacklevel=3,
        )

    def get_fit_report(self) -> FitReport:
        """
        Return a :class:`~sans_fitter.report.FitReport` for the last fit.

        The report carries the goodness-of-fit statistics, the parameter table, the
        covariance and correlation matrices, the convergence verdict and any
        parameter resting on a bound. It renders itself as a table in a notebook
        (``_repr_html_``), as plain text (``print(report)``) and as Markdown
        (``report.to_markdown()``), and serializes through ``report.to_dict()``.

        Returns:
            A snapshot of the current fit result. Mutating the fitter afterwards
            does not change a report already returned.

        Raises:
            ValueError: If no fit has been run in this session. A theory preview
                (``plot_model()``) does not produce a fit result.
        """
        if self._fit_contract is None:
            raise ValueError(
                'No fit result available. Run fit() or fit_bayesian() first; '
                'plot_model() previews the theory without fitting.'
            )
        return FitReport.from_contract(self._fit_contract, self.model_name)

    def _compute_component_curves(
        self, canonical_values: dict[str, float] | None = None
    ) -> dict[str, np.ndarray] | None:
        """Compute per-component curves after a fit of a '+' mixture model.

        Each component curve is ``scale · I_part(q, scale=part_scale,
        background=0)`` — matching the mixture kernel's own computation — so
        the component curves plus the background stack onto the total curve.

        Returns None for atomic models and '*' mixtures (where part curves
        would not stack to the total and would mislead when overlaid).
        Evaluation happens on the same masked q-points as the total curve.

        Args:
            canonical_values: Parameter values keyed by canonical sasmodels
                name, overriding this fitter's own. A simultaneous fit resolves
                its values through the constraint graph rather than from each
                child's parameter table, and must be able to say so; the default
                reads this fitter's values and is what the single-fit path uses.
        """
        components = self._param_manager.get_components()
        if not components:
            return None

        # '*' mixtures: part intensities multiply, so additive component
        # curves are meaningless. Documented no-op.
        operation = getattr(self.kernel.info, 'operation', '+')
        if operation != '+':
            return None

        if canonical_values is None:
            canonical_values = self._param_manager.get_canonical_param_values()
        global_scale = canonical_values.get('scale', 1.0)

        # Active polydispersity settings, keyed by canonical base names.
        pd_settings: dict[str, dict[str, Any]] = {}
        if self._param_manager.is_pd_enabled():
            for base_param in self._param_manager.get_polydisperse_parameters():
                pd_config = self._param_manager.polydisperse_params[base_param]
                if pd_is_active(pd_config):
                    pd_settings[base_param] = pd_config

        # Built once and shared by every component: same resolution as the
        # fit, so the parts and the total are smeared alike.
        evaluation_data = self._evaluation_data(warn=False)

        curves: dict[str, np.ndarray] = {}
        for prefix, moniker, part_name in components:
            # Label: moniker, with the model name appended when they differ;
            # on the raw-string path moniker == prefix ('A: dab').
            if moniker == prefix and moniker != part_name:
                label = f'{prefix}: {part_name}'
            elif moniker != part_name:
                label = f'{moniker} ({part_name})'
            else:
                label = moniker

            part_kernel = load_model(part_name, dtype='single', platform='dll')
            # Evaluated through the same resolution as the fit: otherwise a
            # slit or custom-pinhole fit would overlay sharp component curves
            # under a smeared total.
            calculator = DirectModel(evaluation_data, part_kernel)

            # Map fitted values by stripping the component prefix; fold in
            # active PD settings the same way the posterior evaluator does.
            part_pars: dict[str, Any] = {}
            prefix_marker = f'{prefix}_'
            for canonical, value in canonical_values.items():
                if canonical.startswith(prefix_marker):
                    stripped = canonical[len(prefix_marker) :]
                    part_pars[stripped] = value
            # The part's own scale slot gets the component scale; background
            # is excluded from component curves (shown implicitly in total).
            part_scale = canonical_values.get(f'{prefix}_scale', 1.0)
            part_pars['scale'] = part_scale
            part_pars['background'] = 0.0
            for base_param, pd_config in pd_settings.items():
                if base_param.startswith(prefix_marker):
                    stripped = base_param[len(prefix_marker) :]
                    part_pars[f'{stripped}_pd'] = pd_config['pd']
                    part_pars[f'{stripped}_pd_n'] = pd_config['pd_n']
                    part_pars[f'{stripped}_pd_nsigma'] = pd_config['pd_nsigma']
                    part_pars[f'{stripped}_pd_type'] = pd_config['pd_type']

            curves[label] = global_scale * np.asarray(calculator(**part_pars))

        return curves

    def _legacy_quality_block(
        self, fit_result: dict[str, Any], fit_index: np.ndarray
    ) -> dict[str, Any]:
        """Goodness-of-fit fields for a contract rebuilt from a result dictionary.

        A dictionary that already carries ``reduced_chisq`` came from this version
        and is read verbatim. An older one carries a single ``chisq`` whose meaning
        depended on the engine — bumps stored χ²/dof, the scipy engine stored the
        raw sum — so the conversion is determined, not guessed: recover the counts
        from the fit index and the parameter block, then derive whichever of the two
        χ² values the dictionary does not hold.
        """
        if 'reduced_chisq' in fit_result:
            return {
                'chisq': fit_result['chisq'],
                'reduced_chisq': fit_result['reduced_chisq'],
                'n_points': fit_result['n_points'],
                'n_free': fit_result['n_free'],
                'dof': fit_result['dof'],
                'weighting_note': fit_result.get('weighting_note', 'dI'),
                'converged': fit_result.get('converged'),
                'message': fit_result.get('message', ''),
                'cov': fit_result.get('cov'),
                'cov_labels': list(fit_result.get('cov_labels', [])),
                'cov_source': fit_result.get('cov_source'),
                'on_bounds': list(fit_result.get('on_bounds', [])),
            }

        legacy_chisq = float(fit_result['chisq'])
        n_points = int(np.asarray(fit_index, dtype=bool).sum())
        n_free = sum(1 for info in fit_result['parameters'].values() if not info.get('fixed', True))
        dof = n_points - n_free
        if fit_result['engine'] == 'bumps':
            raw = legacy_chisq * dof if dof > 0 else float('nan')
            reduced = legacy_chisq
        else:
            raw = legacy_chisq
            reduced = reduced_chisq(legacy_chisq, dof)
        return {
            'chisq': raw,
            'reduced_chisq': reduced,
            'n_points': n_points,
            'n_free': n_free,
            'dof': dof,
            'weighting_note': 'dI',
            'message': 'legacy result adapted from fit_result',
        }

    def _get_active_fit_contract(self) -> FitResultContract | None:
        """Return the active fit contract, adapting legacy runtime state if needed."""
        if self._fit_contract is not None:
            return self._fit_contract

        if self.fit_result is None:
            return None

        if self.fit_result['engine'] == 'bumps':
            fit_index = extract_fit_index(self._fitted_model.active_model)
            curve = np.asarray(self._fitted_model.active_model.theory())
            resolved_index = resolve_fit_index(fit_index, len(self.data.x))
            return FitResultContract(
                engine=self.fit_result['engine'],
                method=self.fit_result['method'],
                parameters=self.fit_result['parameters'],
                artifacts=FitArtifacts(fitted_curve=curve, fit_index=fit_index),
                resolution=self._resolution.as_dict(),
                **self._legacy_quality_block(self.fit_result, resolved_index),
            )

        # Re-evaluating legacy lmfit runtime state: use the same evaluation
        # copy the fit itself used, or the re-plotted/saved curve would be
        # unsmeared while the fit was smeared.
        calculator = DirectModel(self._evaluation_data(warn=False), self.kernel)
        par_dict = {name: info['value'] for name, info in self.fit_result['parameters'].items()}
        fit_index = extract_fit_index(calculator)
        resolved_index = resolve_fit_index(fit_index, len(self.data.x))
        return FitResultContract(
            engine=self.fit_result['engine'],
            method=self.fit_result['method'],
            parameters=self.fit_result['parameters'],
            artifacts=FitArtifacts(
                fitted_curve=np.asarray(calculator(**par_dict)),
                fit_index=fit_index,
            ),
            resolution=self._resolution.as_dict(),
            **self._legacy_quality_block(self.fit_result, resolved_index),
        )

    def fit(
        self,
        engine: Literal['bumps', 'lmfit'] = 'bumps',
        method: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Perform the fit using the specified engine.

        Args:
            engine: Fitting engine ('bumps' or 'lmfit')
            method: Optimization method (engine-specific)
                   - BUMPS: 'amoeba', 'lm', 'newton', 'de' (default: 'amoeba')
                   - LMFit: 'leastsq', 'least_squares', 'differential_evolution', etc.
            **kwargs: Additional arguments passed to the fitting engine

        Returns:
            Dictionary with ``engine``, ``method``, ``parameters`` and an
            engine-independent goodness-of-fit block: ``chisq`` (the **raw**
            weighted sum of squared residuals), ``reduced_chisq``
            (``chisq / dof``, not a number when ``dof <= 0``), ``n_points``,
            ``n_free``, ``dof``, ``converged`` (``None`` on the bumps engine,
            which reports success unconditionally), ``message``,
            ``weighting_note``, ``cov`` / ``cov_labels`` / ``cov_source``, and
            ``on_bounds``. Call :meth:`get_fit_report` for the same information
            as a self-rendering object.

            The ``parameters`` block is engine-independent too: one entry per
            model parameter, each carrying ``value``, ``stderr``, ``formatted``,
            a ``fixed`` flag (``False`` only for the parameters the optimizer
            varied) and ``linked_to`` (the parameter it follows, or ``None``).
            A follower reports its target's fitted value.

            Changed in 0.4: ``chisq`` from the bumps engine used to be
            χ²/dof. Use ``reduced_chisq`` for that value.

        Raises:
            ValueError: If data or model not loaded, or invalid engine
            NotImplementedError: If a composite model is used with an engine
                other than 'bumps'.
        """
        self._require_data()
        self._require_model()

        if engine not in ('bumps', 'lmfit'):
            raise ValueError(f"Unknown engine '{engine}'. Use 'bumps' or 'lmfit'.")

        self._check_composite_engine_support(engine)
        self._check_scale_degeneracy()
        self._check_fit_uncertainties(engine)
        self._log_active_resolution()

        if engine == 'bumps':
            return self._fit_bumps(method or 'amoeba', **kwargs)
        if not LMFIT_AVAILABLE:
            raise ValueError("scipy is not installed. Use 'bumps' engine or install scipy.")
        return self._fit_lmfit(method or 'leastsq', **kwargs)

    def _log_active_resolution(self) -> None:
        """Report the resolution a fit is about to use, before the engine runs.

        Before, not after: the engine's own "Initial χ²" line is already a
        smeared number, so the reader needs to know what smeared it first.
        """
        logger.info(f'Resolution: {self._resolution.describe()}')

    def _check_composite_engine_support(self, engine: str) -> None:
        """Gate composite models to the bumps engine.

        The scipy path would probably work for composites (DirectModel accepts
        prefixed kwargs) but it is untested; failing loudly beats silently
        unvalidated results. Parameter links are *not* gated: both engines apply
        them on every model evaluation. shared= needs no gate of its own — it
        only exists on composite models, which this check already covers.
        """
        if engine == 'bumps':
            return
        snapshot = self._param_manager.snapshot_fit_state()
        if snapshot.components:
            raise NotImplementedError(
                "Composite models are currently supported by the 'bumps' engine only."
            )

    def _check_scale_degeneracy(self) -> None:
        """Warn when the global scale and a component scale are both free.

        Under a mixture, the total intensity is scale · Σ(part_scale · I_part);
        varying both the global scale and any component scale is degenerate —
        only their product is fitted.
        """
        varying = self._param_manager.get_varying_params()
        if 'scale' not in varying:
            return
        # Atomic models can expose their own *_scale parameters (broad_peak,
        # gel_fit, ...) that are not mixture component scales.
        if not self._param_manager.get_components():
            return
        component_scales = [name for name in varying if name.endswith('_scale') and name != 'scale']
        if component_scales:
            warnings.warn(
                "Both the global 'scale' and component scale(s) "
                f'{", ".join(component_scales)} are varying. Their product is '
                'what the fit sees, so the split between them is degenerate. '
                'Fix one of them.',
                stacklevel=3,
            )

    def _check_fit_uncertainties(self, engine: str) -> None:
        """Validate intensity uncertainties (dI) before fitting.

        Both engines weight residuals by dI. Zero (or absent) uncertainties
        make the BUMPS χ² infinite for every parameter set, so the fit cannot
        proceed; the scipy/lmfit engine falls back to unit weights for the
        affected points (with a warning from the engine itself).
        """
        index = get_fit_index(self.data)
        dy = getattr(self.data, 'dy', None)
        if dy is None or np.asarray(dy).size == 0:
            n_zero = int(index.sum())
        else:
            dy_fit = np.asarray(dy, dtype=float)[index]
            n_zero = int(np.sum(np.nan_to_num(dy_fit) == 0))
        if n_zero == 0:
            return

        n_fit = int(index.sum())
        detail = (
            'has no intensity uncertainties (dI)'
            if n_zero == n_fit
            else f'has {n_zero} of {n_fit} fitted points with zero intensity uncertainty (dI)'
        )
        if engine == 'bumps':
            raise ValueError(
                f'Data {detail}. The bumps engine cannot weight such points '
                '(χ² becomes infinite). Provide dI values, exclude the points '
                "(mask or set_q_range), or use engine='lmfit', which treats "
                'them as unweighted.'
            )
        warnings.warn(
            f'Data {detail}. Affected residuals will be unweighted (dI treated as 1.0), '
            'so these points may dominate χ² relative to points with small errors.',
            stacklevel=2,
        )

    def _fit_bumps(self, method: str = 'amoeba', **kwargs: Any) -> dict[str, Any]:
        """Fit using BUMPS engine."""
        engine_output = fit_bumps(
            data=self._evaluation_data(),
            kernel=self.kernel,
            fit_state=self._param_manager.snapshot_fit_state(),
            method=method,
            **kwargs,
        )
        return self._finalize_fit(engine_output)

    def _fit_lmfit(self, method: str = 'leastsq', **kwargs: Any) -> dict[str, Any]:
        """Fit using scipy.optimize (leastsq/least_squares) engine."""
        engine_output = fit_scipy(
            data=self._evaluation_data(),
            kernel=self.kernel,
            fit_state=self._param_manager.snapshot_fit_state(),
            method=method,
            **kwargs,
        )
        return self._finalize_fit(engine_output)

    def fit_bayesian(
        self,
        method: str = 'dream',
        samples: int = DEFAULT_DREAM_SAMPLES,
        burn: int = DEFAULT_DREAM_BURN,
        thin: int = DEFAULT_DREAM_THIN,
        pop: int = DEFAULT_DREAM_POP,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Perform a Bayesian (MCMC) fit using bumps' DREAM sampler.

        Samples the posterior distribution of the varying parameters and
        stores the chain alongside the usual point-estimate results, enabling
        the posterior displays: plot_posterior_pairs(),
        plot_param_distribution(), plot_posterior_predictive(),
        plot_param_correlations(), and plot_trace().

        The reported parameter values are the best (maximum-likelihood)
        posterior sample; the reported stderr is the posterior 68% credible
        half-width.

        Args:
            method: Sampler method (default 'dream').
            samples: Number of posterior samples to draw.
            burn: Number of burn-in generations to discard (DREAM's native
                unit: each generation advances every chain by one step).
            thin: Keep every nth sample.
            pop: Population (chain) scale factor per varying parameter.
            **kwargs: Additional arguments passed to bumps.fitters.fit.

        Returns:
            The same dictionary shape as :meth:`fit`, including the
            goodness-of-fit block. ``cov`` here is the sample covariance of the
            posterior draw rather than a Jacobian estimate, and ``converged`` is
            None — a sampler has no optimizer verdict, so the ``message``
            carries the sampler settings and the largest R-hat instead. The
            posterior itself is available via get_posterior(), and
            get_fit_report() renders everything as a table.

        Raises:
            ValueError: If data or model is not loaded, or no parameter varies.
            NotImplementedError: If a composite model is used — the DREAM path
                does not support them yet.
        """
        self._require_data()
        self._require_model()

        snapshot = self._param_manager.snapshot_fit_state()
        if snapshot.components:
            raise NotImplementedError(
                "Composite models are currently supported by the 'bumps' "
                "point-estimate engine only (fit(engine='bumps'))."
            )
        self._check_scale_degeneracy()
        self._log_active_resolution()

        engine_output = fit_bumps_dream(
            data=self._evaluation_data(),
            kernel=self.kernel,
            fit_state=self._param_manager.snapshot_fit_state(),
            method=method,
            samples=samples,
            burn=burn,
            thin=thin,
            pop=pop,
            **kwargs,
        )
        return self._finalize_fit(engine_output)

    def get_posterior(self) -> PosteriorSummary:
        """
        Return the posterior summary from the last Bayesian fit.

        Raises:
            ValueError: If no fit has been run or the last fit was not Bayesian.
        """
        contract = self._get_active_fit_contract()
        if contract is None:
            raise ValueError('No fit results available. Run fit_bayesian() first.')
        return contract.require_posterior()

    def plot_posterior_pairs(
        self,
        params: list[str] | None = None,
        show_contours: bool = True,
        show: bool | None = None,
    ) -> Figure:
        """
        Corner plot of the posterior: marginal densities and pairwise clouds.

        Args:
            params: Optional subset of parameter names (default: all sampled).
            show_contours: Overlay density contours on the pairwise panels.
            show: Same display convention as plot_results().

        Raises:
            ValueError: If the last fit was not Bayesian.
        """
        return plotting.plot_posterior_pairs(
            self.get_posterior(), params=params, show_contours=show_contours, show=show
        )

    def plot_param_distribution(
        self,
        param: str,
        bins: int = 50,
        show: bool | None = None,
    ) -> Figure:
        """
        Marginal posterior distribution for one parameter.

        Args:
            param: Name of a sampled (varying) parameter.
            bins: Number of histogram bins.
            show: Same display convention as plot_results().

        Raises:
            ValueError: If the last fit was not Bayesian.
            KeyError: If the parameter was not sampled.
        """
        return plotting.plot_param_distribution(self.get_posterior(), param, bins=bins, show=show)

    def plot_posterior_predictive(
        self,
        style: str = 'band',
        n_draws: int = DEFAULT_POSTERIOR_PREDICTIVE_DRAWS,
        log_scale: bool = True,
        show: bool | None = None,
    ) -> Figure:
        """
        Posterior predictive check: credible band and/or draws over the data.

        Args:
            style: 'band' (95% credible interval), 'draws' (sampled curves),
                or 'band+draws'.
            n_draws: Number of posterior samples to evaluate through the
                model. Each draw costs one sasmodels evaluation, so large
                values can be slow (especially with polydispersity).
            log_scale: Use log axes.
            show: Same display convention as plot_results().

        Raises:
            ValueError: If the last fit was not Bayesian or no data is loaded.
        """
        contract = self._get_active_fit_contract()
        if contract is None:
            raise ValueError('No fit results available. Run fit_bayesian() first.')
        posterior = contract.require_posterior()
        posterior_data = contract.artifacts.posterior_data
        model_eval = contract.artifacts.posterior_model_eval
        if posterior_data is None or model_eval is None:
            raise ValueError('Bayesian fit does not include posterior predictive artifacts.')

        return plotting.plot_posterior_predictive(
            data=posterior_data,
            posterior=posterior,
            model_eval=model_eval,
            style=style,
            n_draws=n_draws,
            fit_index=contract.artifacts.fit_index,
            log_scale=log_scale,
            show=show,
        )

    def plot_param_correlations(
        self,
        threshold: float = 0.0,
        show: bool | None = None,
    ) -> Figure:
        """
        Heatmap of the posterior parameter correlation matrix.

        Args:
            threshold: Hide cells with |correlation| below this value.
            show: Same display convention as plot_results().

        Raises:
            ValueError: If the last fit was not Bayesian.
        """
        return plotting.plot_param_correlations(
            self.get_posterior(), threshold=threshold, show=show
        )

    def plot_trace(
        self,
        params: list[str] | None = None,
        show: bool | None = None,
    ) -> Figure:
        """
        Trace plot of the MCMC chains for each sampled parameter.

        Falls back to the combined chain when per-chain data is unavailable.

        Args:
            params: Optional subset of parameter names (default: all sampled).
            show: Same display convention as plot_results().

        Raises:
            ValueError: If the last fit was not Bayesian.
        """
        return plotting.plot_trace(self.get_posterior(), params=params, show=show)

    # =========================================================================
    # Theory preview (model evaluation without fitting)
    # =========================================================================

    def _require_data(self) -> None:
        if self.data is None:
            raise ValueError('No data loaded. Use load_data() first.')

    def _require_model(self) -> None:
        if self.kernel is None:
            raise ValueError('No model loaded. Use set_model() first.')

    def _theory_target(self, q: np.ndarray | None, dq: float | None) -> Any:
        """The dataset to evaluate on: an empty one on *q*, else the loaded data."""
        if q is not None:
            return theory_data(q, dq)
        if dq is not None:
            raise ValueError(
                "dq applies to an explicit q grid only; the loaded data's own "
                'resolution columns are used when q is omitted.'
            )
        self._require_data()
        # The same copy a fit is handed, so a preview under set_resolution('none' |
        # 'pinhole' | 'slit') is smeared exactly like the fit that follows it. Without
        # this the preview would apply the file's own columns and its chi-squared would
        # not match the "Initial chi-squared" a bumps fit prints. warn=False: the fit
        # itself warns about missing or shadowed columns.
        return self._evaluation_data(warn=False)

    def _evaluate(
        self, data: Any, overrides: dict[str, float] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the model on *data* at the current parameters (+ overrides)."""
        self._require_model()
        pars = build_model_parameters(self._param_manager.snapshot_fit_state(), overrides)
        return evaluate_theory(data, self.kernel, pars)

    def calculate(self, q: np.ndarray | None = None, dq: float | None = None) -> np.ndarray:
        """
        Evaluate the model at the current parameter values, without fitting.

        Args:
            q: Q values in Å⁻¹ to evaluate on. When omitted, the loaded
                dataset's Q values are used and its resolution (dQ or slit
                columns) is applied, exactly as during a fit.
            dq: Relative resolution width ΔQ/Q applied to the *q* grid.
                Only valid together with *q*; the dataset carries its own
                resolution.

        Returns:
            Intensities as a float64 array. On the data grid the result has
            one entry per data point, with NaN where a point is excluded from
            the fit (outside the Q range, masked, or NaN), so it can be
            plotted directly against ``fitter.data.x``.

        Raises:
            ValueError: If no model is loaded, if no data is loaded and *q* is
                omitted, if *dq* is given without *q*, or if *q*/*dq* are
                invalid.
        """
        curve, fit_index = self._evaluate(self._theory_target(q, dq))
        return scatter_to_full_length(curve, fit_index)

    def plot_model(
        self,
        show_residuals: bool = True,
        log_scale: bool = True,
        show: bool | None = None,
        show_components: bool = False,
    ) -> Figure:
        """
        Plot the data and the model at the current parameters, without fitting.

        The notebook equivalent of watching SasView redraw the theory as you
        change a parameter: it answers "are my starting values sane?" before
        committing to a fit. Fit results are untouched — ``plot_results()``
        keeps showing the last real fit.

        The reported goodness of fit is χ²/dof — the same convention every
        engine reports as ``result['reduced_chisq']``, and the same number the
        bumps engine prints as "Initial χ²" at the start of a fit. The model is
        evaluated through the active resolution mode, exactly as a fit would,
        so the preview and the fit that follows it are directly comparable. The
        value is reported as not available when the data carries no intensity
        uncertainties, and when the free parameters outnumber the fitted points.

        Args:
            show_residuals: If True, show residuals in a separate panel.
            log_scale: If True, use log scale for both axes.
            show: Same display convention as plot_results().
            show_components: If True and the model is a '+' mixture, overlay
                one dashed curve per component.

        Returns:
            Plotly Figure object

        Raises:
            ValueError: If no data or no model is loaded.
        """
        self._require_data()
        evaluation_data = self._evaluation_data(warn=False)
        curve, fit_index = self._evaluate(evaluation_data)

        fit_state = self._param_manager.snapshot_fit_state()
        n_free = len(fit_state.varying_params) + len(fit_state.varying_pd_params)
        quality = preview_quality(evaluation_data, curve, fit_index, n_free)

        # Current values only: nothing was estimated, so there is no stderr.
        contract = FitResultContract(
            engine=PREVIEW_ENGINE,
            method=PREVIEW_ENGINE,
            chisq=quality.chisq,
            reduced_chisq=quality.reduced_chisq,
            n_points=quality.n_points,
            n_free=quality.n_free,
            dof=quality.dof,
            weighting_note='dI',
            parameters={
                self._param_manager.to_display_name(name): {'value': info['value']}
                for name, info in fit_state.params.items()
            },
            resolution=self._resolution.as_dict(),
            artifacts=FitArtifacts(
                fitted_curve=curve,
                fit_index=fit_index,
                residuals=quality.residuals,
                component_curves=self._compute_component_curves() if show_components else None,
            ),
        )

        logger.info(
            f'{OK} Model preview: {self.model_name} — '
            f'{format_reduced_chisq(quality.reduced_chisq, quality.dof, CHI_SQUARED)} at current '
            f'parameters ({quality.n_points} points, {n_free} free)'
        )

        return plot_fit(
            data=self.data,
            fit_result=contract,
            model_name=self.model_name,
            show_residuals=show_residuals,
            log_scale=log_scale,
            show=show,
            show_components=show_components,
        )

    def compare(
        self,
        cases: dict[str, dict[str, float]] | None = None,
        q: np.ndarray | None = None,
        dq: float | None = None,
        log_scale: bool = True,
        show: bool | None = None,
        show_data: bool = True,
        **sweep: Sequence[float],
    ) -> Figure:
        """
        Overlay theory curves for several parameter sets on one plot.

        Each case starts from the current parameters and applies its own
        overrides; the fitter's own parameters are never changed.

        Args:
            cases: Label -> parameter overrides, e.g.
                ``{'thin': {'radius': 20}, 'thick': {'radius': 40}}``. An
                empty override dict means the current parameters. Override
                names accept aliases, canonical names, shared names and
                polydispersity widths (``radius_pd``).
            q: Q values to evaluate on. When omitted, the loaded dataset's Q
                values and resolution are used.
            dq: Relative resolution width ΔQ/Q for the *q* grid.
            log_scale: If True, use log scale for both axes.
            show: Same display convention as plot_results().
            show_data: If True and data is loaded, draw the measured points.
                With an explicit *q*, the data keeps its own Q grid.
            **sweep: One parameter name mapped to a sequence of values, as an
                alternative to *cases*: ``compare(radius=[20, 30, 40])``.

        Returns:
            Plotly Figure object

        Raises:
            ValueError: If no model is loaded, if neither or both spellings
                are used, if more than one sweep parameter is given, or if
                data is required but not loaded.
            KeyError: If an override names an unknown parameter.
        """
        self._require_model()

        if cases and sweep:
            raise ValueError('Pass either cases or a parameter sweep, not both.')
        if sweep:
            if len(sweep) > 1:
                raise ValueError(
                    f'Only one parameter can be swept at a time (got {", ".join(sweep)}). '
                    'Use the cases= form to vary several parameters.'
                )
            name, values = next(iter(sweep.items()))
            cases = {f'{name} = {value:g}': {name: value} for value in values}
        if not cases:
            raise ValueError(
                "Nothing to compare. Pass cases={'label': {...}} or a sweep such as "
                'radius=[20, 30, 40].'
            )

        target = self._theory_target(q, dq)
        curves: dict[str, np.ndarray] = {}
        for label, overrides in cases.items():
            curve, fit_index = self._evaluate(
                target, self._param_manager.canonical_overrides(overrides)
            )
            curves[label] = scatter_to_full_length(curve, fit_index)

        return plotting.plot_model_comparison(
            x=np.asarray(target.x),
            curves=curves,
            data=self.data if (show_data and self.data is not None) else None,
            model_name=self.model_name,
            log_scale=log_scale,
            show=show,
        )

    def plot_results(
        self,
        show_residuals: bool = True,
        log_scale: bool = True,
        show: bool | None = None,
        show_components: bool = False,
    ) -> Figure:
        """
        Plot experimental data and fitted model.

        Args:
            show_residuals: If True, show residuals in a separate panel
            log_scale: If True, use log scale for both axes
            show: If True, display the figure via fig.show(); if False, only
                return it. The default (None) displays the figure except in
                Jupyter notebooks, where the returned figure is rendered by
                the notebook itself (avoids showing the plot twice).
            show_components: If True and the fitted model is a '+' mixture,
                overlay one dashed curve per component (labelled by moniker).
                A documented no-op for atomic models and '*' mixtures.

        Returns:
            Plotly Figure object
        """
        return plot_fit(
            data=self.data,
            fit_result=self._get_active_fit_contract(),
            model_name=self.model_name,
            show_residuals=show_residuals,
            log_scale=log_scale,
            show=show,
            show_components=show_components,
        )

    def save_results(self, filename: str) -> None:
        """
        Save fit results to a file.

        Args:
            filename: Output file path (CSV format)
        """
        if self.fit_result is None:
            raise ValueError('No fit results to save. Run fit() first.')

        fit_contract = self._get_active_fit_contract()
        if fit_contract is None:
            raise ValueError('No fit results to save. Run fit() first.')

        save_fit_result(
            filename=filename,
            model_name=self.model_name,
            data=self.data,
            fit_result=fit_contract,
        )

        logger.info(f'{OK} Results saved to {filename}')

    # =========================================================================
    # Persistence and reporting
    # =========================================================================

    def save_analysis(self, filename: str, include_result: bool = True) -> None:
        """
        Save the complete analysis setup, and the last fit result, as JSON.

        What ``save_results()`` writes is the *outcome* of a fit. This writes
        how the fit was set up: the model expression and its component names,
        every parameter value, bound and vary flag, polydispersity, links, the
        structure factor, the resolution mode and the Q range. Reload it with
        :meth:`load_analysis`.

        JSON rather than pickle, so the file is readable, diffable, reviewable
        in a pull request, and safe to accept from a collaborator.

        **The result is only saved while it still describes the setup.** No
        setter clears a fit result, so a fitter can hold one produced by
        settings that have since changed (a parameter edited, the Q range
        restricted, the resolution switched, the data replaced). Saving the
        two together would pair a chi-squared with a configuration that never
        produced it. When they disagree, the setup is written, the result is
        left out, and the log line says which facet moved. Fit before you save,
        or save before you experiment.

        Args:
            filename: Output path for the JSON analysis file.
            include_result: When False, write the setup alone. Useful as a
                template: a configured model with no sample-specific outcome,
                ready to apply to the next dataset.

        Raises:
            AnalysisFileError: If no model is set, or the target directory
                does not exist. A subclass of ValueError.
        """
        document = write_analysis(self, filename, include_result=include_result)

        lines = [f'{OK} Analysis saved to {filename}']
        if document['result'] is not None:
            lines.append('  Fit result included')
        elif document['result_omitted'] is not None:
            lines.append(f'  Fit result NOT included: {document["result_omitted"]}')
        elif include_result:
            lines.append('  No fit result to include')
        else:
            lines.append('  Setup only (include_result=False)')
        logger.info('\n'.join(lines))

    @classmethod
    def load_analysis(
        cls,
        filename: str,
        data: Any = None,
        allow_custom_models: bool = False,
    ) -> 'SANSFitter':
        """
        Rebuild a fitter from a file written by :meth:`save_analysis`.

        The data file is looked for next to the analysis file first (by the
        relative path recorded when it was saved) and then at its original
        absolute path, so an analysis that travelled together with its data
        loads on another machine.

        Args:
            filename: Path to the JSON analysis file.
            data: Dataset to use instead of the recorded one: a path, or an
                in-memory dataset. Required for an analysis saved from
                ``set_data()``, whose dataset has no file to reload. An
                explicit value that cannot be used is an error rather than a
                silent fall back to the recorded sample.
            allow_custom_models: Permit a model expression that is not built
                into sasmodels. Off by default: loading such an expression
                imports a plugin module named by the file, so it is code
                execution chosen by whoever wrote the file, not by you.

        Returns:
            A configured fitter. When the saved result still describes the
            restored setup and the same data, it is attached too, so
            ``plot_results()``, ``save_results()`` and ``get_fit_report()``
            work without refitting.

        Raises:
            AnalysisFileError: If the file is missing, malformed, written by a
                newer schema, describes a different model, or its data cannot
                be found. A subclass of ValueError.
        """
        return read_analysis(cls, filename, data=data, allow_custom_models=allow_custom_models)

    def report(
        self,
        filename: str | None = None,
        fmt: str | None = None,
        offline: bool = False,
    ) -> Report:
        """
        Render a shareable report: settings, result tables and the fit plot.

        One document holding everything a colleague needs to judge the fit: the
        model and data it used, how it was smeared and weighted, the Q range,
        the goodness-of-fit and parameter tables from
        :meth:`get_fit_report`, and the plot. After ``fit_bayesian()`` the
        posterior summary comes along with it.

        Before any fit this produces a configuration report instead of raising:
        the settings and the current parameter values, with a theory preview in
        place of the fit plot. That is also what an analysis loaded with
        ``include_result=False`` renders.

        **HTML needs nothing beyond the standard dependencies; Markdown needs
        an image renderer.** A Markdown report references a sidecar PNG, named
        after the report file so two reports in one directory cannot overwrite
        each other's figure. With no usable renderer the figure is left out and
        a warning says how to install one. HTML embeds the interactive plot
        either way.

        Args:
            filename: Where to write. The extension chooses the format
                (``.html``/``.htm`` or ``.md``/``.markdown``). When omitted,
                nothing is written and the report is only returned.
            fmt: Format override, ``'html'`` or ``'markdown'``. Required to
                pick a format when *filename* is omitted; defaults to HTML.
            offline: Embed the Plotly library in the HTML instead of loading it
                from a CDN. Produces a much larger file that needs no network.

        Returns:
            A Report. ``str()`` gives the Markdown, ``to_html()`` the HTML, and
            a notebook renders it directly.

        Raises:
            ValueError: If the extension does not name a supported format, or
                the target directory does not exist.
        """
        resolved = fmt or (format_for(filename) if filename else 'html')
        if resolved not in ('html', 'markdown'):
            raise ValueError(f"Unknown report format '{resolved}'. Use 'html' or 'markdown'.")

        stem = os.path.splitext(os.path.basename(filename))[0] if filename else 'report'
        report = render_report(self, offline=offline, asset_stem=stem)

        if resolved == 'markdown' and filename is None and report.assets:
            # Nowhere to put the sidecar, so the figure cannot be referenced.
            report = render_report(self, offline=offline, asset_stem=stem, include_figure=False)

        warn_if_no_image(report, resolved)
        if filename is not None:
            report.write(filename, fmt=resolved)
            logger.info(f'{OK} Report written to {filename}')
        return report

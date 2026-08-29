"""
SANS Model Fitter - A flexible template for fitting SANS data with SasModels

This module provides a unified interface for fitting SANS data using different
optimization engines (BUMPS, LMFit) with any model from the SasModels library.
"""

import difflib
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
from .data.loader import get_fit_index, has_real_data, load_sans_data, normalize_sans_data
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
from .fitting.base import extract_fit_index, pd_is_active
from .modeling.parameters import ParameterManager
from .plotting import DEFAULT_POSTERIOR_PREDICTIVE_DRAWS, plot_fit
from .results import FitArtifacts, FitResultContract, PosteriorSummary, save_fit_result


def get_all_models() -> list[str]:
    """
    Fetch all available models from sasmodels.

    Returns:
        List of model names
    """
    try:
        all_models = core.list_models()
        return sorted(all_models)
    except Exception as e:
        print(f'Error fetching models: {str(e)}')
        return []


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
        self._fit_contract: FitResultContract | None = None
        self._fitted_model = None
        self._full_q_range: tuple[float, float] | None = None

        # Parameter management delegated to ParameterManager
        self._param_manager = ParameterManager()

    def load_data(self, filename: str) -> None:
        """
        Load SANS data from a file.

        Supports CSV, XML, and HDF5 formats through sasdata. Columnar text/CSV
        files are interpreted in the order Q, I, dI, dQ (per the sasdata ASCII
        convention) — a file whose third column is dQ rather than dI will have
        its uncertainties and resolution swapped. Check the column summary
        printed after loading.

        Args:
            filename: Path to the data file

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the data cannot be loaded or is invalid
        """
        self.data = load_sans_data(filename)
        self._full_q_range = (self.data.qmin, self.data.qmax)

        has_dy = has_real_data(self.data.dy)
        has_dx = has_real_data(self.data.dx)

        print(f'✓ Loaded data from {filename}')
        print(f'  Q range: {self.data.qmin:.4f} to {self.data.qmax:.4f} Å⁻¹')
        print(f'  Data points: {len(self.data.x)}')
        print(f'  Error (dI) column: {"yes" if has_dy else "no"}')
        print(f'  Resolution (dQ) column: {"yes" if has_dx else "no"}')

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
        self._full_q_range = (self.data.qmin, self.data.qmax)

        has_dy = has_real_data(self.data.dy)
        has_dx = has_real_data(self.data.dx)
        label = getattr(data, 'title', '') or getattr(data, 'filename', '') or 'in-memory dataset'

        print(f'✓ Data set: {label}')
        print(f'  Q range: {self.data.qmin:.4f} to {self.data.qmax:.4f} Å⁻¹')
        print(f'  Data points: {len(self.data.x)}')
        print(f'  Error (dI) column: {"yes" if has_dy else "no"}')
        print(f'  Resolution (dQ) column: {"yes" if has_dx else "no"}')

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

        print(f'✓ Q range for fitting: {new_qmin:.6g} to {new_qmax:.6g} Å⁻¹')
        print(f'  Points in fit: {n_points} of {len(index)}')

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
        print(f'✓ Q range reset to {self.data.qmin:.6g} to {self.data.qmax:.6g} Å⁻¹')
        print(f'  Points in fit: {n_points}')

    def get_q_range(self) -> tuple[float, float] | None:
        """
        Get the Q range currently used for fitting.

        Returns:
            Tuple (qmin, qmax) in Å⁻¹, or None if no data is loaded.
        """
        if self.data is None:
            return None
        return (self.data.qmin, self.data.qmax)

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

            print(f"✓ Model '{model_name}' loaded successfully")
            print(f'  Available parameters: {len(self._param_manager.params)}')

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

        print(f'✓ Combined {len(components)} models: {expression}')
        print(f'  Components: {", ".join(m for m, _n in components)}')
        if shared:
            print(f'  Shared parameters: {", ".join(shared)}')
        print(f'  Available parameters: {len(self._param_manager.params)}')

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
        print(f'✓ Linked {name} → {to}')

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
        print(f'✓ Unlinked {name}')

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

        Supported structure factors:
        - 'hardsphere': Hard sphere structure factor (Percus-Yevick closure)
        - 'hayter_msa': Hayter-Penfold rescaled MSA for charged spheres
        - 'squarewell': Square well potential
        - 'stickyhardsphere': Sticky hard sphere (Baxter model)

        Args:
            structure_factor_name: Name of the structure factor (e.g., 'hardsphere')
            radius_effective_mode: How to handle the effective radius.
                - 'unconstrained': 'radius_effective' is a separate fitting parameter.
                - 'link_radius': 'radius_effective' is constrained to the form factor's 'radius'.

        Raises:
            ValueError: If no form factor model is set, or if the structure factor is invalid
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

        # Validate structure factor name
        supported_sf = ['hardsphere', 'hayter_msa', 'squarewell', 'stickyhardsphere']
        if structure_factor_name not in supported_sf:
            raise ValueError(
                f"Unsupported structure factor '{structure_factor_name}'. "
                f'Supported: {", ".join(supported_sf)}'
            )

        # Create product model name
        full_model_name = f'{self.model_name}@{structure_factor_name}'

        try:
            # Load the product model
            self.kernel = load_model(full_model_name, dtype='single', platform='dll')

            # Delegate parameter management to ParameterManager
            self._param_manager.update_for_product_model(
                self.kernel, structure_factor_name, radius_effective_mode
            )

            if radius_effective_mode == 'link_radius':
                print("  Note: 'radius_effective' linked to 'radius' value")

            print(f"✓ Structure factor '{structure_factor_name}' applied to '{self.model_name}'")
            print(f'  Product model: {full_model_name}')
            print(f'  Total parameters: {len(self.params)}')

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

        # Reload the original form factor model
        try:
            self.kernel = load_model(self.model_name, dtype='single', platform='dll')

            # Delegate to ParameterManager - this restores params and PD state
            sf_name = self._param_manager.remove_structure_factor()

            print(f"✓ Structure factor '{sf_name}' removed")
            print(f'  Reverted to form factor: {self.model_name}')

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

        # Translate engine result names (canonical) back to user-facing names
        # so saved results and displays never expose A_/B_ on the set_models
        # path (Boundary 2 of the alias layer).
        self._fit_contract.parameters = {
            self._param_manager.to_display_name(name): dict(info)
            for name, info in self._fit_contract.parameters.items()
        }

        # Attach per-component curves for '+' mixture models (no-op otherwise).
        component_curves = self._compute_component_curves()
        if component_curves:
            self._fit_contract.artifacts.component_curves = component_curves

        self.fit_result = self._fit_contract.to_legacy_dict()
        self._fitted_model = engine_output.runtime_model

        print('\n✓ Fit completed!')
        print(f'Final χ² = {self.fit_result["chisq"]:.4f}')
        print('\nFitted parameters:')
        for name, info in self.fit_result['parameters'].items():
            print(f'  {name}: {info["formatted"]}')

        posterior = self._fit_contract.artifacts.posterior
        if posterior is not None:
            print()
            print(posterior.format_summary())

        return self.fit_result

    def _compute_component_curves(self) -> dict[str, np.ndarray] | None:
        """Compute per-component curves after a fit of a '+' mixture model.

        Each component curve is ``scale · I_part(q, scale=part_scale,
        background=0)`` — matching the mixture kernel's own computation — so
        the component curves plus the background stack onto the total curve.

        Returns None for atomic models and '*' mixtures (where part curves
        would not stack to the total and would mislead when overlaid).
        Evaluation happens on the same masked q-points as the total curve.
        """
        components = self._param_manager.get_components()
        if not components:
            return None

        # '*' mixtures: part intensities multiply, so additive component
        # curves are meaningless. Documented no-op.
        operation = getattr(self.kernel.info, 'operation', '+')
        if operation != '+':
            return None

        canonical_values = self._param_manager.get_canonical_param_values()
        global_scale = canonical_values.get('scale', 1.0)

        # Active polydispersity settings, keyed by canonical base names.
        pd_settings: dict[str, dict[str, Any]] = {}
        if self._param_manager.is_pd_enabled():
            for base_param in self._param_manager.get_polydisperse_parameters():
                pd_config = self._param_manager.polydisperse_params[base_param]
                if pd_is_active(pd_config):
                    pd_settings[base_param] = pd_config

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
            calculator = DirectModel(self.data, part_kernel)

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

    def _get_active_fit_contract(self) -> FitResultContract | None:
        """Return the active fit contract, adapting legacy runtime state if needed."""
        if self._fit_contract is not None:
            return self._fit_contract

        if self.fit_result is None:
            return None

        if self.fit_result['engine'] == 'bumps':
            return FitResultContract(
                engine=self.fit_result['engine'],
                method=self.fit_result['method'],
                chisq=self.fit_result['chisq'],
                parameters=self.fit_result['parameters'],
                artifacts=FitArtifacts(
                    fitted_curve=np.asarray(self._fitted_model.active_model.theory()),
                    fit_index=extract_fit_index(self._fitted_model.active_model),
                ),
            )

        calculator = DirectModel(self.data, self.kernel)
        par_dict = {name: info['value'] for name, info in self.fit_result['parameters'].items()}
        return FitResultContract(
            engine=self.fit_result['engine'],
            method=self.fit_result['method'],
            chisq=self.fit_result['chisq'],
            parameters=self.fit_result['parameters'],
            artifacts=FitArtifacts(
                fitted_curve=np.asarray(calculator(**par_dict)),
                fit_index=extract_fit_index(calculator),
            ),
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
            Dictionary with fit results including chi-squared and parameter values

        Raises:
            ValueError: If data or model not loaded, or invalid engine
            NotImplementedError: If a composite model is used with an engine
                other than 'bumps'.
        """
        if self.data is None:
            raise ValueError('No data loaded. Use load_data() first.')
        if self.kernel is None:
            raise ValueError('No model loaded. Use set_model() first.')

        if engine not in ('bumps', 'lmfit'):
            raise ValueError(f"Unknown engine '{engine}'. Use 'bumps' or 'lmfit'.")

        self._check_composite_engine_support(engine)
        self._check_scale_degeneracy()
        self._check_fit_uncertainties(engine)

        if engine == 'bumps':
            return self._fit_bumps(method or 'amoeba', **kwargs)
        if not LMFIT_AVAILABLE:
            raise ValueError("scipy is not installed. Use 'bumps' engine or install scipy.")
        return self._fit_lmfit(method or 'leastsq', **kwargs)

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
            data=self.data,
            kernel=self.kernel,
            fit_state=self._param_manager.snapshot_fit_state(),
            method=method,
            **kwargs,
        )
        return self._finalize_fit(engine_output)

    def _fit_lmfit(self, method: str = 'leastsq', **kwargs: Any) -> dict[str, Any]:
        """Fit using scipy.optimize (leastsq/least_squares) engine."""
        engine_output = fit_scipy(
            data=self.data,
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
            Dictionary with fit results including chi-squared and parameter
            values. The posterior itself is available via get_posterior().

        Raises:
            ValueError: If data or model is not loaded, or no parameter varies.
            NotImplementedError: If a composite model is used — the DREAM path
                does not support them yet.
        """
        if self.data is None:
            raise ValueError('No data loaded. Use load_data() first.')
        if self.kernel is None:
            raise ValueError('No model loaded. Use set_model() first.')

        snapshot = self._param_manager.snapshot_fit_state()
        if snapshot.components:
            raise NotImplementedError(
                "Composite models are currently supported by the 'bumps' "
                "point-estimate engine only (fit(engine='bumps'))."
            )
        self._check_scale_degeneracy()

        engine_output = fit_bumps_dream(
            data=self.data,
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

        print(f'✓ Results saved to {filename}')

"""
SANS Model Fitter - A flexible template for fitting SANS data with SasModels

This module provides a unified interface for fitting SANS data using different
optimization engines (BUMPS, LMFit) with any model from the SasModels library.
"""

import warnings
from typing import Any, Literal, Optional

import numpy as np

# SasModels and SasData imports
from sasmodels import core
from sasmodels.core import load_model
from sasmodels.direct_model import DirectModel

from .data_loader import _has_real_data, get_fit_index, load_sans_data
from .fitting import SCIPY_AVAILABLE, fit_bumps, fit_scipy
from .fitting.base import extract_fit_index
from .parameter_manager import ParameterManager
from .plotting import plot_fit
from .results import FitArtifacts, FitResultContract, save_fit_result


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
        self._fit_contract: Optional[FitResultContract] = None
        self._fitted_model = None
        self._full_q_range: Optional[tuple[float, float]] = None

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

        has_dy = _has_real_data(self.data.dy)
        has_dx = _has_real_data(self.data.dx)

        print(f'✓ Loaded data from {filename}')
        print(f'  Q range: {self.data.qmin:.4f} to {self.data.qmax:.4f} Å⁻¹')
        print(f'  Data points: {len(self.data.x)}')
        print(f'  Error (dI) column: {"yes" if has_dy else "no"}')
        print(f'  Resolution (dQ) column: {"yes" if has_dx else "no"}')

    def set_q_range(self, qmin: Optional[float] = None, qmax: Optional[float] = None) -> None:
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

    def get_q_range(self) -> Optional[tuple[float, float]]:
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

        This resets any active structure factor to ensure a clean state.

        Args:
            model_name: Name of the model from SasModels (e.g., 'cylinder', 'sphere')
            platform: Computation platform ('cpu' or 'opencl')

        Raises:
            ValueError: If the model name is not valid
        """
        try:
            # Force CPU platform to avoid OpenCL issues
            self.kernel = load_model(model_name, dtype='single', platform='dll')

            # Initialize parameters via ParameterManager
            self._param_manager.initialize_from_kernel(self.kernel, model_name)

            print(f"✓ Model '{model_name}' loaded successfully")
            print(f'  Available parameters: {len(self._param_manager.params)}')

        except Exception as e:
            raise ValueError(f"Failed to load model '{model_name}': {str(e)}") from e

    # =========================================================================
    # Property accessors for backward compatibility
    # =========================================================================

    @property
    def model_name(self) -> Optional[str]:
        """Get the current model name."""
        return self._param_manager.model_name

    @model_name.setter
    def model_name(self, value: Optional[str]) -> None:
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
    def _structure_factor_name(self) -> Optional[str]:
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
        value: Optional[float] = None,
        min: Optional[float] = None,
        max: Optional[float] = None,
        vary: Optional[bool] = None,
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

    def get_structure_factor(self) -> Optional[str]:
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
        pd_width: Optional[float] = None,
        pd_n: Optional[int] = None,
        pd_nsigma: Optional[float] = None,
        pd_type: Optional[str] = None,
        vary: Optional[bool] = None,
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
        varying_base = self._param_manager.get_varying_pd_params()
        return [f'{param_name}_pd' for param_name in varying_base]

    def _finalize_fit(self, engine_output) -> dict[str, Any]:
        """Apply engine output to fitter state and return legacy-compatible results."""
        self._param_manager.apply_fitted_values(engine_output.fitted_values)
        self._fit_contract = engine_output.contract
        self.fit_result = self._fit_contract.to_legacy_dict()
        self._fitted_model = engine_output.runtime_model

        print('\n✓ Fit completed!')
        print(f'Final χ² = {self.fit_result["chisq"]:.4f}')
        print('\nFitted parameters:')
        for name, info in self.fit_result['parameters'].items():
            print(f'  {name}: {info["formatted"]}')

        return self.fit_result

    def _get_active_fit_contract(self) -> Optional[FitResultContract]:
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
                    fitted_curve=np.asarray(self._fitted_model.fitness.theory()),
                    fit_index=extract_fit_index(self._fitted_model.fitness),
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
        method: Optional[str] = None,
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
        """
        if self.data is None:
            raise ValueError('No data loaded. Use load_data() first.')
        if self.kernel is None:
            raise ValueError('No model loaded. Use set_model() first.')

        if engine == 'bumps':
            return self._fit_bumps(method or 'amoeba', **kwargs)
        elif engine == 'lmfit':
            if not LMFIT_AVAILABLE:
                raise ValueError("scipy is not installed. Use 'bumps' engine or install scipy.")
            return self._fit_lmfit(method or 'leastsq', **kwargs)
        else:
            raise ValueError(f"Unknown engine '{engine}'. Use 'bumps' or 'lmfit'.")

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

    def plot_results(
        self,
        show_residuals: bool = True,
        log_scale: bool = True,
        show: bool | None = None,
    ):
        """
        Plot experimental data and fitted model.

        Args:
            show_residuals: If True, show residuals in a separate panel
            log_scale: If True, use log scale for both axes
            show: If True, display the figure via fig.show(); if False, only
                return it. The default (None) displays the figure except in
                Jupyter notebooks, where the returned figure is rendered by
                the notebook itself (avoids showing the plot twice).

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

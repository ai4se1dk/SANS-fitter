"""
SANS Model Fitter - A flexible template for fitting SANS data with SasModels

This module provides a unified interface for fitting SANS data using different
optimization engines (BUMPS, LMFit) with any model from the SasModels library.
"""

import warnings
from typing import Any, Literal, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sasdata.dataloader.loader import Loader
from sasmodels.core import load_model
from sasmodels.direct_model import DirectModel

from .bumps_engine import BumpsFittingEngine
from .parameter_manager import ParameterManager

try:
    from .scipy_engine import ScipyFittingEngine

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Keep scipy.optimize for backwards compatibility (even though not directly used)
try:
    from scipy.optimize import differential_evolution, least_squares, leastsq  # noqa: F401

    LMFIT_AVAILABLE = True
except ImportError:
    LMFIT_AVAILABLE = False
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
        self.model_name = None
        self.param_manager = ParameterManager()
        self.fit_result = None
        self._fitted_model = None

        # Initialize fitting engines
        self._fitting_engines = {'bumps': BumpsFittingEngine()}
        if SCIPY_AVAILABLE:
            self._fitting_engines['scipy'] = ScipyFittingEngine()
            self._fitting_engines['lmfit'] = self._fitting_engines['scipy']  # Alias

    @property
    def params(self) -> dict[str, dict[str, Any]]:
        """
        Get parameter dictionary (for backward compatibility).

        Returns:
            Dictionary of parameter configurations
        """
        return self.param_manager.params

    def load_data(self, filename: str) -> None:
        """
        Load SANS data from a file.

        Supports CSV, XML, and HDF5 formats through sasdata.

        Args:
            filename: Path to the data file

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the data cannot be loaded or is invalid
        """
        loader = Loader()
        try:
            data_list = loader.load(filename)
            if not data_list:
                raise ValueError(f'No data loaded from {filename}')

            self.data = data_list[0]

            # Setup required fields for sasmodels
            self.data.qmin = getattr(self.data, 'qmin', None) or self.data.x.min()
            self.data.qmax = getattr(self.data, 'qmax', None) or self.data.x.max()
            self.data.mask = np.isnan(self.data.y)

            print(f'✓ Loaded data from {filename}')
            print(f'  Q range: {self.data.qmin:.4f} to {self.data.qmax:.4f} Å⁻¹')
            print(f'  Data points: {len(self.data.x)}')

        except Exception as e:
            raise ValueError(f'Failed to load data from {filename}: {str(e)}') from e

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
            self.model_name = model_name

            # Initialize parameters using ParameterManager (this also resets structure factor)
            self.param_manager.clear()
            self.param_manager.initialize_from_kernel(self.kernel, model_name)

            print(f"✓ Model '{model_name}' loaded successfully")
            print(f'  Available parameters: {len(self.param_manager.params)}')

        except Exception as e:
            raise ValueError(f"Failed to load model '{model_name}': {str(e)}") from e

    def get_params(self) -> None:
        """Display current parameter values and settings in a readable format."""
        self.param_manager.display_params()

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
        self.param_manager.set_param(name, value=value, min=min, max=max, vary=vary)

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

            # Update parameters using ParameterManager
            self.param_manager.update_for_product_model(
                self.kernel, structure_factor_name, radius_effective_mode
            )

            if radius_effective_mode == 'link_radius':
                if (
                    'radius' in self.param_manager.params
                    and 'radius_effective' in self.param_manager.params
                ):
                    print("  Note: 'radius_effective' linked to 'radius' value")

            print(f"✓ Structure factor '{structure_factor_name}' applied to '{self.model_name}'")
            print(f'  Product model: {full_model_name}')
            print(f'  Total parameters: {len(self.param_manager.params)}')

        except Exception as e:
            raise ValueError(f"Failed to load model '{full_model_name}': {str(e)}") from e

    def remove_structure_factor(self) -> None:
        """
        Remove the current structure factor and revert to the form factor only.

        Raises:
            ValueError: If no structure factor is currently set
        """
        # Reload the original form factor model
        try:
            sf_name = self.param_manager.remove_structure_factor()
            self.kernel = load_model(self.model_name, dtype='single', platform='dll')

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
        return self.param_manager.get_structure_factor()

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
                   - LMFit/Scipy: 'leastsq', 'least_squares', 'differential_evolution', etc.
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

        # Get the appropriate fitting engine
        if engine not in self._fitting_engines:
            available = ', '.join(self._fitting_engines.keys())
            raise ValueError(f"Unknown engine '{engine}'. Available engines: {available}")

        fitting_engine = self._fitting_engines[engine]

        # Set default method if not specified
        if method is None:
            method = 'amoeba' if engine == 'bumps' else 'leastsq'

        # Perform fit using the strategy
        # Pass engine_name for scipy to maintain backward compatibility
        if engine in ['scipy', 'lmfit']:
            self.fit_result = fitting_engine.fit(
                data=self.data,
                kernel=self.kernel,
                param_manager=self.param_manager,
                method=method,
                engine_name=engine,  # Pass the original engine name for compatibility
                **kwargs,
            )
        else:
            self.fit_result = fitting_engine.fit(
                data=self.data,
                kernel=self.kernel,
                param_manager=self.param_manager,
                method=method,
                **kwargs,
            )

        # Store fitted model for later use
        if 'problem' in self.fit_result:
            self._fitted_model = self.fit_result['problem']
        elif 'result' in self.fit_result:
            self._fitted_model = self.fit_result['result']

        return self.fit_result

    def plot_results(self, show_residuals: bool = True, log_scale: bool = True) -> go.Figure:
        """
        Plot experimental data and fitted model.

        Args:
            show_residuals: If True, show residuals in a separate panel
            log_scale: If True, use log scale for both axes

        Returns:
            Plotly Figure object
        """
        if self.data is None:
            raise ValueError('No data to plot. Use load_data() first.')

        if self.fit_result is None:
            print('No fit results available. Plotting data only.')
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=self.data.x,
                    y=self.data.y,
                    error_y={'type': 'data', 'array': self.data.dy, 'visible': True},
                    mode='markers',
                    name='Data',
                    opacity=0.6,
                )
            )
            fig.update_layout(
                title='SANS Data',
                xaxis_title='Q (Å⁻¹)',
                yaxis_title='I(Q)',
                xaxis_type='log' if log_scale else 'linear',
                yaxis_type='log' if log_scale else 'linear',
                template='plotly_white',
            )
            fig.show()
            return fig

        # Calculate fitted curve using the appropriate engine
        engine_name = self.fit_result['engine']
        if engine_name == 'lmfit':
            engine_name = 'scipy'  # Map lmfit to scipy

        fitting_engine = self._fitting_engines.get(engine_name)
        if fitting_engine:
            q, I_fit = fitting_engine.get_fitted_curve(self.fit_result, self.data, self.kernel)
        else:
            # Fallback for backward compatibility
            calculator = DirectModel(self.data, self.kernel)
            par_dict = {name: info['value'] for name, info in self.fit_result['parameters'].items()}
            I_fit = calculator(**par_dict)
            q = self.data.x

        residuals = (self.data.y - I_fit) / self.data.dy

        # Create plot
        if show_residuals:
            fig = make_subplots(
                rows=2,
                cols=1,
                row_heights=[0.75, 0.25],
                shared_xaxes=True,
                vertical_spacing=0.05,
            )
        else:
            fig = go.Figure()

        # Main plot - experimental data with error bars
        data_trace = go.Scatter(
            x=self.data.x,
            y=self.data.y,
            error_y={'type': 'data', 'array': self.data.dy, 'visible': True},
            mode='markers',
            name='Experimental Data',
            opacity=0.6,
            marker={'size': 6},
        )

        # Fitted model line
        fit_trace = go.Scatter(
            x=q,
            y=I_fit,
            mode='lines',
            name='Fitted Model',
            line={'color': 'red', 'width': 2},
        )

        if show_residuals:
            fig.add_trace(data_trace, row=1, col=1)
            fig.add_trace(fit_trace, row=1, col=1)

            # Residuals plot
            fig.add_trace(
                go.Scatter(
                    x=self.data.x,
                    y=residuals,
                    mode='markers',
                    name='Residuals',
                    marker={'size': 6},
                    opacity=0.6,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

            # Add zero line for residuals
            fig.add_hline(y=0, line_dash='dash', line_color='gray', row=2, col=1)

            # Update axes
            fig.update_xaxes(
                title_text='Q (Å⁻¹)',
                type='log' if log_scale else 'linear',
                row=2,
                col=1,
            )
            fig.update_yaxes(
                title_text='I(Q)',
                type='log' if log_scale else 'linear',
                row=1,
                col=1,
            )
            fig.update_yaxes(title_text='Residuals (σ)', row=2, col=1)
            fig.update_xaxes(type='log' if log_scale else 'linear', row=1, col=1)
        else:
            fig.add_trace(data_trace)
            fig.add_trace(fit_trace)
            fig.update_xaxes(
                title_text='Q (Å⁻¹)',
                type='log' if log_scale else 'linear',
            )
            fig.update_yaxes(
                title_text='I(Q)',
                type='log' if log_scale else 'linear',
            )

        fig.update_layout(
            title=f'SANS Fit: {self.model_name} (χ² = {self.fit_result["chisq"]:.4f})',
            template='plotly_white',
            height=800 if show_residuals else 500,
            width=900,
        )

        fig.show()
        return fig

    def save_results(self, filename: str) -> None:
        """
        Save fit results to a file.

        Args:
            filename: Output file path (CSV format)
        """
        if self.fit_result is None:
            raise ValueError('No fit results to save. Run fit() first.')

        # Prepare data
        with open(filename, 'w') as f:
            f.write('# SANS Fit Results\n')
            f.write(f'# Model: {self.model_name}\n')
            f.write(f'# Engine: {self.fit_result["engine"]}\n')
            f.write(f'# Method: {self.fit_result["method"]}\n')
            f.write(f'# Chi-squared: {self.fit_result["chisq"]:.6f}\n')
            f.write('#\n')
            f.write('# Fitted Parameters:\n')
            for name, info in self.fit_result['parameters'].items():
                f.write(f'# {name}: {info["formatted"]}\n')
            f.write('#\n')
            f.write('Q,I_exp,dI_exp,I_fit,Residuals\n')

            # Get fitted curve using the appropriate engine
            engine_name = self.fit_result['engine']
            if engine_name == 'lmfit':
                engine_name = 'scipy'  # Map lmfit to scipy

            fitting_engine = self._fitting_engines.get(engine_name)
            if fitting_engine:
                _, I_fit = fitting_engine.get_fitted_curve(self.fit_result, self.data, self.kernel)
            else:
                # Fallback for backward compatibility
                calculator = DirectModel(self.data, self.kernel)
                par_dict = {
                    name: info['value'] for name, info in self.fit_result['parameters'].items()
                }
                I_fit = calculator(**par_dict)

            residuals = (self.data.y - I_fit) / self.data.dy

            for q, i_exp, di_exp, i_fit, res in zip(
                self.data.x, self.data.y, self.data.dy, I_fit, residuals
            ):
                f.write(f'{q:.6e},{i_exp:.6e},{di_exp:.6e},{i_fit:.6e},{res:.6e}\n')

        print(f'✓ Results saved to {filename}')

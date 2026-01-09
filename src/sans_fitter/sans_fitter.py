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
from bumps.fitters import fit as bumps_fit
from bumps.formatnum import format_uncertainty

# Fitting engine imports
from bumps.names import FitProblem
from sasdata.dataloader.loader import Loader
from sasmodels.bumps_model import Experiment
from sasmodels.bumps_model import Model as BumpsModel

# SasModels and SasData imports
from sasmodels.core import load_model
from sasmodels.direct_model import DirectModel

try:
    from scipy.optimize import differential_evolution, least_squares, leastsq

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
        self.params = {}
        self.fit_result = None
        self._fitted_model = None

        # Structure factor support
        self._structure_factor_name = None
        self._radius_effective_mode = 'unconstrained'
        self._form_factor_params = {}  # Store form factor params separately

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
            # Reset structure factor when changing form factor
            self._structure_factor_name = None
            self._radius_effective_mode = 'unconstrained'
            self._form_factor_params = {}

            # Force CPU platform to avoid OpenCL issues
            self.kernel = load_model(model_name, dtype='single', platform='dll')
            self.model_name = model_name

            # Initialize parameters with default values from the model
            self.params = {}
            for param in self.kernel.info.parameters.kernel_parameters:
                self.params[param.name] = {
                    'value': param.default,
                    'min': param.limits[0] if param.limits[0] > -np.inf else 0,
                    'max': param.limits[1] if param.limits[1] < np.inf else param.default * 10,
                    'vary': False,  # By default, parameters are fixed
                    'description': param.description,
                }

            # Add implicit scale and background parameters (present in all models)
            # These are not in kernel_parameters but are always available
            if 'scale' not in self.params:
                self.params['scale'] = {
                    'value': 1.0,
                    'min': 0.0,
                    'max': np.inf,
                    'vary': False,
                    'description': 'Scale factor for the model intensity',
                }

            if 'background' not in self.params:
                self.params['background'] = {
                    'value': 0.0,
                    'min': 0.0,
                    'max': np.inf,
                    'vary': False,
                    'description': 'Constant background level',
                }

            print(f"✓ Model '{model_name}' loaded successfully")
            print(f'  Available parameters: {len(self.params)}')

        except Exception as e:
            raise ValueError(f"Failed to load model '{model_name}': {str(e)}") from e

    def get_params(self) -> None:
        """Display current parameter values and settings in a readable format."""
        if not self.params:
            print('No model loaded. Use set_model() first.')
            return

        print(f'\n{"=" * 80}')
        print(f'Model: {self.model_name}')
        if self._structure_factor_name:
            print(f'Structure Factor: {self._structure_factor_name}')
            print(f'Radius Effective Mode: {self._radius_effective_mode}')
        print(f'{"=" * 80}')
        print(f'{"Parameter":<20} {"Value":<12} {"Min":<12} {"Max":<12} {"Vary":<8}')
        print(f'{"-" * 80}')

        for name, info in self.params.items():
            vary_str = '✓' if info['vary'] else '✗'
            # Show linked indicator for radius_effective in link_radius mode
            if name == 'radius_effective' and self._radius_effective_mode == 'link_radius':
                vary_str = '→radius'
            print(
                f'{name:<20} {info["value"]:<12.4g} {info["min"]:<12.4g} '
                f'{info["max"]:<12.4g} {vary_str:<8}'
            )
        print(f'{"=" * 80}\n')

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
        if name not in self.params:
            available = ', '.join(self.params.keys())
            raise KeyError(f"Parameter '{name}' not found. Available: {available}")

        if value is not None:
            self.params[name]['value'] = value
            # Sync radius_effective when radius is updated in link_radius mode
            if (
                name == 'radius'
                and self._radius_effective_mode == 'link_radius'
                and 'radius_effective' in self.params
            ):
                self.params['radius_effective']['value'] = value
        if min is not None:
            self.params[name]['min'] = min
        if max is not None:
            self.params[name]['max'] = max
        if vary is not None:
            self.params[name]['vary'] = vary

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

        # Validate radius_effective_mode
        if radius_effective_mode not in ['unconstrained', 'link_radius']:
            raise ValueError(
                f"Invalid radius_effective_mode '{radius_effective_mode}'. "
                "Use 'unconstrained' or 'link_radius'."
            )

        # Store form factor parameters before switching to product model
        if not self._form_factor_params:
            self._form_factor_params = {k: dict(v) for k, v in self.params.items()}

        # Create product model name
        full_model_name = f'{self.model_name}@{structure_factor_name}'

        try:
            # Load the product model
            self.kernel = load_model(full_model_name, dtype='single', platform='dll')
            self._structure_factor_name = structure_factor_name
            self._radius_effective_mode = radius_effective_mode

            # Rebuild parameters from product model
            new_params = {}
            for param in self.kernel.info.parameters.kernel_parameters:
                # Preserve existing values if parameter already exists
                if param.name in self._form_factor_params:
                    new_params[param.name] = dict(self._form_factor_params[param.name])
                else:
                    new_params[param.name] = {
                        'value': param.default,
                        'min': param.limits[0] if param.limits[0] > -np.inf else 0,
                        'max': param.limits[1] if param.limits[1] < np.inf else param.default * 10,
                        'vary': False,
                        'description': param.description,
                    }

            # Ensure scale and background are present
            if 'scale' not in new_params:
                if 'scale' in self._form_factor_params:
                    new_params['scale'] = dict(self._form_factor_params['scale'])
                else:
                    new_params['scale'] = {
                        'value': 1.0,
                        'min': 0.0,
                        'max': np.inf,
                        'vary': False,
                        'description': 'Scale factor for the model intensity',
                    }

            if 'background' not in new_params:
                if 'background' in self._form_factor_params:
                    new_params['background'] = dict(self._form_factor_params['background'])
                else:
                    new_params['background'] = {
                        'value': 0.0,
                        'min': 0.0,
                        'max': np.inf,
                        'vary': False,
                        'description': 'Constant background level',
                    }

            self.params = new_params

            # Handle radius_effective linking
            if radius_effective_mode == 'link_radius':
                if 'radius' in self.params and 'radius_effective' in self.params:
                    # Link radius_effective to radius
                    self.params['radius_effective']['value'] = self.params['radius']['value']
                    self.params['radius_effective']['vary'] = False
                    print("  Note: 'radius_effective' linked to 'radius' value")
                else:
                    warnings.warn(
                        'Cannot link radius_effective to radius: one or both parameters not found. '
                        'Using unconstrained mode.',
                        stacklevel=2,
                    )
                    self._radius_effective_mode = 'unconstrained'

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

            # Restore form factor parameters
            self.params = {k: dict(v) for k, v in self._form_factor_params.items()}

            sf_name = self._structure_factor_name
            self._structure_factor_name = None
            self._radius_effective_mode = 'unconstrained'
            self._form_factor_params = {}

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
        # Prepare parameter dictionary for BumpsModel
        pars = {name: info['value'] for name, info in self.params.items()}

        # Create BUMPS model
        model = BumpsModel(self.kernel, **pars)

        # Set parameter ranges for fitting
        for name, info in self.params.items():
            if info['vary']:
                param_obj = getattr(model, name)
                param_obj.range(info['min'], info['max'])

        # Handle radius_effective linking in link_radius mode
        if (
            self._radius_effective_mode == 'link_radius'
            and hasattr(model, 'radius_effective')
            and hasattr(model, 'radius')
        ):
            # Constrain radius_effective to equal radius
            model.radius_effective = model.radius

        # Create experiment and fit problem
        experiment = Experiment(data=self.data, model=model)
        problem = FitProblem(experiment)

        print(f'\nInitial χ² = {problem.chisq():.4f}')
        print(f'Fitting with BUMPS (method: {method})...')

        # Perform fit
        result = bumps_fit(problem, method=method, **kwargs)

        # Store results
        self.fit_result = {
            'engine': 'bumps',
            'method': method,
            'chisq': problem.chisq(),
            'parameters': {},
            'problem': problem,
            'result': result,
        }

        # Extract fitted parameters
        for k, v, dv in zip(problem.labels(), result.x, result.dx):
            self.fit_result['parameters'][k] = {
                'value': v,
                'stderr': dv,
                'formatted': format_uncertainty(v, dv),
            }
            # Update internal parameter values
            if k in self.params:
                self.params[k]['value'] = v

        self._fitted_model = problem

        # Print results
        print('\n✓ Fit completed!')
        print(f'Final χ² = {self.fit_result["chisq"]:.4f}')
        print('\nFitted parameters:')
        for name, info in self.fit_result['parameters'].items():
            print(f'  {name}: {info["formatted"]}')

        return self.fit_result

    def _fit_lmfit(self, method: str = 'leastsq', **kwargs: Any) -> dict[str, Any]:
        """Fit using scipy.optimize (leastsq/least_squares) engine."""
        # Get initial parameter values and build bounds
        param_names = [name for name, info in self.params.items() if info['vary']]
        x0 = np.array([self.params[name]['value'] for name in param_names])
        bounds_lower = np.array([self.params[name]['min'] for name in param_names])
        bounds_upper = np.array([self.params[name]['max'] for name in param_names])

        # Create direct model calculator (kernel already set to CPU in set_model)
        calculator = DirectModel(self.data, self.kernel)

        # Capture instance attributes for use in residual closure
        radius_effective_mode = self._radius_effective_mode

        # Define residual function
        def residual(x):
            # Build full parameter dictionary
            par_dict = {name: info['value'] for name, info in self.params.items()}
            # Update with fitted parameters
            for i, name in enumerate(param_names):
                par_dict[name] = x[i]

            # Handle radius_effective linking in link_radius mode
            if (
                radius_effective_mode == 'link_radius'
                and 'radius' in par_dict
                and 'radius_effective' in par_dict
            ):
                par_dict['radius_effective'] = par_dict['radius']

            # Calculate model
            I_calc = calculator(**par_dict)
            # Return weighted residuals
            return (self.data.y - I_calc) / self.data.dy

        print(f'\nFitting with scipy.optimize (method: {method})...')

        # Perform fit based on method
        if method == 'leastsq':
            # Levenberg-Marquardt (no bounds support)
            result = leastsq(residual, x0, full_output=True, **kwargs)
            fitted_params = result[0]
            cov_matrix = result[1]
            result[2]

            # Calculate parameter errors from covariance matrix
            if cov_matrix is not None:
                param_errors = np.sqrt(np.diag(cov_matrix))
            else:
                param_errors = np.zeros_like(fitted_params)

            # Calculate chi-squared
            final_residuals = residual(fitted_params)
            chisq = np.sum(final_residuals**2)

        elif method == 'least_squares':
            # Trust Region Reflective (supports bounds)
            bounds = (bounds_lower, bounds_upper)
            result = least_squares(residual, x0, bounds=bounds, **kwargs)
            fitted_params = result.x

            # Estimate parameter errors from Jacobian
            try:
                # Compute covariance from Jacobian
                J = result.jac
                cov_matrix = np.linalg.inv(J.T @ J)
                param_errors = np.sqrt(np.diag(cov_matrix))
            except Exception as e:
                # If Jacobian-based covariance estimation fails, fall back to zeros
                # and emit a warning so users can investigate the cause.
                warnings.warn(f'Failed to compute covariance from Jacobian: {e}', stacklevel=2)
                param_errors = np.zeros_like(fitted_params)

            chisq = np.sum(result.fun**2)

        elif method == 'differential_evolution':
            # Global optimizer (supports bounds)
            bounds_list = list(zip(bounds_lower, bounds_upper))

            def objective(x):
                return np.sum(residual(x) ** 2)

            result = differential_evolution(objective, bounds_list, **kwargs)
            fitted_params = result.x
            param_errors = np.zeros_like(fitted_params)  # DE doesn't provide errors
            chisq = result.fun

        else:
            raise ValueError(
                f"Unknown method '{method}'. Use 'leastsq', 'least_squares', or 'differential_evolution'."
            )

        # Store results
        self.fit_result = {
            'engine': 'lmfit',
            'method': method,
            'chisq': chisq,
            'parameters': {},
            'result': result,
        }

        # Extract fitted parameters
        for i, name in enumerate(param_names):
            self.fit_result['parameters'][name] = {
                'value': fitted_params[i],
                'stderr': param_errors[i],
                'formatted': f'{fitted_params[i]:.6g} ± {param_errors[i]:.6g}'
                if param_errors[i] > 0
                else f'{fitted_params[i]:.6g}',
            }
            # Update internal parameter values
            self.params[name]['value'] = fitted_params[i]

        # Add fixed parameters to results
        for name, info in self.params.items():
            if name not in param_names:
                self.fit_result['parameters'][name] = {
                    'value': info['value'],
                    'stderr': 0.0,
                    'formatted': f'{info["value"]:.6g} (fixed)',
                }

        self._fitted_model = result

        # Print results
        print('\n✓ Fit completed!')
        print(f'Final χ² = {self.fit_result["chisq"]:.4f}')
        print('\nFitted parameters:')
        for name, info in self.fit_result['parameters'].items():
            print(f'  {name}: {info["formatted"]}')

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
                    error_y=dict(type='data', array=self.data.dy, visible=True),
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

        # Calculate fitted curve
        if self.fit_result['engine'] == 'bumps':
            problem = self._fitted_model
            q = self.data.x
            I_fit = problem.fitness.theory()
        else:  # lmfit
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
            error_y=dict(type='data', array=self.data.dy, visible=True),
            mode='markers',
            name='Experimental Data',
            opacity=0.6,
            marker=dict(size=6),
        )

        # Fitted model line
        fit_trace = go.Scatter(
            x=q,
            y=I_fit,
            mode='lines',
            name='Fitted Model',
            line=dict(color='red', width=2),
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
                    marker=dict(size=6),
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

            # Get fitted curve
            if self.fit_result['engine'] == 'bumps':
                I_fit = self._fitted_model.fitness.theory()
            else:
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

"""
Scipy Fitting Engine - Implementation using scipy.optimize.

This module provides a concrete fitting engine that uses scipy's optimization
methods for parameter fitting.
"""

import warnings
from typing import Any

import numpy as np
from sasmodels.direct_model import DirectModel

from .fitting_engine import FittingEngine

try:
    from scipy.optimize import differential_evolution, least_squares, leastsq

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class ScipyFittingEngine(FittingEngine):
    """
    Fitting engine using scipy.optimize methods.

    Supports various optimization algorithms including:
    - Levenberg-Marquardt (leastsq)
    - Trust Region Reflective (least_squares)
    - Differential Evolution (differential_evolution)
    """

    def __init__(self):
        """Initialize the Scipy fitting engine."""
        if not SCIPY_AVAILABLE:
            raise ImportError(
                'scipy is not available. Install scipy to use this engine: pip install scipy'
            )

    @property
    def name(self) -> str:
        """Get the name of this fitting engine."""
        return 'scipy'

    @property
    def available_methods(self) -> list[str]:
        """Get list of available scipy optimization methods."""
        return ['leastsq', 'least_squares', 'differential_evolution']

    def fit(
        self,
        data: Any,
        kernel: Any,
        param_manager: Any,
        method: str = 'leastsq',
        engine_name: str = 'scipy',
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Perform fit using scipy.optimize engine.

        Args:
            data: SANS data object from sasdata
            kernel: SasModels kernel
            param_manager: ParameterManager instance
            method: scipy method (default: 'leastsq')
            engine_name: Name to report in results (default: 'scipy', can be 'lmfit' for compatibility)
            **kwargs: Additional arguments passed to scipy optimizer

        Returns:
            Dictionary with fit results
        """
        # Get initial parameter values and build bounds
        param_names = param_manager.get_varying_params()
        params_dict = param_manager.get_param_dict()
        x0 = np.array([params_dict[name]['value'] for name in param_names])
        bounds_lower = np.array([params_dict[name]['min'] for name in param_names])
        bounds_upper = np.array([params_dict[name]['max'] for name in param_names])

        # Create direct model calculator
        calculator = DirectModel(data, kernel)

        # Capture instance attributes for use in residual closure
        radius_effective_mode = param_manager.get_radius_effective_mode()

        # Define residual function
        def residual(x):
            # Build full parameter dictionary
            par_dict = param_manager.get_param_values()
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
            return (data.y - I_calc) / data.dy

        print(f'\nFitting with scipy.optimize (method: {method})...')

        # Perform fit based on method
        if method == 'leastsq':
            # Levenberg-Marquardt (no bounds support)
            result = leastsq(residual, x0, full_output=True, **kwargs)
            fitted_params = result[0]
            cov_matrix = result[1]

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
        fit_result = {
            'engine': engine_name,  # Use the provided engine name for compatibility
            'method': method,
            'chisq': chisq,
            'parameters': {},
            'result': result,
            'calculator': calculator,  # Store for get_fitted_curve
            'param_names': param_names,
            'fitted_params': fitted_params,
        }

        # Extract fitted parameters
        for i, name in enumerate(param_names):
            fit_result['parameters'][name] = {
                'value': fitted_params[i],
                'stderr': param_errors[i],
                'formatted': f'{fitted_params[i]:.6g} ± {param_errors[i]:.6g}'
                if param_errors[i] > 0
                else f'{fitted_params[i]:.6g}',
            }
            # Update internal parameter values
            param_manager.update_param_value(name, fitted_params[i])

        # Add fixed parameters to results
        for name, info in params_dict.items():
            if name not in param_names:
                fit_result['parameters'][name] = {
                    'value': info['value'],
                    'stderr': 0.0,
                    'formatted': f'{info["value"]:.6g} (fixed)',
                }

        # Print results
        print('\n✓ Fit completed!')
        print(f'Final χ² = {fit_result["chisq"]:.4f}')
        print('\nFitted parameters:')
        for name, info in fit_result['parameters'].items():
            print(f'  {name}: {info["formatted"]}')

        return fit_result

    def get_fitted_curve(self, fit_result: dict[str, Any], data: Any, kernel: Any) -> tuple:
        """
        Calculate the fitted curve from scipy fit results.

        Args:
            fit_result: Dictionary from fit() method
            data: SANS data object
            kernel: SasModels kernel

        Returns:
            Tuple of (q_values, intensity_values)
        """
        # Use stored calculator if available, otherwise create new one
        if 'calculator' in fit_result:
            calculator = fit_result['calculator']
        else:
            calculator = DirectModel(data, kernel)

        # Build parameter dictionary from fit results
        par_dict = {name: info['value'] for name, info in fit_result['parameters'].items()}

        # Calculate fitted curve
        q = data.x
        I_fit = calculator(**par_dict)
        return q, I_fit

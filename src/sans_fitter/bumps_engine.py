"""
BUMPS Fitting Engine - Implementation using the BUMPS optimization library.

This module provides a concrete fitting engine that uses BUMPS for
parameter optimization.
"""

from typing import Any

from bumps.fitters import fit as bumps_fit
from bumps.formatnum import format_uncertainty
from bumps.names import FitProblem
from sasmodels.bumps_model import Experiment
from sasmodels.bumps_model import Model as BumpsModel

from .fitting_engine import FittingEngine


class BumpsFittingEngine(FittingEngine):
    """
    Fitting engine using the BUMPS optimization library.

    BUMPS provides various optimization methods including:
    - Nelder-Mead simplex (amoeba)
    - Levenberg-Marquardt (lm)
    - Newton's method (newton)
    - Differential Evolution (de)
    - DREAM Markov Chain Monte Carlo (dream)
    """

    @property
    def name(self) -> str:
        """Get the name of this fitting engine."""
        return 'bumps'

    @property
    def available_methods(self) -> list[str]:
        """Get list of available BUMPS optimization methods."""
        return ['amoeba', 'lm', 'newton', 'de', 'dream', 'pt', 'rl', 'ps']

    def fit(
        self,
        data: Any,
        kernel: Any,
        param_manager: Any,
        method: str = 'amoeba',
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Perform fit using BUMPS engine.

        Args:
            data: SANS data object from sasdata
            kernel: SasModels kernel
            param_manager: ParameterManager instance
            method: BUMPS method (default: 'amoeba')
            **kwargs: Additional arguments passed to bumps_fit

        Returns:
            Dictionary with fit results
        """
        # Prepare parameter dictionary for BumpsModel
        pars = param_manager.get_param_values()

        # Create BUMPS model
        model = BumpsModel(kernel, **pars)

        # Set parameter ranges for fitting
        for name, info in param_manager.params.items():
            if info['vary']:
                param_obj = getattr(model, name)
                param_obj.range(info['min'], info['max'])

        # Handle radius_effective linking in link_radius mode
        if (
            param_manager.get_radius_effective_mode() == 'link_radius'
            and hasattr(model, 'radius_effective')
            and hasattr(model, 'radius')
        ):
            # Constrain radius_effective to equal radius
            model.radius_effective = model.radius

        # Create experiment and fit problem
        experiment = Experiment(data=data, model=model)
        problem = FitProblem(experiment)

        print(f'\nInitial χ² = {problem.chisq():.4f}')
        print(f'Fitting with BUMPS (method: {method})...')

        # Perform fit
        result = bumps_fit(problem, method=method, **kwargs)

        # Store results
        fit_result = {
            'engine': 'bumps',
            'method': method,
            'chisq': problem.chisq(),
            'parameters': {},
            'problem': problem,
            'result': result,
        }

        # Extract fitted parameters
        for k, v, dv in zip(problem.labels(), result.x, result.dx):
            fit_result['parameters'][k] = {
                'value': v,
                'stderr': dv,
                'formatted': format_uncertainty(v, dv),
            }
            # Update internal parameter values
            if param_manager.validate_param(k):
                param_manager.update_param_value(k, v)

        # Print results
        print('\n✓ Fit completed!')
        print(f'Final χ² = {fit_result["chisq"]:.4f}')
        print('\nFitted parameters:')
        for name, info in fit_result['parameters'].items():
            print(f'  {name}: {info["formatted"]}')

        return fit_result

    def get_fitted_curve(self, fit_result: dict[str, Any], data: Any, kernel: Any) -> tuple:
        """
        Calculate the fitted curve from BUMPS fit results.

        Args:
            fit_result: Dictionary from fit() method
            data: SANS data object
            kernel: SasModels kernel (not used for BUMPS)

        Returns:
            Tuple of (q_values, intensity_values)
        """
        problem = fit_result['problem']
        q = data.x
        I_fit = problem.fitness.theory()
        return q, I_fit

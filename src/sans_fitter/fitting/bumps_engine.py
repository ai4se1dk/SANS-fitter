from typing import Any

import numpy as np
from bumps.fitters import fit as bumps_fit

try:  # bumps >= 1.0.4
    from bumps.util import format_uncertainty
except ImportError:  # bumps <= 1.0.3
    from bumps.formatnum import format_uncertainty
from bumps.names import FitProblem
from sasmodels.bumps_model import Experiment
from sasmodels.bumps_model import Model as BumpsModel

from ..contracts import ParameterStateSnapshot
from ..results import FitArtifacts, FitResultContract
from .base import EngineFitOutput, link_radius_effective_model, pd_is_active


def fit_bumps(
    data: Any,
    kernel: Any,
    fit_state: ParameterStateSnapshot,
    method: str = 'amoeba',
    **kwargs: Any,
) -> EngineFitOutput:
    """Fit using the BUMPS engine."""
    pars = {name: info['value'] for name, info in fit_state.params.items()}

    if fit_state.pd_enabled:
        for param_name in fit_state.polydisperse_param_names:
            pd_config = fit_state.polydisperse_params[param_name]
            if pd_is_active(pd_config):
                pars[f'{param_name}_pd'] = pd_config['pd']
                pars[f'{param_name}_pd_n'] = pd_config['pd_n']
                pars[f'{param_name}_pd_nsigma'] = pd_config['pd_nsigma']
                pars[f'{param_name}_pd_type'] = pd_config['pd_type']

    model = BumpsModel(kernel, **pars)

    for name, info in fit_state.params.items():
        if info['vary']:
            getattr(model, name).range(info['min'], info['max'])

    if fit_state.pd_enabled:
        for param_name in fit_state.polydisperse_param_names:
            pd_config = fit_state.polydisperse_params[param_name]
            if pd_is_active(pd_config) and pd_config.get('vary', False):
                getattr(model, f'{param_name}_pd').range(0, 1)

    link_radius_effective_model(model, fit_state.radius_effective_mode)

    experiment = Experiment(data=data, model=model)
    problem = FitProblem(experiment)

    print(f'\nInitial χ² = {problem.chisq():.4f}')
    print(f'Fitting with BUMPS (method: {method})...')

    result = bumps_fit(problem, method=method, **kwargs)

    result_parameters: dict[str, dict[str, Any]] = {}
    fitted_values: dict[str, float] = {}

    for name, value, stderr in zip(problem.labels(), result.x, result.dx):
        result_parameters[name] = {
            'value': value,
            'stderr': stderr,
            'formatted': format_uncertainty(value, stderr),
        }
        fitted_values[name] = value

    contract = FitResultContract(
        engine='bumps',
        method=method,
        chisq=problem.chisq(),
        parameters=result_parameters,
        artifacts=FitArtifacts(
            fitted_curve=np.asarray(problem.fitness.theory()),
            raw_result=result,
            runtime_handle=problem,
            runtime_key='problem',
        ),
    )

    return EngineFitOutput(contract=contract, fitted_values=fitted_values, runtime_model=problem)

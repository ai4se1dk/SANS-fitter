import warnings
from typing import Any

import numpy as np
from sasmodels.direct_model import DirectModel

from ..results import FitArtifacts, FitResultContract, ParameterStateSnapshot
from .base import EngineFitOutput, apply_parameter_links, extract_fit_index, pd_is_active

try:
    from scipy.optimize import differential_evolution, least_squares, leastsq

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def fit_scipy(
    data: Any,
    kernel: Any,
    fit_state: ParameterStateSnapshot,
    method: str = 'leastsq',
    **kwargs: Any,
) -> EngineFitOutput:
    """Fit using scipy.optimize based fitting methods."""
    param_names = [name for name, info in fit_state.params.items() if info['vary']]
    x0_list = [fit_state.params[name]['value'] for name in param_names]
    bounds_lower_list = [fit_state.params[name]['min'] for name in param_names]
    bounds_upper_list = [fit_state.params[name]['max'] for name in param_names]

    if fit_state.pd_enabled:
        for base_param in fit_state.polydisperse_param_names:
            pd_config = fit_state.polydisperse_params[base_param]
            if pd_is_active(pd_config) and pd_config.get('vary', False):
                pd_name = f'{base_param}_pd'
                param_names.append(pd_name)
                x0_list.append(pd_config['pd'])
                bounds_lower_list.append(0.0)
                bounds_upper_list.append(1.0)

    x0 = np.array(x0_list)
    bounds_lower = np.array(bounds_lower_list)
    bounds_upper = np.array(bounds_upper_list)
    calculator = DirectModel(data, kernel)
    # sasmodels evaluates the theory only at the fitted points (inside
    # [qmin, qmax], unmasked); compare against the matching data subset.
    y_fit = np.asarray(calculator.Iq)
    dy_fit = np.asarray(calculator.dIq, dtype=float)
    # Zero dI would make the weighted residual infinite. Weight such points as
    # 1.0 instead — they will dominate χ² relative to small-error points, but
    # masking them would silently change the degrees of freedom.
    zero_dy = np.nan_to_num(dy_fit) == 0
    if zero_dy.any():
        if zero_dy.all():
            warnings.warn(
                'All intensity uncertainties (dI) are zero; using unweighted residuals.',
                stacklevel=2,
            )
        else:
            warnings.warn(
                f'{int(zero_dy.sum())} of {zero_dy.size} fitted points have zero intensity '
                'uncertainty (dI); weighting them as 1.0.',
                stacklevel=2,
            )
        dy_fit = np.where(zero_dy, 1.0, dy_fit)

    def build_parameter_dict(x: np.ndarray) -> dict[str, Any]:
        par_dict = {name: info['value'] for name, info in fit_state.params.items()}

        for index, name in enumerate(param_names):
            if name in par_dict or name.endswith('_pd'):
                par_dict[name] = x[index]

        if fit_state.pd_enabled:
            for base_param in fit_state.polydisperse_param_names:
                pd_config = fit_state.polydisperse_params[base_param]
                if pd_is_active(pd_config):
                    if f'{base_param}_pd' not in par_dict:
                        par_dict[f'{base_param}_pd'] = pd_config['pd']
                    par_dict[f'{base_param}_pd_n'] = pd_config['pd_n']
                    par_dict[f'{base_param}_pd_nsigma'] = pd_config['pd_nsigma']
                    par_dict[f'{base_param}_pd_type'] = pd_config['pd_type']

        apply_parameter_links(par_dict, fit_state.linked_params)
        return par_dict

    def residual(x: np.ndarray) -> np.ndarray:
        i_calc = calculator(**build_parameter_dict(x))
        return (y_fit - i_calc) / dy_fit

    print(f'\nFitting with scipy.optimize (method: {method})...')

    if method == 'leastsq':
        result = leastsq(residual, x0, full_output=True, **kwargs)
        fitted_params = result[0]
        cov_matrix = result[1]
        if cov_matrix is not None:
            param_errors = np.sqrt(np.diag(cov_matrix))
        else:
            param_errors = np.zeros_like(fitted_params)
        chisq = np.sum(residual(fitted_params) ** 2)
    elif method == 'least_squares':
        result = least_squares(residual, x0, bounds=(bounds_lower, bounds_upper), **kwargs)
        fitted_params = result.x
        try:
            cov_matrix = np.linalg.inv(result.jac.T @ result.jac)
            param_errors = np.sqrt(np.diag(cov_matrix))
        except Exception as e:
            warnings.warn(f'Failed to compute covariance from Jacobian: {e}', stacklevel=2)
            param_errors = np.zeros_like(fitted_params)
        chisq = np.sum(result.fun**2)
    elif method == 'differential_evolution':
        bounds_list = list(zip(bounds_lower, bounds_upper, strict=True))

        def objective(x: np.ndarray) -> np.floating[Any]:
            return np.sum(residual(x) ** 2)

        result = differential_evolution(objective, bounds_list, **kwargs)
        fitted_params = result.x
        param_errors = np.zeros_like(fitted_params)
        chisq = result.fun
    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'leastsq', 'least_squares', or 'differential_evolution'."
        )

    # The parameter set the fit actually landed on: link followers carry their
    # target's fitted value here, which their stale fit_state entry does not.
    final_pars = build_parameter_dict(fitted_params)

    result_parameters: dict[str, dict[str, Any]] = {}
    fitted_values: dict[str, float] = {}

    for index, name in enumerate(param_names):
        result_parameters[name] = {
            'value': fitted_params[index],
            'stderr': param_errors[index],
            'formatted': f'{fitted_params[index]:.6g} ± {param_errors[index]:.6g}'
            if param_errors[index] > 0
            else f'{fitted_params[index]:.6g}',
        }
        fitted_values[name] = fitted_params[index]

    for name, info in fit_state.params.items():
        if name not in param_names:
            value = final_pars.get(name, info['value'])
            label = (
                f'{value:.6g} (= {fit_state.linked_params[name]})'
                if name in fit_state.linked_params
                else f'{value:.6g} (fixed)'
            )
            result_parameters[name] = {
                'value': value,
                'stderr': 0.0,
                'formatted': label,
            }

    contract = FitResultContract(
        engine='lmfit',
        method=method,
        chisq=chisq,
        parameters=result_parameters,
        artifacts=FitArtifacts(
            fitted_curve=np.asarray(calculator(**final_pars)),
            fit_index=extract_fit_index(calculator),
            raw_result=result,
        ),
    )

    return EngineFitOutput(contract=contract, fitted_values=fitted_values, runtime_model=result)

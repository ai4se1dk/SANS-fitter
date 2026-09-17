import warnings
from typing import Any

import numpy as np
from sasmodels.direct_model import DirectModel

from ..console import logger
from ..results import FitArtifacts, FitResultContract, ParameterStateSnapshot
from .base import (
    EngineFitOutput,
    apply_parameter_links,
    build_result_parameters,
    extract_fit_index,
    normalize_message,
    pd_is_active,
    reduced_chisq,
    validate_covariance,
)

try:
    from scipy.optimize import differential_evolution, least_squares, leastsq

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Relative accuracy of the model curve. SANSFitter compiles sasmodels kernels
# in single precision (``load_model(..., dtype='single')``), so two evaluations
# agree to about 1e-7 and no further.
KERNEL_PRECISION = float(np.finfo(np.float32).eps)
# Forward-difference step for the numerical Jacobian, as a fraction of each
# parameter. scipy defaults to sqrt(double eps) ~= 1.5e-8, which is *below* the
# kernel's own resolution: the perturbed curve comes back bit-identical, the
# Jacobian is exactly zero, and the optimizer stops at the starting point while
# reporting success. Sizing the step to the kernel's precision instead makes
# the derivative observable.
JACOBIAN_STEP = float(np.sqrt(KERNEL_PRECISION))


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
    weighting_note = 'dI'
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
        weighting_note = (
            f'dI ({int(zero_dy.sum())} of {zero_dy.size} points unit-weighted because dI = 0)'
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

    logger.info(f'\nFitting with scipy.optimize (method: {method})...')

    # Initialized before the branching so a method that cannot supply a covariance
    # (differential evolution) or whose inversion fails leaves it None rather than
    # unbound.
    cov_matrix: np.ndarray | None = None
    cov_source: str | None = None
    converged: bool | None = None
    message = ''

    if method == 'leastsq':
        # epsfcn is the assumed relative error in the function; leastsq derives
        # its step as sqrt(epsfcn)*x. See KERNEL_PRECISION.
        kwargs.setdefault('epsfcn', KERNEL_PRECISION)
        result = leastsq(residual, x0, full_output=True, **kwargs)
        fitted_params = result[0]
        cov_matrix = result[1]
        if cov_matrix is not None:
            param_errors = np.sqrt(np.diag(cov_matrix))
            cov_source = 'scipy cov_x'
        else:
            param_errors = np.zeros_like(fitted_params)
        # ier 1-4 are leastsq's success codes; 0 and 5+ are failures.
        converged = result[4] in (1, 2, 3, 4)
        message = normalize_message(result[3])
    elif method == 'least_squares':
        # diff_step is the relative step itself, not its square.
        kwargs.setdefault('diff_step', JACOBIAN_STEP)
        result = least_squares(residual, x0, bounds=(bounds_lower, bounds_upper), **kwargs)
        fitted_params = result.x
        try:
            cov_matrix = np.linalg.inv(result.jac.T @ result.jac)
            param_errors = np.sqrt(np.diag(cov_matrix))
            cov_source = 'jacobian'
        except np.linalg.LinAlgError as e:
            warnings.warn(f'Failed to compute covariance from Jacobian: {e}', stacklevel=2)
            cov_matrix = None
            param_errors = np.zeros_like(fitted_params)
        # status > 0 is a termination criterion being met; 0 is the evaluation
        # budget running out and -1 an infeasible start.
        converged = result.status > 0
        message = normalize_message(result.message)
    elif method == 'differential_evolution':
        bounds_list = list(zip(bounds_lower, bounds_upper, strict=True))

        def objective(x: np.ndarray) -> np.floating[Any]:
            return np.sum(residual(x) ** 2)

        result = differential_evolution(objective, bounds_list, **kwargs)
        fitted_params = result.x
        param_errors = np.zeros_like(fitted_params)
        converged = bool(result.success)
        message = normalize_message(result.message)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'leastsq', 'least_squares', or 'differential_evolution'."
        )

    # One evaluation at the optimum, so chisq and the exported residuals cannot
    # disagree. differential_evolution's result.fun is the same objective value.
    final_residuals = residual(fitted_params)
    chisq = float(np.sum(final_residuals**2))

    varied: dict[str, dict[str, Any]] = {}
    # The parameter set the fit actually landed on: link followers carry their
    # target's fitted value here, which their stale fit_state entry does not.
    final_pars = build_parameter_dict(fitted_params)

    fitted_values: dict[str, float] = {}

    for index, name in enumerate(param_names):
        varied[name] = {
            'value': fitted_params[index],
            'stderr': param_errors[index],
            'formatted': f'{fitted_params[index]:.6g} ± {param_errors[index]:.6g}'
            if param_errors[index] > 0
            else f'{fitted_params[index]:.6g}',
        }
        fitted_values[name] = fitted_params[index]

    n_points = int(y_fit.size)
    n_free = len(param_names)
    dof = n_points - n_free

    contract = FitResultContract(
        engine='lmfit',
        method=method,
        chisq=chisq,
        reduced_chisq=reduced_chisq(chisq, dof),
        n_points=n_points,
        n_free=n_free,
        dof=dof,
        weighting_note=weighting_note,
        parameters=build_result_parameters(fit_state, varied),
        converged=converged,
        message=message,
        cov=None if cov_matrix is None else validate_covariance(cov_matrix, param_names),
        cov_labels=list(param_names),
        cov_source=cov_source,
        artifacts=FitArtifacts(
            fitted_curve=np.asarray(calculator(**final_pars)),
            fit_index=extract_fit_index(calculator),
            residuals=np.asarray(final_residuals, dtype=float),
            raw_result=result,
        ),
    )

    return EngineFitOutput(contract=contract, fitted_values=fitted_values, runtime_model=result)

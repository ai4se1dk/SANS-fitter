from copy import deepcopy
from typing import Any, Callable, Optional

import numpy as np
from bumps.fitters import fit as bumps_fit

try:  # bumps >= 1.0.4
    from bumps.util import format_uncertainty
except ImportError:  # bumps <= 1.0.3
    from bumps.formatnum import format_uncertainty
from bumps.names import FitProblem
from sasmodels.bumps_model import Experiment
from sasmodels.bumps_model import Model as BumpsModel
from sasmodels.direct_model import DirectModel

from ..contracts import ParameterStateSnapshot
from ..results import (
    MIN_POSTERIOR_PARAMETER_COUNT,
    MIN_POSTERIOR_SAMPLE_COUNT,
    FitArtifacts,
    FitResultContract,
    PosteriorSummary,
)
from .base import (
    EngineFitOutput,
    extract_fit_index,
    link_radius_effective_dict,
    link_radius_effective_model,
    pd_is_active,
)

DEFAULT_DREAM_SAMPLES = 10000
# bumps expresses burn and pop in DREAM's native units: burn counts
# generations (not samples) and pop scales the chain count per parameter.
DEFAULT_DREAM_BURN = 200
DEFAULT_DREAM_THIN = 1
DEFAULT_DREAM_POP = 10

__all__ = [
    'DEFAULT_DREAM_SAMPLES',
    'DEFAULT_DREAM_BURN',
    'DEFAULT_DREAM_THIN',
    'DEFAULT_DREAM_POP',
    'MIN_POSTERIOR_PARAMETER_COUNT',
    'MIN_POSTERIOR_SAMPLE_COUNT',
    'fit_bumps',
    'fit_bumps_dream',
]


def _build_bumps_problem(
    data: Any,
    kernel: Any,
    fit_state: ParameterStateSnapshot,
) -> tuple[Any, Any]:
    """Build the bumps FitProblem shared by point-estimate and DREAM fits.

    Returns the (problem, experiment) pair so callers can extract the fit
    index from the experiment after fitting.
    """
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
    return problem, experiment


def fit_bumps(
    data: Any,
    kernel: Any,
    fit_state: ParameterStateSnapshot,
    method: str = 'amoeba',
    **kwargs: Any,
) -> EngineFitOutput:
    """Fit using the BUMPS engine."""
    problem, experiment = _build_bumps_problem(data, kernel, fit_state)

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
            fitted_curve=np.asarray(experiment.theory()),
            fit_index=extract_fit_index(experiment),
            raw_result=result,
            runtime_handle=problem,
            runtime_key='problem',
        ),
    )

    return EngineFitOutput(contract=contract, fitted_values=fitted_values, runtime_model=problem)


def _require_dream_state(result: Any) -> Any:
    """Return the DREAM MCMC state from a bumps fit result.

    Guards against bumps API drift: the posterior extraction relies on
    ``result.state`` exposing ``draw()`` (bumps' MCMCDraw interface).
    """
    state = getattr(result, 'state', None)
    if state is None or not callable(getattr(state, 'draw', None)):
        import bumps

        raise RuntimeError(
            f'bumps {bumps.__version__} did not return a DREAM sampler state with a '
            'draw() method; the installed bumps version is incompatible with '
            'posterior extraction. Expected result.state to be an MCMCDraw.'
        )
    return state


def _effective_sample_size(series: np.ndarray) -> float:
    """Estimate the effective sample size of one chain via autocorrelation.

    Uses the initial-positive-sequence truncation of the FFT-based
    autocorrelation function.
    """
    series = np.asarray(series, dtype=float)
    n = series.size
    if n < 2:
        return float(n)
    centered = series - series.mean()
    variance = np.dot(centered, centered)
    if variance == 0:
        return float(n)
    transform = np.fft.rfft(centered, n=2 * n)
    acf = np.fft.irfft(transform * np.conj(transform))[:n].real / variance
    tau = 1.0
    for lag in range(1, n):
        if acf[lag] <= 0:
            break
        tau += 2.0 * acf[lag]
    return float(min(n, n / tau))


def _compute_diagnostics(
    state: Any, labels: list[str], chains: Optional[np.ndarray]
) -> Optional[dict[str, dict[str, float]]]:
    """Compute per-parameter r-hat/ESS, or None when unavailable."""
    if chains is None:
        return None
    try:
        r_hat = np.asarray(state.gelman(), dtype=float).ravel()
        if r_hat.size != len(labels):
            return None
        return {
            name: {
                'r_hat': float(r_hat[i]),
                # DREAM's flattened draws interleave chains. Estimate each
                # chain in generation order, then combine independent ESS.
                'ess': float(
                    min(
                        chains.shape[0] * chains.shape[1],
                        sum(
                            _effective_sample_size(chains[:, chain_index, i])
                            for chain_index in range(chains.shape[1])
                        ),
                    )
                ),
            }
            for i, name in enumerate(labels)
        }
    except Exception:
        return None


def _extract_posterior(state: Any, labels: list[str], best_x: np.ndarray) -> PosteriorSummary:
    """Build a PosteriorSummary from a bumps DREAM state."""
    draw = state.draw()
    samples = np.asarray(draw.points, dtype=float)
    draw_labels = list(getattr(draw, 'labels', labels))
    if draw_labels != labels:
        raise RuntimeError(
            f'Posterior chain labels {draw_labels} do not match the problem '
            f'parameter labels {labels}; refusing to build a mislabelled posterior.'
        )
    if samples.ndim != 2 or samples.shape[0] < MIN_POSTERIOR_SAMPLE_COUNT:
        raise RuntimeError(
            'DREAM returned a degenerate posterior chain '
            f'(shape {samples.shape}). Increase samples/burn and retry.'
        )

    logp = getattr(draw, 'logp', None)
    logp = None if logp is None else np.asarray(logp, dtype=float)

    stds = samples.std(axis=0)
    if np.all(stds == 0):
        raise RuntimeError(
            'DREAM produced a flat posterior chain (all samples identical). '
            'The sampler failed to explore the parameter space; check the '
            'parameter bounds and increase samples/burn.'
        )

    try:
        _, chains, _ = state.chains()
        chains = np.asarray(chains, dtype=float)
        good_chains = getattr(state, '_good_chains', None)
        if good_chains is not None:
            chains = chains[:, good_chains, :]
        if chains.ndim != 3 or chains.shape[-1] != len(labels):
            chains = None
    except Exception:
        chains = None

    means = samples.mean(axis=0)
    medians = np.median(samples, axis=0)
    percentiles = np.percentile(samples, [16, 84, 2.5, 97.5], axis=0)

    return PosteriorSummary(
        labels=labels,
        samples=samples,
        logp=logp,
        chains=chains,
        best={name: float(best_x[i]) for i, name in enumerate(labels)},
        mean={name: float(means[i]) for i, name in enumerate(labels)},
        median={name: float(medians[i]) for i, name in enumerate(labels)},
        std={name: float(stds[i]) for i, name in enumerate(labels)},
        ci_68={
            name: (float(percentiles[0, i]), float(percentiles[1, i]))
            for i, name in enumerate(labels)
        },
        ci_95={
            name: (float(percentiles[2, i]), float(percentiles[3, i]))
            for i, name in enumerate(labels)
        },
        diagnostics=_compute_diagnostics(state, labels, chains),
    )


def _build_posterior_evaluator(
    data: Any, kernel: Any, fit_state: ParameterStateSnapshot
) -> tuple[Any, Callable[[dict[str, float]], np.ndarray]]:
    """Freeze the data and fixed model state used for posterior predictions."""
    posterior_data = deepcopy(data)
    calculator = DirectModel(posterior_data, kernel)
    base_pars = {name: info['value'] for name, info in fit_state.params.items()}

    if fit_state.pd_enabled:
        for base_param in fit_state.polydisperse_param_names:
            pd_config = fit_state.polydisperse_params[base_param]
            if pd_is_active(pd_config):
                base_pars[f'{base_param}_pd'] = pd_config['pd']
                base_pars[f'{base_param}_pd_n'] = pd_config['pd_n']
                base_pars[f'{base_param}_pd_nsigma'] = pd_config['pd_nsigma']
                base_pars[f'{base_param}_pd_type'] = pd_config['pd_type']

    def model_eval(sample: dict[str, float]) -> np.ndarray:
        pars = dict(base_pars)
        pars.update(sample)
        link_radius_effective_dict(pars, fit_state.radius_effective_mode)
        return np.asarray(calculator(**pars))

    return posterior_data, model_eval


def fit_bumps_dream(
    data: Any,
    kernel: Any,
    fit_state: ParameterStateSnapshot,
    method: str = 'dream',
    samples: int = DEFAULT_DREAM_SAMPLES,
    burn: int = DEFAULT_DREAM_BURN,
    thin: int = DEFAULT_DREAM_THIN,
    pop: int = DEFAULT_DREAM_POP,
    **kwargs: Any,
) -> EngineFitOutput:
    """Bayesian fit using bumps' DREAM sampler.

    Runs MCMC over the varying parameters and returns the same contract as
    fit_bumps, with the posterior chain attached as artifacts.posterior.
    The reported point estimate is the best (maximum-likelihood) posterior
    sample, and the fitted curve/chisq are re-evaluated at that point.
    """
    problem, experiment = _build_bumps_problem(data, kernel, fit_state)

    labels = list(problem.labels())
    if not labels:
        raise ValueError(
            'Bayesian fitting requires at least one varying parameter. '
            'Use set_param(..., vary=True) before fit_bayesian().'
        )

    print(f'\nInitial χ² = {problem.chisq():.4f}')
    print(f'Sampling posterior with BUMPS DREAM (samples={samples}, burn={burn} generations)...')

    result = bumps_fit(
        problem, method=method, samples=samples, burn=burn, thin=thin, pop=pop, **kwargs
    )
    state = _require_dream_state(result)

    # Re-set the problem to the chosen point estimate and re-evaluate the
    # curve and chisq; the problem's residual state after sampling is
    # whatever point DREAM evaluated last, not the best one.
    point_estimate = np.asarray(result.x, dtype=float)
    problem.setp(point_estimate)
    fitted_curve = np.asarray(experiment.theory())
    chisq = problem.chisq()

    posterior = _extract_posterior(state, labels, point_estimate)
    posterior_data, posterior_model_eval = _build_posterior_evaluator(data, kernel, fit_state)

    result_parameters: dict[str, dict[str, Any]] = {}
    fitted_values: dict[str, float] = {}

    # For DREAM, result.dx is the posterior 68% credible half-width, so the
    # existing formatted-uncertainty convention carries over unchanged.
    for name, value, stderr in zip(labels, result.x, result.dx):
        result_parameters[name] = {
            'value': value,
            'stderr': stderr,
            'formatted': format_uncertainty(value, stderr),
        }
        fitted_values[name] = value

    contract = FitResultContract(
        engine='bumps',
        method=method,
        chisq=chisq,
        parameters=result_parameters,
        artifacts=FitArtifacts(
            fitted_curve=fitted_curve,
            fit_index=extract_fit_index(experiment),
            raw_result=result,
            runtime_handle=problem,
            runtime_key='problem',
            posterior=posterior,
            posterior_data=posterior_data,
            posterior_model_eval=posterior_model_eval,
        ),
    )

    return EngineFitOutput(contract=contract, fitted_values=fitted_values, runtime_model=problem)

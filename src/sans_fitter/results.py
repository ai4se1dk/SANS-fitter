from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .data.loader import has_real_data
from .data.resolution import ResolutionSetting

MIN_POSTERIOR_PARAMETER_COUNT = 2
MIN_POSTERIOR_SAMPLE_COUNT = 2

# Engine name carried by contracts built from a theory preview rather than a
# fit (SANSFitter.plot_model). Plotting branches on it for titles and labels.
PREVIEW_ENGINE = 'preview'


def _format_resolution(setting: dict[str, Any]) -> str:
    """Render a stored resolution setting for the CSV header, in plain ASCII."""
    try:
        return ResolutionSetting(**setting).describe(ascii_only=True)
    except TypeError:  # a setting saved by a future/other shape — show it raw
        return str(setting)


@dataclass(slots=True)
class ParameterStateSnapshot:
    """Read-only parameter state passed into fitting engines.

    All parameter names carried by the snapshot are **canonical sasmodels
    names** (e.g. ``A_sld`` under a composite model). Any user-facing alias
    layer is translated away before the snapshot is built.
    """

    params: dict[str, dict[str, Any]]
    polydisperse_param_names: list[str]
    polydisperse_params: dict[str, dict[str, Any]]
    pd_enabled: bool
    # Informational: the 'radius_effective' -> 'radius' constraint it describes
    # is carried by linked_params like any other link, so engines read that
    # instead of branching on this mode.
    radius_effective_mode: str
    structure_factor_name: str | None
    varying_params: list[str]
    varying_pd_params: list[str]
    # Equality links (follower -> target), canonical names. Populated by
    # link_params(), by the shared= sugar of set_models(), and by
    # radius_effective_mode='link_radius'.
    linked_params: dict[str, str] = field(default_factory=dict)
    # Composite-model components as (prefix, moniker, part_model_name)
    # triples; empty for atomic models.
    components: tuple[tuple[str, str, str], ...] = ()


def _validate_export_lengths(**arrays: Any) -> None:
    """Raise when export arrays do not all share the same length."""
    lengths = {name: len(values) for name, values in arrays.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) > 1:
        mismatch = ', '.join(f'{name}={length}' for name, length in lengths.items())
        raise ValueError(f'Cannot export fit results with mismatched array lengths: {mismatch}')


def resolve_fit_index(fit_index: Any, n_points: int) -> np.ndarray:
    """Normalize a stored fit index to a boolean array of length *n_points*.

    A missing index (None) means every point was fitted.
    """
    if fit_index is None:
        return np.ones(n_points, dtype=bool)
    index = np.asarray(fit_index, dtype=bool)
    if index.shape != (n_points,):
        raise ValueError(
            f'Fit index length ({index.size}) does not match the data length ({n_points}).'
        )
    return index


@dataclass(slots=True)
class PosteriorSummary:
    """Posterior sample chain and per-parameter statistics from a Bayesian fit.

    ``labels`` follows the sampler's chain order (``problem.labels()`` for
    bumps DREAM) and indexes the columns of ``samples``.
    """

    labels: list[str]
    samples: np.ndarray  # [n_samples, n_params]
    logp: np.ndarray | None = None  # [n_samples]
    chains: np.ndarray | None = None  # [n_generations, n_chains, n_params]
    best: dict[str, float] = field(default_factory=dict)
    mean: dict[str, float] = field(default_factory=dict)
    median: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)
    ci_68: dict[str, tuple[float, float]] = field(default_factory=dict)
    ci_95: dict[str, tuple[float, float]] = field(default_factory=dict)
    diagnostics: dict[str, dict[str, float]] | None = None

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    @property
    def n_params(self) -> int:
        return int(self.samples.shape[1])

    def index_of(self, param: str) -> int:
        """Return the chain column for a parameter name."""
        try:
            return self.labels.index(param)
        except ValueError:
            available = ', '.join(self.labels)
            raise KeyError(
                f"Parameter '{param}' is not part of the posterior sample. Available: {available}"
            ) from None

    def format_summary(self) -> str:
        """Return a table of per-parameter posterior statistics."""
        header = (
            f'{"Parameter":<20} {"Best":>12} {"Mean":>12} {"Median":>12} '
            f'{"Std":>12} {"68% CI":>26} {"95% CI":>26}'
        )
        lines = ['Posterior summary:', header, '-' * len(header)]
        for name in self.labels:
            lo68, hi68 = self.ci_68[name]
            lo95, hi95 = self.ci_95[name]
            lines.append(
                f'{name:<20} {self.best[name]:>12.6g} {self.mean[name]:>12.6g} '
                f'{self.median[name]:>12.6g} {self.std[name]:>12.6g} '
                f'{f"[{lo68:.6g}, {hi68:.6g}]":>26} {f"[{lo95:.6g}, {hi95:.6g}]":>26}'
            )
        if self.diagnostics is not None:
            lines.append('')
            lines.append(f'{"Parameter":<20} {"R-hat":>10} {"ESS":>10}')
            lines.append('-' * 42)
            for name in self.labels:
                stats = self.diagnostics.get(name, {})
                r_hat = stats.get('r_hat')
                ess = stats.get('ess')
                r_hat_text = f'{r_hat:.4f}' if r_hat is not None else 'n/a'
                ess_text = f'{ess:.0f}' if ess is not None else 'n/a'
                lines.append(f'{name:<20} {r_hat_text:>10} {ess_text:>10}')
        return '\n'.join(lines)

    def save_posterior_csv(self, filename: str) -> None:
        """Dump the raw posterior chain to CSV for external analysis."""
        columns = list(self.labels)
        data = [np.asarray(self.samples)]
        if self.logp is not None:
            columns.append('logp')
            data.append(np.asarray(self.logp).reshape(-1, 1))
        table = np.hstack(data)
        with open(filename, 'w') as f:
            f.write(','.join(columns) + '\n')
            for row in table:
                f.write(','.join(f'{value:.8e}' for value in row) + '\n')


@dataclass(slots=True)
class PosteriorDigest:
    """Posterior statistics without the sample chain.

    What a saved analysis restores. ``PosteriorSummary`` cannot serve here: it
    requires ``samples`` and derives its counts from that array, so a
    summary-only instance would either be impossible or would misreport having
    a chain. This carries exactly the read surface ``FitReport`` uses, and
    nothing that implies a chain is available.

    Sample-dependent displays (trace, pair and posterior-predictive plots)
    check for the real thing and raise; see ``SANSFitter.get_posterior``.
    """

    labels: list[str]
    n_samples: int
    n_params: int
    best: dict[str, float] = field(default_factory=dict)
    mean: dict[str, float] = field(default_factory=dict)
    median: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)
    ci_68: dict[str, tuple[float, float]] = field(default_factory=dict)
    ci_95: dict[str, tuple[float, float]] = field(default_factory=dict)
    diagnostics: dict[str, dict[str, float]] | None = None

    @classmethod
    def from_summary(cls, posterior: PosteriorSummary) -> 'PosteriorDigest':
        """Drop the chain from a full posterior, keeping every statistic."""
        return cls(
            labels=list(posterior.labels),
            n_samples=posterior.n_samples,
            n_params=posterior.n_params,
            best=dict(posterior.best),
            mean=dict(posterior.mean),
            median=dict(posterior.median),
            std=dict(posterior.std),
            ci_68={name: tuple(bounds) for name, bounds in posterior.ci_68.items()},
            ci_95={name: tuple(bounds) for name, bounds in posterior.ci_95.items()},
            diagnostics=posterior.diagnostics,
        )

    # The statistics table is identical whether or not the chain is present, so
    # the formatting lives on PosteriorSummary and is borrowed here rather than
    # duplicated. Only attributes this class also defines are touched.
    format_summary = PosteriorSummary.format_summary


@dataclass(slots=True)
class FitArtifacts:
    """Engine-specific runtime data needed after fitting."""

    fitted_curve: np.ndarray | None = None
    fit_index: np.ndarray | None = None
    raw_result: Any = None
    runtime_handle: Any = None
    runtime_key: str | None = None
    # The weighted residual vector the engine actually minimized, in the same
    # order and length as fitted_curve. Stored rather than re-derived because the
    # scipy engine unit-weights zero-dI points: recomputing (y - fit) / dy at
    # export time would write inf where the fit used a weight of 1.0, and the
    # exported residuals would no longer square-sum to the reported chisq.
    residuals: np.ndarray | None = None
    posterior: 'PosteriorSummary | PosteriorDigest | None' = None
    posterior_data: Any = None
    posterior_model_eval: Any = None
    # Per-component curves for '+' mixture models: label -> I(q) evaluated on
    # the same fit_index points as fitted_curve. None for atomic models and
    # '*' mixtures (where part curves would not stack to the total).
    component_curves: dict[str, np.ndarray] | None = None


@dataclass(slots=True)
class FitResultContract:
    """Stable internal fit-result contract used across post-fit operations.

    The goodness-of-fit block means the same thing on every engine:

    - ``chisq`` is the **raw** weighted sum of squared residuals over the fitted
      points, Σ((I − I_fit)/σ)², with σ the *effective* fit uncertainty (the scipy
      engine substitutes 1.0 where dI is zero). It carries data residuals only —
      no parameter prior or constraint penalty.
    - ``reduced_chisq`` is ``chisq / dof``, NaN when ``dof <= 0``.
    - ``dof`` is ``n_points - n_free`` on every path, computed here rather than
      taken from an engine, so a future addition of priors cannot silently change
      the public meaning on one engine only.

    Before 0.4 ``chisq`` held ``problem.chisq()`` (already χ²/dof) on the bumps
    path and the raw sum on the scipy path; the two were ~dof apart.
    """

    engine: str
    method: str
    chisq: float
    reduced_chisq: float
    n_points: int
    n_free: int
    dof: int
    weighting_note: str
    parameters: dict[str, dict[str, Any]]
    artifacts: FitArtifacts = field(default_factory=FitArtifacts)
    # The resolution setting this fit ran under, as
    # ``ResolutionSetting.as_dict()``. Recorded because a fitted parameter set
    # only means something alongside the smearing that produced it — and
    # because saved analyses will need to restore it.
    resolution: dict[str, Any] | None = None
    # Optimizer verdict: True/False when the engine reports one, None when it does
    # not. bumps hard-codes success=True regardless of the outcome, so None is the
    # only honest answer there.
    converged: bool | None = None
    message: str = ''
    # Covariance over the varied parameters, in cov_labels order. None when the
    # engine cannot supply one; never a zero matrix.
    cov: np.ndarray | None = None
    cov_labels: list[str] = field(default_factory=list)
    # Provenance of ``cov``: matrices from a Jacobian, from scipy's own cov_x and
    # from a posterior sample are not interchangeable and must not be shown under
    # one unqualified heading.
    cov_source: str | None = None
    # (parameter, 'min' | 'max') for each varied parameter sitting on a bound.
    on_bounds: list[tuple[str, str]] = field(default_factory=list)
    # The configuration and data this result belongs to, recorded when the fit
    # finished. No setter clears a fit result, so a fitter can hold a result
    # produced by settings the user has since changed; save_analysis() compares
    # this against the live configuration and omits a result that no longer
    # describes it. See sans_fitter.persistence.fit_context.
    fit_context: dict[str, Any] | None = None

    @property
    def is_preview(self) -> bool:
        """True for a theory preview (model at current values, no fit)."""
        return self.engine == PREVIEW_ENGINE

    def to_legacy_dict(self) -> dict[str, Any]:
        """Expose the historical dict-based result shape for public compatibility."""
        result = {
            'engine': self.engine,
            'method': self.method,
            'chisq': self.chisq,
            'reduced_chisq': self.reduced_chisq,
            'n_points': self.n_points,
            'n_free': self.n_free,
            'dof': self.dof,
            'converged': self.converged,
            'message': self.message,
            'weighting_note': self.weighting_note,
            'cov': None if self.cov is None else np.array(self.cov, copy=True),
            'cov_labels': list(self.cov_labels),
            'cov_source': self.cov_source,
            'on_bounds': list(self.on_bounds),
            'parameters': {name: dict(info) for name, info in self.parameters.items()},
        }

        if self.artifacts.raw_result is not None:
            result['result'] = self.artifacts.raw_result

        if self.artifacts.runtime_key and self.artifacts.runtime_handle is not None:
            result[self.artifacts.runtime_key] = self.artifacts.runtime_handle

        return result

    def require_fitted_curve(self) -> np.ndarray:
        """Return the fitted curve or raise if the contract is incomplete."""
        if self.artifacts.fitted_curve is None:
            raise ValueError('Fit result does not include a fitted curve.')
        return self.artifacts.fitted_curve

    def require_posterior(self) -> PosteriorSummary:
        """Return the posterior sample, or raise if this fit has none.

        A summary restored from a saved analysis is rejected here as firmly as
        no posterior at all: the statistics survive a save, the chain does not,
        and every caller of this method needs the chain itself.
        """
        posterior = self.artifacts.posterior
        if posterior is None:
            raise ValueError(
                'Fit result does not include a posterior sample. '
                'Run fit_bayesian() to enable Bayesian displays.'
            )
        if isinstance(posterior, PosteriorDigest):
            raise ValueError(
                'This analysis was loaded from a file, which stores posterior '
                'statistics but not the sample chain. The summary is available '
                'through get_fit_report(); re-run fit_bayesian() for displays '
                'that need the samples themselves.'
            )
        return posterior

    def save_csv(self, filename: str, model_name: str, data: Any) -> None:
        """Save fit results, fitted curve, and residuals to CSV.

        Only the points included in the fit (inside the Q range, unmasked)
        are exported, so every row carries a fitted intensity and residual.
        """
        fitted_curve = self.require_fitted_curve()
        has_dx = has_real_data(data.dx)

        index = resolve_fit_index(self.artifacts.fit_index, len(data.x))
        x = np.asarray(data.x)[index]
        y = np.asarray(data.y)[index]
        dy = np.asarray(data.dy)[index]
        dx = np.asarray(data.dx)[index] if has_dx else None

        arrays_to_validate = {
            'x': x,
            'y': y,
            'dy': dy,
            'fitted_curve': fitted_curve,
        }
        if has_dx:
            arrays_to_validate['dx'] = dx
        _validate_export_lengths(**arrays_to_validate)

        # The residuals the fit actually minimized, not a re-derivation. The scipy
        # engine unit-weights points whose dI is zero; dividing by the file's dy
        # here would export inf for those and break the identity
        # chisq == sum(residuals**2).
        if self.artifacts.residuals is not None:
            residuals = np.asarray(self.artifacts.residuals, dtype=float)
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                residuals = (y - fitted_curve) / dy
        _validate_export_lengths(residuals=residuals, **arrays_to_validate)

        # Explicit encoding: the default is the locale codepage, so on a
        # Windows console an export carrying any non-ASCII text (a model name,
        # a moniker) would fail late, at write time.
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('# SANS Fit Results\n')
            f.write(f'# Model: {model_name}\n')
            f.write(f'# Engine: {self.engine}\n')
            f.write(f'# Method: {self.method}\n')
            f.write(f'# Chi-squared: {self.chisq:.6f}\n')
            f.write(f'# Reduced chi-squared: {self.reduced_chisq:.6f}\n')
            if self.resolution is not None:
                f.write(f'# Resolution mode: {_format_resolution(self.resolution)}\n')
            f.write(f'# Q range: {x.min():.6g} to {x.max():.6g}\n')
            f.write(f'# Points fitted: {len(x)} of {len(index)}\n')
            f.write(f'# Free parameters: {self.n_free}\n')
            f.write(f'# Degrees of freedom: {self.dof}\n')
            f.write(f'# Weighting: {self.weighting_note}\n')
            if self.converged is not None or self.message:
                verdict = {True: 'yes', False: 'no', None: 'not reported'}[self.converged]
                f.write(f'# Converged: {verdict}\n')
                if self.message:
                    f.write(f'# Optimizer message: {self.message}\n')
            if self.on_bounds:
                hits = ', '.join(f'{name} ({side})' for name, side in self.on_bounds)
                f.write(f'# Parameters at a bound: {hits}\n')
            f.write('#\n')
            f.write('# Fitted Parameters:\n')
            for name, info in self.parameters.items():
                f.write(f'# {name}: {info["formatted"]}\n')
            posterior = self.artifacts.posterior
            if posterior is not None:
                f.write('#\n')
                f.write('# Posterior credible intervals:\n')
                for name in posterior.labels:
                    lo68, hi68 = posterior.ci_68[name]
                    lo95, hi95 = posterior.ci_95[name]
                    f.write(
                        f'# {name}: 68% CI [{lo68:.6g}, {hi68:.6g}], '
                        f'95% CI [{lo95:.6g}, {hi95:.6g}]\n'
                    )
            f.write('#\n')

            if has_dx:
                # The exported dQ is the dataset's own column. Under a custom
                # resolution mode it is not the width the fit smeared with —
                # that is on the '# Resolution mode' line above.
                f.write('Q,dQ,I_exp,dI_exp,I_fit,Residuals\n')
                for q, dq, i_exp, di_exp, i_fit, res in zip(
                    x, dx, y, dy, fitted_curve, residuals, strict=True
                ):
                    f.write(f'{q:.6e},{dq:.6e},{i_exp:.6e},{di_exp:.6e},{i_fit:.6e},{res:.6e}\n')
            else:
                f.write('Q,I_exp,dI_exp,I_fit,Residuals\n')
                for q, i_exp, di_exp, i_fit, res in zip(
                    x, y, dy, fitted_curve, residuals, strict=True
                ):
                    f.write(f'{q:.6e},{i_exp:.6e},{di_exp:.6e},{i_fit:.6e},{res:.6e}\n')


def save_fit_result(
    filename: str, model_name: str, data: Any, fit_result: FitResultContract
) -> None:
    """Compatibility wrapper for saving fit results from SANSFitter."""
    fit_result.save_csv(filename=filename, model_name=model_name, data=data)

"""Results of a simultaneous fit: one global answer with per-dataset detail.

Deliberately a new contract rather than lists stuffed into the single-fit
dictionary. Three of its fields exist because a joint fit can be misread in ways
a single fit cannot:

- ``chisq`` and ``objective`` are separate. They coincide until a dataset factor
  is set, after which one is the goodness of fit and the other is the thing that
  was minimised, and neither is a substitute for the other.
- every parameter carries a ``status`` and a ``root``. A shared radius appears
  once as a fitted coordinate and once per dataset as a member of it; without
  saying which is which, a reader counts the same measurement twice.
- ``stderr`` is ``None`` when it is *unknown*, not ``0.0``. Zero is what a fixed
  parameter has, and conflating the two turns a failed error estimate into a
  suspiciously precise one. A rank-deficient problem is reported two ways
  depending on the weighting, for reasons given in
  :func:`sans_fitter.fitting.multi_bumps._covariance`: under unit weights the
  clamped Jacobian estimate is kept (so a one-entry fit still matches
  ``SANSFitter``) with the deficiency stated in ``cov_note`` and warned about;
  under priority weights there is no covariance at all. Either way ``cov_note``
  is the field that says so — a large error bar is never left to speak for
  itself.
"""

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:  # bumps >= 1.0.4
    from bumps.util import format_uncertainty
except ImportError:  # bumps <= 1.0.3
    from bumps.formatnum import format_uncertainty

from .console import ARROW, CHI_SQUARED
from .fitting.base import correlation_matrix
from .report import (
    DEFAULT_CORRELATION_THRESHOLD,
    escape_html_cell,
    escape_markdown_cell,
    json_safe,
)

__all__ = ['MultiParameter', 'DatasetResult', 'MultiFitResult', 'MultiFitReport']

#: Parameter statuses, in the order the documentation introduces them.
PARAMETER_STATUSES = ('free', 'shared', 'derived', 'fixed')


def _number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f'{number:.6g}' if math.isfinite(number) else 'n/a'


def format_estimate(value: float, stderr: float | None) -> str:
    """Render a value with its uncertainty, in the engine's own convention.

    ``None`` means the uncertainty could not be estimated and says so, which is
    different from a fixed parameter's exact zero.
    """
    if stderr is None:
        return f'{value:.6g} (no uncertainty available)'
    if stderr == 0.0:
        return f'{value:.6g}'
    return str(format_uncertainty(value, stderr))


@dataclass(slots=True)
class MultiParameter:
    """One parameter of one dataset, as the joint fit resolved it."""

    qualified: str
    dataset: str
    name: str
    value: float
    #: None when no uncertainty could be estimated; 0.0 only when genuinely fixed.
    stderr: float | None
    formatted: str
    #: 'free', 'shared', 'derived' or 'fixed'.
    status: str
    #: Qualified label of the class this parameter belongs to.
    root: str
    #: Every qualified name that is this same quantity, including this one.
    members: tuple[str, ...] = ()
    #: Constraint text for a derived or constant parameter, '' otherwise.
    expression: str = ''
    #: Where the uncertainty came from, or why there is none.
    uncertainty_source: str = ''

    @property
    def is_root(self) -> bool:
        """True for the member whose qualified name labels the class."""
        return self.qualified == self.root

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                'qualified': self.qualified,
                'dataset': self.dataset,
                'name': self.name,
                'value': self.value,
                'stderr': self.stderr,
                'formatted': self.formatted,
                'status': self.status,
                'root': self.root,
                'members': list(self.members),
                'expression': self.expression,
                'uncertainty_source': self.uncertainty_source,
            }
        )


@dataclass(slots=True)
class DatasetResult:
    """One dataset's share of a joint fit, including the data it was fitted to.

    ``chisq`` here is the raw sum over this dataset's points; it is a
    diagnostic, not a goodness-of-fit test, because the degrees of freedom of a
    joint fit belong to the fit and cannot be split between datasets. See
    :attr:`mean_squared_residual` for the per-point figure that can be compared
    across datasets.

    **The observations are copied in, not referenced.** A result outlives the
    configuration that produced it: datasets can be reconfigured, removed, or
    replaced by different measurements under the same name. Plots and exports
    built from this snapshot therefore always show one coherent fit, rather than
    this fit's model curve beside whatever data the fitter holds now.
    """

    name: str
    model: str
    n_points: int
    chisq: float
    objective_contribution: float
    weight: float
    q_range: tuple[float, float] | None
    resolution: str
    parameters: dict[str, MultiParameter]
    #: Full-length copies of the measured arrays, as they were when fitted.
    observed_q: np.ndarray
    observed_intensity: np.ndarray
    observed_uncertainty: np.ndarray | None
    observed_dq: np.ndarray | None
    #: Model intensities at the fitted points only.
    fitted_curve: np.ndarray
    #: Which of the observed points were fitted; None means all of them.
    fit_index: np.ndarray | None
    #: Residuals in the dataset's own sigma units, (model - I)/dI.
    residuals: np.ndarray
    #: sqrt(weight) times the above: what the optimizer actually minimised.
    objective_residuals: np.ndarray
    #: Per-component curves for a '+' mixture model, on the fitted points.
    #: None for atomic models and '*' mixtures, where the parts do not stack.
    component_curves: dict[str, np.ndarray] | None = None

    def selection(self) -> np.ndarray:
        """Boolean mask of the fitted points over the observed arrays."""
        if self.fit_index is None:
            return np.ones(len(self.observed_q), dtype=bool)
        return np.asarray(self.fit_index, dtype=bool)

    @property
    def rms_residual(self) -> float:
        """Root mean squared normalized residual, sqrt(chi2_d / N_d)."""
        return math.sqrt(self.chisq / self.n_points) if self.n_points else float('nan')

    @property
    def mean_squared_residual(self) -> float:
        """chi2_d / N_d. **Not** a reduced chi-squared: no dof is allocated here."""
        return self.chisq / self.n_points if self.n_points else float('nan')

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                'name': self.name,
                'model': self.model,
                'n_points': self.n_points,
                'chisq': self.chisq,
                'objective_contribution': self.objective_contribution,
                'rms_residual': self.rms_residual,
                'mean_squared_residual': self.mean_squared_residual,
                'weight': self.weight,
                'q_range': None if self.q_range is None else list(self.q_range),
                'resolution': self.resolution,
                'parameters': {name: entry.to_dict() for name, entry in self.parameters.items()},
            }
        )


@dataclass(slots=True)
class MultiFitResult:
    """The result of one simultaneous fit.

    Returned by :meth:`sans_fitter.MultiFitter.fit`, which hands back the same
    object it retains as ``fitter.result`` — there is one result per fit, not a
    copy per caller. It is **caller-owned and mutable**: nothing stops you
    editing its dictionaries or arrays, and if you do, the fitter's view changes
    with it. Take a ``copy.deepcopy`` before modifying one if you need the two
    to diverge, or work from :meth:`to_dict`, which is already a fresh
    structure.

    What the result does *not* share with the fitter is the analysis it came
    from: the observations, the model curves, the selection and every parameter
    value are copied in at fit time, so reconfiguring or removing a dataset
    afterwards cannot change what this result reports.
    """

    engine: str
    method: str
    converged: bool | None
    message: str
    n_datasets: int
    n_points: int
    n_free: int
    dof: int
    chisq: float
    reduced_chisq: float
    objective: float
    reduced_objective: float
    #: Dataset name -> objective coefficient.
    weighting: dict[str, float]
    #: One entry per explicit constraint, as ``{'target', 'kind', 'text'}``.
    constraints: list[dict[str, Any]] = field(default_factory=list)
    #: One entry per multi-member class, as ``{'root', 'members', 'status'}``.
    sharing: list[dict[str, Any]] = field(default_factory=list)
    parameters: dict[str, MultiParameter] = field(default_factory=dict)
    cov: np.ndarray | None = None
    cov_labels: list[str] = field(default_factory=list)
    cov_source: str | None = None
    #: Why there is no covariance, when there is none.
    cov_note: str = ''
    on_bounds: list[tuple[str, str]] = field(default_factory=list)
    datasets: dict[str, DatasetResult] = field(default_factory=dict)
    fit_context: dict[str, Any] | None = None

    @property
    def weighted(self) -> bool:
        """True when any dataset carries a factor other than 1."""
        return any(not math.isclose(value, 1.0) for value in self.weighting.values())

    @property
    def corr(self) -> np.ndarray | None:
        """Correlation matrix over the free roots, or None when unavailable."""
        return None if self.cov is None else correlation_matrix(self.cov)

    def root_parameters(self) -> list[MultiParameter]:
        """One entry per distinct quantity, whether or not the fit varied it.

        A shared radius appears here once, under its root label; its per-dataset
        members are in :attr:`parameters`. This is the list to report or tabulate
        — iterating :attr:`parameters` would count a shared measurement once per
        dataset.
        """
        return [entry for entry in self.parameters.values() if entry.is_root]

    def free_parameters(self) -> list[MultiParameter]:
        """The quantities the optimizer actually moved, each reported once."""
        return [entry for entry in self.root_parameters() if entry.status in ('free', 'shared')]

    def to_dict(self) -> dict[str, Any]:
        """The whole result as strictly JSON-serializable data."""
        return json_safe(
            {
                'engine': self.engine,
                'method': self.method,
                'converged': self.converged,
                'message': self.message,
                'n_datasets': self.n_datasets,
                'n_points': self.n_points,
                'n_free': self.n_free,
                'dof': self.dof,
                'chisq': self.chisq,
                'reduced_chisq': self.reduced_chisq,
                'objective': self.objective,
                'reduced_objective': self.reduced_objective,
                'weighting': self.weighting,
                'constraints': self.constraints,
                'sharing': self.sharing,
                'parameters': {name: entry.to_dict() for name, entry in self.parameters.items()},
                'cov': self.cov,
                'cov_labels': list(self.cov_labels),
                'cov_source': self.cov_source,
                'cov_note': self.cov_note,
                'on_bounds': [{'parameter': name, 'bound': side} for name, side in self.on_bounds],
                'datasets': {name: entry.to_dict() for name, entry in self.datasets.items()},
                'fit_context': self.fit_context,
            }
        )

    def __str__(self) -> str:
        return str(MultiFitReport(self))

    def _repr_html_(self) -> str:
        return MultiFitReport(self)._repr_html_()


class MultiFitReport:
    """Self-rendering summary of a :class:`MultiFitResult`.

    Text for a terminal, HTML for a notebook cell, Markdown for a document —
    the same three renderings :class:`sans_fitter.report.FitReport` provides for
    a single fit, plus the two tables a joint fit needs: which parameters are
    the same quantity, and what each dataset contributed.
    """

    def __init__(self, result: MultiFitResult) -> None:
        self.result = result

    # ------------------------------------------------------------------
    # Rows
    # ------------------------------------------------------------------

    def _quality_rows(self, symbol: str) -> list[tuple[str, str]]:
        result = self.result
        rows = [
            ('Datasets', str(result.n_datasets)),
            (f'{symbol}/dof', _number(result.reduced_chisq)),
            (symbol, _number(result.chisq)),
        ]
        if result.weighted:
            rows.extend(
                [
                    ('Weighted objective', _number(result.objective)),
                    ('Weighted objective/dof', _number(result.reduced_objective)),
                ]
            )
        rows.extend(
            [
                ('Points fitted', str(result.n_points)),
                ('Free parameters', str(result.n_free)),
                ('Degrees of freedom', str(result.dof)),
                (
                    'Converged',
                    'not reported'
                    if result.converged is None
                    else ('yes' if result.converged else 'no'),
                ),
            ]
        )
        if result.message:
            rows.append(('Optimizer message', result.message))
        return rows

    def _bound_side(self, name: str) -> str | None:
        for hit_name, side in self.result.on_bounds:
            if hit_name == name:
                return side
        return None

    def _root_rows(self) -> list[tuple[str, str, str]]:
        """(label, estimate, status) for each distinct quantity, once each."""
        rows: list[tuple[str, str, str]] = []
        for entry in self.result.root_parameters():
            status = entry.status
            if entry.status == 'derived':
                status = f'derived = {entry.expression}'
            elif entry.status == 'fixed' and entry.expression:
                status = f'fixed = {entry.expression}'
            elif entry.status in ('free', 'shared'):
                side = self._bound_side(entry.qualified)
                if side is not None:
                    status = f'{status}, on bound ({side})'
            rows.append((entry.qualified, entry.formatted, status))
        return rows

    def _sharing_rows(self, arrow: str = ARROW) -> list[tuple[str, str]]:
        return [
            (str(group['root']), f'{arrow} ' + ', '.join(group['members']))
            for group in self.result.sharing
        ]

    def _dataset_rows(self) -> list[tuple[str, ...]]:
        rows: list[tuple[str, ...]] = []
        for entry in self.result.datasets.values():
            q_range = (
                'full'
                if entry.q_range is None
                else f'{entry.q_range[0]:.4g}-{entry.q_range[1]:.4g}'
            )
            rows.append(
                (
                    entry.name,
                    entry.model,
                    str(entry.n_points),
                    _number(entry.chisq),
                    _number(entry.mean_squared_residual),
                    _number(entry.weight),
                    q_range,
                    entry.resolution,
                )
            )
        return rows

    DATASET_HEADERS = (
        'Dataset',
        'Model',
        'Points',
        CHI_SQUARED,
        f'{CHI_SQUARED}/N',
        'Weight',
        'Q range',
        'Resolution',
    )

    # ------------------------------------------------------------------
    # Renderers
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        result = self.result
        lines = [
            f'Simultaneous fit report: {result.n_datasets} datasets · '
            f'{result.engine}/{result.method}',
            '=' * 78,
        ]

        quality = self._quality_rows(CHI_SQUARED)
        width = max(len(label) for label, _ in quality)
        lines.extend(f'{label:<{width}}  {value}' for label, value in quality)

        rows = self._root_rows()
        if rows:
            lines.extend(['', 'Parameters (one row per distinct quantity):'])
            name_width = max(9, *(len(name) for name, _, _ in rows))
            estimate_width = max(8, *(len(estimate) for _, estimate, _ in rows))
            header = f'{"Parameter":<{name_width}}  {"Estimate":<{estimate_width}}  Status'
            lines.extend([header, '-' * len(header)])
            lines.extend(
                f'{name:<{name_width}}  {estimate:<{estimate_width}}  {status}'
                for name, estimate, status in rows
            )

        sharing = self._sharing_rows()
        if sharing:
            lines.extend(['', 'Shared quantities:'])
            width = max(len(root) for root, _ in sharing)
            lines.extend(f'{root:<{width}}  {members}' for root, members in sharing)

        dataset_rows = self._dataset_rows()
        if dataset_rows:
            lines.extend(['', 'Per dataset:'])
            widths = [
                max(len(str(header)), *(len(row[i]) for row in dataset_rows))
                for i, header in enumerate(self.DATASET_HEADERS)
            ]
            header = '  '.join(
                f'{name:<{widths[i]}}' for i, name in enumerate(self.DATASET_HEADERS)
            )
            lines.extend([header, '-' * len(header)])
            lines.extend(
                '  '.join(f'{cell:<{widths[i]}}' for i, cell in enumerate(row))
                for row in dataset_rows
            )
            lines.append(
                f'{CHI_SQUARED}/N is a mean squared normalized residual, not a reduced '
                f'{CHI_SQUARED}: the {result.dof} degrees of freedom belong to the '
                'joint fit.'
            )

        if result.weighted:
            lines.extend(
                [
                    '',
                    'Dataset factors are fitting priorities; the supplied dI remains the '
                    'error model, so uncertainties use the known-error sandwich '
                    'covariance.',
                ]
            )

        if result.cov is None:
            if result.cov_note:
                lines.extend(['', f'No covariance: {result.cov_note}'])
        else:
            lines.extend(['', f'Covariance source: {result.cov_source}'])
            # A note alongside a covariance is a warning about that covariance,
            # not an explanation of its absence, and must not be swallowed.
            if result.cov_note:
                lines.append(f'  WARNING: {result.cov_note}')
            strong = self.strongly_correlated()
            lines.extend(f'  strongly correlated: {a} / {b} ({rho:+.2f})' for a, b, rho in strong)

        if result.on_bounds:
            hits = ', '.join(f'{name} ({side})' for name, side in result.on_bounds)
            lines.extend(['', f'At a bound: {hits}'])

        return '\n'.join(lines)

    def strongly_correlated(
        self, threshold: float = DEFAULT_CORRELATION_THRESHOLD
    ) -> list[tuple[str, str, float]]:
        """Root pairs whose correlation magnitude reaches *threshold*."""
        corr = self.result.corr
        if corr is None:
            return []
        labels = self.result.cov_labels
        pairs: list[tuple[str, str, float]] = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                rho = float(corr[i, j])
                if math.isfinite(rho) and abs(rho) >= threshold:
                    pairs.append((labels[i], labels[j], rho))
        return sorted(pairs, key=lambda item: abs(item[2]), reverse=True)

    def to_markdown(self) -> str:
        """Markdown report, for a document, an issue comment or a pull request."""
        result = self.result
        cell = escape_markdown_cell
        out = [
            f'**Simultaneous fit:** {result.n_datasets} datasets · '
            f'{cell(result.engine)}/{cell(result.method)}',
            '',
            '| Statistic | Value |',
            '|---|---|',
        ]
        out.extend(
            f'| {cell(label)} | {cell(value)} |' for label, value in self._quality_rows('χ²')
        )

        rows = self._root_rows()
        if rows:
            out.extend(['', '| Parameter | Estimate | Status |', '|---|---|---|'])
            out.extend(
                f'| `{cell(name)}` | {cell(estimate)} | {cell(status)} |'
                for name, estimate, status in rows
            )

        sharing = self._sharing_rows(arrow='=')
        if sharing:
            out.extend(['', '| Quantity | Members |', '|---|---|'])
            out.extend(f'| `{cell(root)}` | {cell(members)} |' for root, members in sharing)

        dataset_rows = self._dataset_rows()
        if dataset_rows:
            headers = (
                'Dataset',
                'Model',
                'Points',
                'χ²',
                'χ²/N',
                'Weight',
                'Q range',
                'Resolution',
            )
            out.extend(['', '| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)])
            out.extend(
                '| ' + ' | '.join(cell(value) for value in row) + ' |' for row in dataset_rows
            )
            out.append('')
            out.append(
                f'χ²/N is a mean squared normalized residual, not a reduced χ²: the '
                f'{result.dof} degrees of freedom belong to the joint fit.'
            )

        if result.cov is None:
            if result.cov_note:
                out.extend(['', f'**No covariance:** {cell(result.cov_note)}'])
        else:
            out.extend(['', f'Covariance source: `{cell(result.cov_source)}`'])
            if result.cov_note:
                out.append(f'- **Warning:** {cell(result.cov_note)}')
            out.extend(
                f'- Strongly correlated: `{cell(a)}` / `{cell(b)}` ({rho:+.2f})'
                for a, b, rho in self.strongly_correlated()
            )

        if result.on_bounds:
            hits = ', '.join(f'`{cell(name)}` ({side})' for name, side in result.on_bounds)
            out.extend(['', f'At a bound: {hits}'])

        return '\n'.join(out)

    def _repr_html_(self) -> str:
        """HTML report, so a notebook cell shows tables instead of text."""
        result = self.result
        style = 'border-collapse:collapse;margin-bottom:0.75em'
        cell = 'padding:2px 10px;border:1px solid #ddd;text-align:left'
        head = 'padding:2px 10px;border:1px solid #ddd;text-align:left;background:#f5f5f5'

        parts = [
            '<div><p><strong>Simultaneous fit:</strong> '
            f'{result.n_datasets} datasets · '
            f'{escape_html_cell(result.engine)}/{escape_html_cell(result.method)}</p>'
        ]

        parts.append(f'<table style="{style}">')
        for label, value in self._quality_rows('χ²'):
            parts.append(
                f'<tr><th style="{head}">{escape_html_cell(label)}</th>'
                f'<td style="{cell}">{escape_html_cell(value)}</td></tr>'
            )
        parts.append('</table>')

        parts.append(
            self._html_table(
                ('Parameter', 'Estimate', 'Status'), self._root_rows(), style, cell, head
            )
        )
        sharing = self._sharing_rows(arrow='=')
        if sharing:
            parts.append(self._html_table(('Quantity', 'Members'), sharing, style, cell, head))

        dataset_rows = self._dataset_rows()
        if dataset_rows:
            headers = (
                'Dataset',
                'Model',
                'Points',
                'χ²',
                'χ²/N',
                'Weight',
                'Q range',
                'Resolution',
            )
            parts.append(self._html_table(headers, dataset_rows, style, cell, head))
            parts.append(
                '<p><small>χ²/N is a mean squared normalized residual, not a reduced '
                f'χ²: the {result.dof} degrees of freedom belong to the joint fit.'
                '</small></p>'
            )

        if result.cov is None:
            if result.cov_note:
                parts.append(
                    f'<p><strong>No covariance:</strong> {escape_html_cell(result.cov_note)}</p>'
                )
        else:
            parts.append(f'<p>Covariance source: {escape_html_cell(result.cov_source)}</p>')
            if result.cov_note:
                parts.append(
                    f'<p><strong>Warning:</strong> {escape_html_cell(result.cov_note)}</p>'
                )
            strong = self.strongly_correlated()
            if strong:
                items = ''.join(
                    f'<li>{escape_html_cell(a)} / {escape_html_cell(b)} ({rho:+.2f})</li>'
                    for a, b, rho in strong
                )
                parts.append(f'<p>Strongly correlated:</p><ul>{items}</ul>')

        if result.on_bounds:
            hits = ', '.join(
                f'{escape_html_cell(name)} ({side})' for name, side in result.on_bounds
            )
            parts.append(f'<p><strong>At a bound:</strong> {hits}</p>')

        parts.append('</div>')
        return ''.join(parts)

    @staticmethod
    def _html_table(
        headers: tuple[str, ...],
        rows: list[Any],
        style: str,
        cell: str,
        head: str,
    ) -> str:
        if not rows:
            return ''
        parts = [f'<table style="{style}"><tr>']
        parts.extend(f'<th style="{head}">{escape_html_cell(name)}</th>' for name in headers)
        parts.append('</tr>')
        for row in rows:
            parts.append('<tr>')
            parts.extend(f'<td style="{cell}">{escape_html_cell(value)}</td>' for value in row)
            parts.append('</tr>')
        parts.append('</table>')
        return ''.join(parts)

    def to_dict(self) -> dict[str, Any]:
        """The underlying result as JSON-serializable data."""
        return self.result.to_dict()

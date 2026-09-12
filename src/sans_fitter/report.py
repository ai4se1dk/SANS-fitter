"""Fit-quality reporting: the :class:`FitReport` object and its renderers.

A fit returns a dictionary, which is convenient for code and poor for reading. This
module carries the same information as an object that renders itself three ways —
plain text for a terminal, HTML for a notebook cell, Markdown for a document — plus
:meth:`FitReport.to_dict` for serialization.

The report is a **snapshot**: it copies the parameter block and the covariance out
of the fit contract, so a later fit (or a caller mutating the returned dictionary)
cannot change a report already handed out.
"""

import copy
import html
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .console import ARROW, CHI_SQUARED
from .data.resolution import ResolutionSetting
from .fitting.base import correlation_matrix
from .results import FitResultContract, PosteriorSummary

#: |ρ| at or above this counts as a strong correlation in the report.
DEFAULT_CORRELATION_THRESHOLD = 0.95

__all__ = ['FitReport', 'DEFAULT_CORRELATION_THRESHOLD']


def json_safe(value: Any) -> Any:
    """Convert *value* into something ``json.dumps(..., allow_nan=False)`` accepts.

    Walks dictionaries, sequences and arrays; turns NumPy scalars into Python
    scalars and every non-finite float into None. Recursion matters: NaN can appear
    in the goodness-of-fit fields, in a parameter's value or stderr, in a
    correlation cell, and in a posterior R-hat, and a single missed branch makes
    the whole document unserializable.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


def escape_markdown_cell(text: Any) -> str:
    """Escape *text* for a Markdown table cell.

    Pipes would start a new column and a newline would end the row, so both are
    neutralized. Model names come from user-chosen ``set_models`` monikers and
    messages come from scipy, so neither is trusted to be table-safe.
    """
    rendered = str(text)
    rendered = rendered.replace('\\', '\\\\').replace('|', '\\|')
    return ' '.join(rendered.split())


def escape_html_cell(text: Any) -> str:
    """Escape *text* for an HTML table cell."""
    return html.escape(str(text), quote=True)


def _format_number(value: Any) -> str:
    """Render a number for a report cell, with 'n/a' for anything non-finite."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f'{number:.6g}' if math.isfinite(number) else 'n/a'


def _verdict(converged: bool | None) -> str:
    if converged is None:
        return 'not reported'
    return 'yes' if converged else 'no'


@dataclass(slots=True)
class FitReport:
    """Goodness-of-fit summary for one completed fit.

    Built by :meth:`sans_fitter.SANSFitter.get_fit_report`. Not exported at the
    top level of the package, but a public type all the same: the import path
    ``sans_fitter.report.FitReport`` and the methods below are stable.
    """

    model: str
    engine: str
    method: str
    resolution: str
    weighting_note: str
    chisq: float
    reduced_chisq: float
    n_points: int
    n_free: int
    dof: int
    converged: bool | None
    message: str
    parameters: dict[str, dict[str, Any]]
    cov_labels: list[str] = field(default_factory=list)
    cov: np.ndarray | None = None
    cov_source: str | None = None
    on_bounds: list[tuple[str, str]] = field(default_factory=list)
    posterior: PosteriorSummary | None = None

    def __post_init__(self) -> None:
        if self.cov is not None:
            matrix = np.asarray(self.cov, dtype=float)
            expected = (len(self.cov_labels), len(self.cov_labels))
            if matrix.ndim != 2 or matrix.shape != expected:
                raise ValueError(
                    f'Covariance matrix shape {matrix.shape} does not match the '
                    f'{len(self.cov_labels)} label(s) {self.cov_labels} '
                    f'(expected {expected}).'
                )
            self.cov = matrix

    @classmethod
    def from_contract(cls, contract: FitResultContract, model_name: str) -> 'FitReport':
        """Snapshot *contract* into a report.

        Nothing mutable is shared with the fitter: the parameter block is copied
        one level deep (each entry becomes a fresh dictionary), the covariance is
        copied, and the posterior is deep-copied — its ``samples`` array and its
        per-parameter statistic and diagnostic dictionaries are all mutable, and a
        caller holding the object from ``get_posterior()`` would otherwise be able
        to change a report that was already handed out.
        """
        resolution = 'not recorded'
        if contract.resolution is not None:
            try:
                resolution = ResolutionSetting(**contract.resolution).describe()
            except TypeError:  # a setting shape from another version — show it raw
                resolution = str(contract.resolution)

        return cls(
            model=model_name or 'unknown model',
            engine=contract.engine,
            method=contract.method,
            resolution=resolution,
            weighting_note=contract.weighting_note,
            chisq=contract.chisq,
            reduced_chisq=contract.reduced_chisq,
            n_points=contract.n_points,
            n_free=contract.n_free,
            dof=contract.dof,
            converged=contract.converged,
            message=contract.message,
            parameters={name: dict(info) for name, info in contract.parameters.items()},
            cov_labels=list(contract.cov_labels),
            cov=None if contract.cov is None else np.array(contract.cov, copy=True),
            cov_source=contract.cov_source,
            on_bounds=[tuple(hit) for hit in contract.on_bounds],
            posterior=(
                None
                if contract.artifacts.posterior is None
                else copy.deepcopy(contract.artifacts.posterior)
            ),
        )

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def corr(self) -> np.ndarray | None:
        """Correlation matrix derived from :attr:`cov`, or None when unavailable."""
        if self.cov is None:
            return None
        return correlation_matrix(self.cov)

    def strongly_correlated(
        self, threshold: float = DEFAULT_CORRELATION_THRESHOLD
    ) -> list[tuple[str, str, float]]:
        """Parameter pairs whose correlation magnitude reaches *threshold*.

        Walks the strict upper triangle, so each pair appears once, and skips cells
        that are not finite (a zero-variance dimension). The returned coefficient
        keeps its sign: an anti-correlation of -0.99 is as informative as +0.99.

        Args:
            threshold: Magnitude at or above which a pair is reported, in [0, 1].

        Raises:
            ValueError: If *threshold* lies outside [0, 1].
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f'threshold must be between 0 and 1, got {threshold}.')
        corr = self.corr
        if corr is None:
            return []

        pairs: list[tuple[str, str, float]] = []
        for i in range(len(self.cov_labels)):
            for j in range(i + 1, len(self.cov_labels)):
                rho = float(corr[i, j])
                if math.isfinite(rho) and abs(rho) >= threshold:
                    pairs.append((self.cov_labels[i], self.cov_labels[j], rho))
        return sorted(pairs, key=lambda item: abs(item[2]), reverse=True)

    def _bound_side(self, name: str) -> str | None:
        for hit_name, side in self.on_bounds:
            if hit_name == name:
                return side
        return None

    def _parameter_rows(self, arrow: str = ARROW) -> list[tuple[str, str, str]]:
        """(name, estimate, status) for every parameter, in contract order.

        A fitted parameter's estimate is the engine's own ``formatted`` string, so
        the bumps ``45.041(46)`` convention survives. A fixed or linked one is
        rendered as a plain number, because its ``formatted`` string carries a
        ``(fixed)`` / ``(linked)`` suffix that the status column already states.
        """
        rows: list[tuple[str, str, str]] = []
        for name, info in self.parameters.items():
            linked_to = info.get('linked_to')
            is_fixed = info.get('fixed', False)

            if linked_to:
                status = f'linked {arrow} {linked_to}'
                estimate = _format_number(info.get('value'))
            elif is_fixed:
                status = 'fixed'
                estimate = _format_number(info.get('value'))
            else:
                status = 'fitted'
                side = self._bound_side(name)
                if side is not None:
                    status = f'fitted, on bound ({side})'
                estimate = str(info.get('formatted', _format_number(info.get('value'))))
            rows.append((name, estimate, status))
        return rows

    def _quality_rows(self, symbol: str) -> list[tuple[str, str]]:
        rows = [
            (f'{symbol}/dof', _format_number(self.reduced_chisq)),
            (symbol, _format_number(self.chisq)),
            ('Points fitted', str(self.n_points)),
            ('Free parameters', str(self.n_free)),
            ('Degrees of freedom', str(self.dof)),
            ('Converged', _verdict(self.converged)),
        ]
        if self.message:
            rows.append(('Optimizer message', self.message))
        return rows

    def _header(self) -> str:
        return (
            f'{self.model} · {self.engine}/{self.method} · '
            f'resolution: {self.resolution} · weighting: {self.weighting_note}'
        )

    def _show_correlations(self) -> bool:
        return self.cov is not None and len(self.cov_labels) > 1

    # ------------------------------------------------------------------
    # Renderers
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """Plain-text report, using the console's glyph fallbacks.

        Safe to print on a console that cannot encode χ²: the symbol comes from
        ``console.CHI_SQUARED``, which degrades to ASCII.
        """
        symbol = CHI_SQUARED
        lines = [f'Fit report: {self._header()}', '=' * 72]

        quality = self._quality_rows(symbol)
        width = max(len(label) for label, _ in quality)
        lines.extend(f'{label:<{width}}  {value}' for label, value in quality)

        rows = self._parameter_rows()
        if rows:
            lines.extend(['', 'Parameters:'])
            name_width = max(9, *(len(name) for name, _, _ in rows))
            estimate_width = max(8, *(len(estimate) for _, estimate, _ in rows))
            header = f'{"Parameter":<{name_width}}  {"Estimate":<{estimate_width}}  Status'
            lines.extend([header, '-' * len(header)])
            lines.extend(
                f'{name:<{name_width}}  {estimate:<{estimate_width}}  {status}'
                for name, estimate, status in rows
            )

        if self._show_correlations():
            corr = self.corr
            lines.extend(['', f'Correlations (source: {self.cov_source}):'])
            label_width = max(len(label) for label in self.cov_labels)
            cell_width = max(6, label_width)
            head = ' ' * label_width + ''.join(
                f'  {label:>{cell_width}}' for label in self.cov_labels
            )
            lines.extend([head, '-' * len(head)])
            for i, label in enumerate(self.cov_labels):
                cells = ''.join(
                    f'  {_correlation_text(corr[i, j]):>{cell_width}}'
                    for j in range(len(self.cov_labels))
                )
                lines.append(f'{label:<{label_width}}{cells}')
            strong = self.strongly_correlated()
            if strong:
                lines.append('')
                lines.extend(
                    f'  strongly correlated: {a} / {b} ({rho:+.2f})' for a, b, rho in strong
                )

        if self.on_bounds:
            hits = ', '.join(f'{name} ({side})' for name, side in self.on_bounds)
            lines.extend(['', f'At a bound: {hits}'])

        if self.posterior is not None:
            lines.extend(['', self.posterior.format_summary()])

        return '\n'.join(lines)

    def to_markdown(self) -> str:
        """Markdown report, for a document, an issue comment or a pull request.

        Writes the Unicode χ² directly: the result is a string the caller places,
        not console output, so the ASCII fallback would only make it uglier.
        """
        symbol = 'χ²'
        out = [f'**Fit report:** {escape_markdown_cell(self._header())}', '']
        out.extend(['| Statistic | Value |', '|---|---|'])
        out.extend(
            f'| {escape_markdown_cell(label)} | {escape_markdown_cell(value)} |'
            for label, value in self._quality_rows(symbol)
        )

        rows = self._parameter_rows(arrow='->')
        if rows:
            out.extend(['', '| Parameter | Estimate | Status |', '|---|---|---|'])
            out.extend(
                f'| {escape_markdown_cell(name)} | {escape_markdown_cell(estimate)} '
                f'| {escape_markdown_cell(status)} |'
                for name, estimate, status in rows
            )

        if self._show_correlations():
            corr = self.corr
            out.extend(['', f'Correlations (source: {escape_markdown_cell(self.cov_source)}):', ''])
            header = ' | '.join([''] + [escape_markdown_cell(n) for n in self.cov_labels])
            out.append(f'| {header} |')
            out.append('|' + '---|' * (len(self.cov_labels) + 1))
            threshold = DEFAULT_CORRELATION_THRESHOLD
            for i, label in enumerate(self.cov_labels):
                cells = []
                for j in range(len(self.cov_labels)):
                    text = _correlation_text(corr[i, j])
                    rho = float(corr[i, j])
                    if i != j and math.isfinite(rho) and abs(rho) >= threshold:
                        text = f'**{text}**'
                    cells.append(text)
                out.append(f'| {escape_markdown_cell(label)} | ' + ' | '.join(cells) + ' |')
            strong = self.strongly_correlated()
            if strong:
                out.append('')
                out.extend(
                    f'- Strongly correlated: `{escape_markdown_cell(a)}` / '
                    f'`{escape_markdown_cell(b)}` ({rho:+.2f})'
                    for a, b, rho in strong
                )

        if self.on_bounds:
            hits = ', '.join(f'`{escape_markdown_cell(n)}` ({side})' for n, side in self.on_bounds)
            out.extend(['', f'At a bound: {hits}'])

        if self.posterior is not None:
            out.extend(['', '```', self.posterior.format_summary(), '```'])

        return '\n'.join(out)

    def _repr_html_(self) -> str:
        """HTML report, so a notebook cell shows a table instead of text."""
        symbol = 'χ²'
        style = 'border-collapse:collapse;margin-bottom:0.75em'
        cell = 'padding:2px 10px;border:1px solid #ddd;text-align:left'
        head = 'padding:2px 10px;border:1px solid #ddd;text-align:left;background:#f5f5f5'

        parts = [f'<div><p><strong>Fit report:</strong> {escape_html_cell(self._header())}</p>']

        parts.append(f'<table style="{style}">')
        for label, value in self._quality_rows(symbol):
            parts.append(
                f'<tr><th style="{head}">{escape_html_cell(label)}</th>'
                f'<td style="{cell}">{escape_html_cell(value)}</td></tr>'
            )
        parts.append('</table>')

        rows = self._parameter_rows()
        if rows:
            parts.append(f'<table style="{style}">')
            parts.append(
                f'<tr><th style="{head}">Parameter</th><th style="{head}">Estimate</th>'
                f'<th style="{head}">Status</th></tr>'
            )
            for name, estimate, status in rows:
                parts.append(
                    f'<tr><td style="{cell}">{escape_html_cell(name)}</td>'
                    f'<td style="{cell}">{escape_html_cell(estimate)}</td>'
                    f'<td style="{cell}">{escape_html_cell(status)}</td></tr>'
                )
            parts.append('</table>')

        if self._show_correlations():
            corr = self.corr
            parts.append(
                f'<p>Correlations (source: {escape_html_cell(self.cov_source)}):</p>'
                f'<table style="{style}">'
            )
            heads = ''.join(
                f'<th style="{head}">{escape_html_cell(n)}</th>' for n in self.cov_labels
            )
            parts.append(f'<tr><th style="{head}"></th>{heads}</tr>')
            threshold = DEFAULT_CORRELATION_THRESHOLD
            for i, label in enumerate(self.cov_labels):
                cells = []
                for j in range(len(self.cov_labels)):
                    text = escape_html_cell(_correlation_text(corr[i, j]))
                    rho = float(corr[i, j])
                    if i != j and math.isfinite(rho) and abs(rho) >= threshold:
                        text = f'<strong>{text}</strong>'
                    cells.append(f'<td style="{cell}">{text}</td>')
                parts.append(
                    f'<tr><th style="{head}">{escape_html_cell(label)}</th>'
                    + ''.join(cells)
                    + '</tr>'
                )
            parts.append('</table>')
            strong = self.strongly_correlated()
            if strong:
                items = ''.join(
                    f'<li>{escape_html_cell(a)} / {escape_html_cell(b)} ({rho:+.2f})</li>'
                    for a, b, rho in strong
                )
                parts.append(f'<p>Strongly correlated:</p><ul>{items}</ul>')

        if self.on_bounds:
            hits = ', '.join(f'{escape_html_cell(n)} ({side})' for n, side in self.on_bounds)
            parts.append(f'<p><strong>At a bound:</strong> {hits}</p>')

        if self.posterior is not None:
            parts.append(f'<pre>{escape_html_cell(self.posterior.format_summary())}</pre>')

        parts.append('</div>')
        return ''.join(parts)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """The report as strictly JSON-serializable data.

        Every non-finite float becomes None and every array becomes nested lists,
        at any depth, so ``json.dumps(report.to_dict(), allow_nan=False)`` always
        succeeds. This is the payload a future ``save_analysis()`` will store.
        """
        corr = self.corr
        posterior: dict[str, Any] | None = None
        if self.posterior is not None:
            posterior = {
                'labels': list(self.posterior.labels),
                'n_samples': self.posterior.n_samples,
                'n_params': self.posterior.n_params,
                'best': dict(self.posterior.best),
                'mean': dict(self.posterior.mean),
                'median': dict(self.posterior.median),
                'std': dict(self.posterior.std),
                'ci_68': {name: list(bounds) for name, bounds in self.posterior.ci_68.items()},
                'ci_95': {name: list(bounds) for name, bounds in self.posterior.ci_95.items()},
                'diagnostics': self.posterior.diagnostics,
            }

        payload = {
            'model': self.model,
            'engine': self.engine,
            'method': self.method,
            'resolution': self.resolution,
            'weighting_note': self.weighting_note,
            'chisq': self.chisq,
            'reduced_chisq': self.reduced_chisq,
            'n_points': self.n_points,
            'n_free': self.n_free,
            'dof': self.dof,
            'converged': self.converged,
            'message': self.message,
            'parameters': self.parameters,
            'cov_labels': list(self.cov_labels),
            'cov': self.cov,
            'corr': corr,
            'cov_source': self.cov_source,
            'strong_correlations': [
                {'a': a, 'b': b, 'rho': rho} for a, b, rho in self.strongly_correlated()
            ],
            'on_bounds': [{'parameter': name, 'bound': side} for name, side in self.on_bounds],
            'posterior': posterior,
        }
        return json_safe(payload)

    def to_json(self, indent: int | None = 2) -> str:
        """The report as a JSON string, with non-finite values already handled."""
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)


def _correlation_text(rho: Any) -> str:
    """Render one correlation cell, never as the bare token 'nan'."""
    value = float(rho)
    return f'{value:+.2f}' if math.isfinite(value) else 'n/a'

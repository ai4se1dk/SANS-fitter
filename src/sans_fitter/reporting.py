"""One shareable document per analysis: settings, tables and the fit plot.

``FitReport`` already renders the goodness-of-fit and parameter tables. This
wraps them in the rest of what a collaborator needs to judge a fit: where the
data came from, how it was smeared and weighted, which Q range was used, and
the plot itself.

Two things here are deliberate rather than incidental:

**Markdown needs an image renderer, HTML does not.** Base64 embedding and a
sidecar PNG are two destinations for the same rendered image, so falling back
from one to the other does not remove the dependency. HTML embeds the live
Plotly figure and works on the standard dependency set; Markdown asks for the
optional ``[report]`` extra and says so plainly when it is missing.

**Everything interpolated is escaped.** Model expressions, monikers, file paths
and optimizer messages all reach the document, and the table helpers only
escape what goes through the tables.
"""

import html
import os
import warnings
from typing import Any

from .console import CHI_SQUARED
from .report import FitReport, escape_html_cell, escape_markdown_cell

#: Formats ``SANSFitter.report`` can write, by lowercased file extension.
FORMATS = {'.html': 'html', '.htm': 'html', '.md': 'markdown', '.markdown': 'markdown'}


class Report:
    """A rendered analysis report.

    Returned by ``SANSFitter.report()``. A plain string could not be this: it
    has no ``_repr_html_``, so it would not render in a notebook, and the
    format would have to be chosen before the caller knew what to do with it.
    """

    def __init__(self, html_text: str, markdown_text: str, assets: dict[str, bytes]):
        self._html = html_text
        self._markdown = markdown_text
        #: Files the Markdown form references, keyed by filename.
        self.assets = assets

    def __str__(self) -> str:
        return self._markdown

    def to_markdown(self) -> str:
        """The report as Markdown."""
        return self._markdown

    def to_html(self) -> str:
        """The report as a self-contained HTML document."""
        return self._html

    def _repr_html_(self) -> str:
        return self._html

    def write(self, filename: str, fmt: str | None = None) -> None:
        """Write the report to *filename*, plus any sidecar assets.

        Rendered before anything is replaced, through a temporary sibling and
        an atomic rename, so a failed write cannot destroy a report that is
        already there.
        """
        resolved = fmt or format_for(filename)
        target = os.path.abspath(filename)
        directory = os.path.dirname(target)
        if directory and not os.path.isdir(directory):
            raise ValueError(f'Directory does not exist: {directory}')

        text = self._html if resolved == 'html' else self._markdown
        written: list[str] = []
        try:
            for name, payload in self.assets.items() if resolved == 'markdown' else {}.items():
                asset_path = os.path.join(directory, name)
                with open(asset_path, 'wb') as handle:
                    handle.write(payload)
                written.append(asset_path)
            _atomic_write(target, text)
        except BaseException:
            for path in written:
                try:
                    os.remove(path)
                except OSError:
                    pass
            raise


def _atomic_write(target: str, text: str) -> None:
    temporary = f'{target}.tmp-{os.getpid()}'
    try:
        with open(temporary, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(text)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise


def format_for(filename: str) -> str:
    """Resolve an output format from a filename extension."""
    extension = os.path.splitext(filename)[1].lower()
    if extension not in FORMATS:
        supported = ', '.join(sorted(set(FORMATS)))
        raise ValueError(
            f"Cannot tell the report format from '{filename}'. Use one of: {supported}."
        )
    return FORMATS[extension]


def _settings_rows(fitter: Any) -> list[tuple[str, str]]:
    """The analysis settings a reader needs to interpret the numbers."""
    rows: list[tuple[str, str]] = [('Model', fitter.model_name or 'not set')]

    source = getattr(fitter, '_data_source', None)
    rows.append(('Data', source.describe() if source is not None else 'not loaded'))
    if source is not None:
        rows.append(('Data points', str(source.n_points)))

    if fitter.data is not None:
        qmin, qmax = fitter.get_q_range()
        full = fitter._full_q_range
        suffix = (
            '' if (qmin, qmax) == full else f' (restricted from {full[0]:.4g} to {full[1]:.4g})'
        )
        rows.append(('Q range', f'{qmin:.4g} to {qmax:.4g} 1/A{suffix}'))

    from .data.resolution import ResolutionSetting

    rows.append(
        ('Resolution', ResolutionSetting(**fitter.get_resolution()).describe(ascii_only=True))
    )

    structure_factor = fitter.get_structure_factor()
    if structure_factor:
        rows.append(
            (
                'Structure factor',
                f'{structure_factor} (radius_effective: {fitter._radius_effective_mode})',
            )
        )

    components = fitter.get_components()
    if components:
        rows.append(
            ('Components', ', '.join(f'{moniker} = {part}' for _p, moniker, part in components))
        )

    links = fitter.get_links()
    if links:
        rows.append(('Links', ', '.join(f'{a} = {b}' for a, b in links.items())))

    if fitter.supports_polydispersity():
        enabled = fitter.is_polydispersity_enabled()
        active = [
            f'{name} {fitter.get_pd_param(name)["pd"]:.4g} {fitter.get_pd_param(name)["pd_type"]}'
            for name in fitter.get_polydisperse_parameters()
            if fitter.get_pd_param(name)['pd'] > 0
        ]
        detail = ', '.join(active) if active else 'none configured'
        rows.append(('Polydispersity', f'{"on" if enabled else "off"}: {detail}'))

    return rows


def _configuration_rows(fitter: Any) -> list[tuple[str, str, str]]:
    """Parameter table for a report with no fit behind it."""
    rows = []
    links = fitter.get_links()
    for name, info in fitter.params.items():
        if name in links:
            status = f'linked to {links[name]}'
        elif info['vary']:
            status = 'free'
        else:
            status = 'fixed'
        rows.append((name, f'{info["value"]:.6g}', status))
    return rows


def render(
    fitter: Any,
    *,
    offline: bool = False,
    asset_stem: str = 'report',
    include_figure: bool = True,
) -> Report:
    """Build both renderings of *fitter*'s current state.

    Both formats are produced together so the caller can choose afterwards and
    so a notebook can show the HTML while ``str()`` still gives Markdown.
    """
    report = fitter.get_fit_report() if fitter._fit_contract is not None else None
    settings = _settings_rows(fitter)
    figure = _build_figure(fitter) if include_figure and fitter.data is not None else None

    assets: dict[str, bytes] = {}
    image = None
    if figure is not None:
        image = _render_image(figure)
        if image is not None:
            assets[f'{asset_stem}_fit.png'] = image

    return Report(
        html_text=_render_html(fitter, report, settings, figure, offline),
        markdown_text=_render_markdown(fitter, report, settings, image, asset_stem),
        assets=assets,
    )


def _build_figure(fitter: Any) -> Any:
    """The fit plot, or the theory preview when there is no fit.

    ``show=False`` matters: the plotting default opens a browser window outside
    a notebook, which a report render must never do.
    """
    if fitter._fit_contract is not None:
        return fitter.plot_results(show=False)
    return fitter.plot_model(show=False)


def _render_image(figure: Any) -> bytes | None:
    """Rasterize a figure, or None when no usable backend is installed.

    Both a missing package and an installed-but-unusable one land here: current
    kaleido needs a compatible Chrome available separately, so importing it
    successfully is not proof that it can render.
    """
    try:
        return figure.to_image(format='png', width=1000, height=700, scale=2)
    except Exception:  # noqa: BLE001 - any backend failure means "no image"
        return None


def _render_markdown(
    fitter: Any,
    report: FitReport | None,
    settings: list[tuple[str, str]],
    image: bytes | None,
    asset_stem: str,
) -> str:
    out: list[str] = [
        f'# SANS fit report: {escape_markdown_cell(fitter.model_name or "no model")}',
        '',
    ]
    out += ['## Settings', '', '| Setting | Value |', '| --- | --- |']
    out += [f'| {escape_markdown_cell(k)} | {escape_markdown_cell(v)} |' for k, v in settings]
    out.append('')

    if report is not None:
        out += [report.to_markdown(), '']
    else:
        out += [
            '## Parameters (no fit has been run)',
            '',
            '| Parameter | Value | Status |',
            '| --- | --- | --- |',
        ]
        out += [
            f'| {escape_markdown_cell(n)} | {v} | {s} |' for n, v, s in _configuration_rows(fitter)
        ]
        out.append('')

    if image is not None:
        out += ['## Fit', '', f'![Fit plot]({asset_stem}_fit.png)', '']
    else:
        out += [
            '## Fit',
            '',
            '_The figure was omitted: no static image backend is available. '
            'Install the optional dependency with `pip install "sans-fitter[report]"` '
            'to include it, or use an HTML report, which needs no renderer._',
            '',
        ]
    return '\n'.join(out)


def _render_html(
    fitter: Any,
    report: FitReport | None,
    settings: list[tuple[str, str]],
    figure: Any,
    offline: bool,
) -> str:
    title = escape_html_cell(fitter.model_name or 'no model')
    body: list[str] = [f'<h1>SANS fit report: {title}</h1>']

    body.append('<h2>Settings</h2><table><tbody>')
    body += [
        f'<tr><th>{escape_html_cell(k)}</th><td>{escape_html_cell(v)}</td></tr>'
        for k, v in settings
    ]
    body.append('</tbody></table>')

    if report is not None:
        body.append(report._repr_html_())
    else:
        body.append('<h2>Parameters (no fit has been run)</h2>')
        body.append(
            '<table><thead><tr><th>Parameter</th><th>Value</th><th>Status</th></tr></thead><tbody>'
        )
        body += [
            f'<tr><td>{escape_html_cell(n)}</td><td>{v}</td><td>{s}</td></tr>'
            for n, v, s in _configuration_rows(fitter)
        ]
        body.append('</tbody></table>')

    if figure is not None:
        body.append('<h2>Fit</h2>')
        # Plotly's JS is included once for the whole document, however many
        # figures it grows to hold.
        body.append(
            figure.to_html(
                full_html=False,
                include_plotlyjs=True if offline else 'cdn',
            )
        )

    return _HTML_DOCUMENT.format(title=title, chi=html.escape(CHI_SQUARED), body='\n'.join(body))


_HTML_DOCUMENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SANS fit report: {title}</title>
<style>
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
          margin: 2rem auto; max-width: 60rem; padding: 0 1rem; line-height: 1.5; }}
  table {{ border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ border: 1px solid #d0d7de; padding: 0.35rem 0.7rem; text-align: left; }}
  th {{ background: #f6f8fa; }}
  h1 {{ font-size: 1.6rem; }}
  h2 {{ font-size: 1.2rem; margin-top: 2rem; }}
  pre {{ background: #f6f8fa; padding: 0.8rem; overflow-x: auto; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def warn_if_no_image(report: Report, fmt: str) -> None:
    """Warn once when a Markdown report had to drop its figure."""
    if fmt == 'markdown' and not report.assets:
        warnings.warn(
            'No static image backend is available, so the Markdown report has '
            'no figure. Install it with: pip install "sans-fitter[report]". An '
            'HTML report embeds the plot without needing one.',
            stacklevel=3,
        )

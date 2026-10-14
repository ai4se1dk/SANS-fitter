"""Figures for a simultaneous fit, checked as data rather than as pictures.

Every assertion here reads the traces back out of the figure, because the thing
worth protecting is that the right numbers reached the right panel — a plot can
look perfectly reasonable while showing one dataset's residuals under another
dataset's curve.
"""

import copy

import numpy as np
import pytest

from sans_fitter import MultiFitter, examples
from sans_fitter.multi_plotting import DatasetPanel, plot_multi


@pytest.fixture(scope='module')
def curves():
    return {
        'a': examples.simulate(
            'sphere', radius=45, scale=0.02, background=0.01, noise=0.02, seed=21, npoints=40
        ),
        'b': examples.simulate(
            'sphere',
            radius=45,
            scale=0.02,
            background=0.03,
            noise=0.02,
            seed=22,
            qmin=0.01,
            qmax=0.25,
            npoints=25,
        ),
    }


@pytest.fixture(scope='module')
def fitted(curves):
    fit = MultiFitter()
    for name, data in curves.items():
        fit.add(name, copy.deepcopy(data), model='sphere')
        entry = fit[name]
        entry.set_param('radius', value=40, min=10, max=100, vary=True)
        entry.set_param('scale', value=0.02, min=0.001, max=0.1, vary=True)
        entry.set_param('background', value=0.02, min=0.0, max=0.1, vary=True)
    fit.share('radius')
    fit.fit(method='lm')
    return fit


def traces_named(figure, name):
    return [trace for trace in figure.data if trace.name == name]


class TestPreview:
    def test_a_preview_draws_one_model_curve_per_dataset(self, fitted):
        figure = fitted.plot_model(show=False)
        assert len(traces_named(figure, 'Model (current parameters)')) == 2
        assert len(traces_named(figure, 'Experimental Data')) == 2

    def test_a_preview_needs_no_fit(self, curves):
        fit = MultiFitter()
        for name, data in curves.items():
            fit.add(name, copy.deepcopy(data), model='sphere')
            fit[name].set_param('radius', value=40, min=10, max=100, vary=True)
        figure = fit.plot_model(show=False)
        assert 'Model preview' in figure.layout.title.text

    def test_a_preview_shows_the_constrained_values(self, curves):
        fit = MultiFitter()
        for name, data in curves.items():
            fit.add(name, copy.deepcopy(data), model='sphere')
            fit[name].set_param('scale', value=0.02, min=0.001, max=0.1, vary=True)
        fit.constrain('b.scale', '3 * a.scale')
        figure = fit.plot_model(show=False)
        models = traces_named(figure, 'Model (current parameters)')
        # Same shape, three times the amplitude above the shared background.
        assert max(models[1].y) > 2 * max(models[0].y)


class TestFittedFigure:
    def test_each_dataset_gets_its_own_pair_of_panels(self, fitted):
        figure = fitted.plot_results(show=False)
        assert len(traces_named(figure, 'Joint fit')) == 2
        residuals = [trace for trace in figure.data if trace.name.endswith('residuals')]
        assert {trace.name for trace in residuals} == {'a residuals', 'b residuals'}

    def test_residuals_sit_under_their_own_curve(self, fitted):
        figure = fitted.plot_results(show=False)
        by_name = {trace.name: trace for trace in figure.data}
        for name in ('a', 'b'):
            curve_x = by_name[f'{name} residuals'].x
            expected = fitted.result.datasets[name].n_points
            assert len(curve_x) == expected

    def test_residual_values_match_the_result(self, fitted):
        figure = fitted.plot_results(show=False)
        by_name = {trace.name: trace for trace in figure.data}
        np.testing.assert_allclose(by_name['a residuals'].y, fitted.result.datasets['a'].residuals)

    def test_objective_residuals_are_offered_and_labelled_differently(self, curves):
        fit = MultiFitter()
        for name, data in curves.items():
            fit.add(name, copy.deepcopy(data), model='sphere')
            fit[name].set_param('radius', value=40, min=10, max=100, vary=True)
            fit[name].set_param('scale', value=0.02, min=0.001, max=0.1, vary=True)
            fit[name].set_param('background', value=0.02, min=0.0, max=0.1, vary=True)
        fit.share('radius')
        fit.set_dataset_weight('b', 0.25)
        fit.fit(method='lm')

        plain = fit.plot_results(show=False)
        weighted = fit.plot_results(show=False, objective_residuals=True)
        plain_y = {t.name: t.y for t in plain.data}['b residuals']
        weighted_y = {t.name: t.y for t in weighted.data}['b residuals']
        np.testing.assert_allclose(weighted_y, 0.5 * np.asarray(plain_y))

        titles = [
            value.get('title', {}).get('text')
            for key, value in weighted.layout.to_plotly_json().items()
            if key.startswith('yaxis') and isinstance(value, dict)
        ]
        assert 'Residuals (√w·σ)' in titles
        assert 'Residuals (σ)' not in titles

    def test_the_title_carries_the_joint_goodness_of_fit(self, fitted):
        figure = fitted.plot_results(show=False)
        assert 'Simultaneous fit: 2 datasets' in figure.layout.title.text
        assert f'{fitted.result.reduced_chisq:.4f}' in figure.layout.title.text

    def test_datasets_keep_independent_axes(self, fitted):
        figure = fitted.plot_results(show=False)
        layout = figure.layout.to_plotly_json()
        # Four rows: curve, residuals, curve, residuals. Each residual row is
        # tied to the curve above it, and the two datasets are not tied together.
        assert layout['xaxis2'].get('matches') == 'x'
        assert layout['xaxis4'].get('matches') == 'x3'

    def test_show_false_returns_without_displaying(self, fitted, monkeypatch):
        shown = []
        monkeypatch.setattr(
            'plotly.graph_objects.Figure.show', lambda self, *a, **k: shown.append(1)
        )
        fitted.plot_results(show=False)
        assert shown == []

    def test_excluded_points_are_drawn_separately(self, curves):
        fit = MultiFitter()
        for name, data in curves.items():
            fit.add(name, copy.deepcopy(data), model='sphere')
            fit[name].set_param('radius', value=40, min=10, max=100, vary=True)
        fit['a'].set_q_range(qmin=0.02, qmax=0.2)
        figure = fit.plot_model(show=False)
        assert len(traces_named(figure, 'Excluded Data')) == 1


class TestPanelAssembly:
    def _panel(self, data, length=None):
        curve = np.ones(len(data.x) if length is None else length)
        return DatasetPanel.from_dataset(
            name='a',
            model='sphere',
            data=data,
            fitted_curve=curve,
            fit_index=None,
            residuals=None,
        )

    def test_an_empty_analysis_cannot_be_plotted(self):
        with pytest.raises(ValueError, match='no datasets'):
            plot_multi([], title='x', show=False)

    def test_a_mismatched_curve_length_is_refused(self, curves):
        panel = self._panel(curves['a'], length=3)
        with pytest.raises(ValueError, match='but 40 were fitted'):
            plot_multi([panel], title='x', show=False)

    def test_a_panel_without_uncertainties_still_plots(self, curves):
        data = copy.deepcopy(curves['a'])
        data.dy = None
        panel = self._panel(data)
        figure = plot_multi([panel], title='x', show=False)
        titles = [
            value.get('title', {}).get('text')
            for key, value in figure.layout.to_plotly_json().items()
            if key.startswith('yaxis') and isinstance(value, dict)
        ]
        assert 'Residuals (no dI)' in titles

    def test_overlay_puts_everything_on_one_axis(self, fitted):
        panels = [
            DatasetPanel(
                name=entry.name,
                model=entry.model,
                q=entry.observed_q,
                intensity=entry.observed_intensity,
                uncertainty=entry.observed_uncertainty,
                dq=entry.observed_dq,
                fitted_curve=entry.fitted_curve,
                fit_index=entry.fit_index,
                residuals=entry.residuals,
            )
            for entry in fitted.result.datasets.values()
        ]
        figure = plot_multi(panels, title='overlay', show=False, overlay=True)
        layout = figure.layout.to_plotly_json()
        assert 'xaxis2' not in layout
        assert len(figure.data) == 4

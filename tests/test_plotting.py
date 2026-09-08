import builtins
import os
import unittest
from unittest.mock import patch

import numpy as np
from sasdata.dataloader.data_info import Data1D

from sans_fitter import SANSFitter, examples
from sans_fitter.data.loader import normalize_sans_data
from sans_fitter.plotting import PREVIEW_MODEL_TRACE_NAME, PREVIEW_TITLE_PREFIX, plot_fit
from sans_fitter.results import PREVIEW_ENGINE, FitArtifacts, FitResultContract
from tests.helpers import create_decay_data_file, create_loading_test_data_file_with_resolution


def make_preview_contract(data, chisq=1.5):
    """A preview contract covering every point of *data*."""
    return FitResultContract(
        engine=PREVIEW_ENGINE,
        method=PREVIEW_ENGINE,
        chisq=chisq,
        parameters={},
        artifacts=FitArtifacts(fitted_curve=np.asarray(data.y, dtype=float) * 1.1),
    )
class TestNotebookDetectionWithoutIPython(unittest.TestCase):
    """IPython is an optional extra, so the import guard is load-bearing."""

    def test_running_in_notebook_is_false_when_ipython_is_absent(self):
        from sans_fitter import plotting

        real_import = builtins.__import__

        def without_ipython(name, *args, **kwargs):
            if name.startswith('IPython'):
                raise ImportError(f'No module named {name!r}')
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, '__import__', side_effect=without_ipython):
            self.assertFalse(plotting._running_in_notebook())

    def test_plot_still_returns_a_figure_when_ipython_is_absent(self):
        from sans_fitter import plotting

        real_import = builtins.__import__

        def without_ipython(name, *args, **kwargs):
            if name.startswith('IPython'):
                raise ImportError(f'No module named {name!r}')
            return real_import(name, *args, **kwargs)

        fitter = SANSFitter()
        data_file = create_decay_data_file(num_points=20)
        try:
            fitter.load_data(data_file)
            with (
                patch.object(builtins, '__import__', side_effect=without_ipython),
                patch('plotly.graph_objects.Figure.show'),
            ):
                self.assertIsNotNone(fitter.plot_results(show=False))
        finally:
            if os.path.exists(data_file):
                os.unlink(data_file)


class TestVisualization(unittest.TestCase):
    """Test visualization functionality."""

    def setUp(self):
        self.fitter = SANSFitter()
        self.data_file = create_decay_data_file(num_points=20)
        self.fitter.load_data(self.data_file)

    def tearDown(self):
        if os.path.exists(self.data_file):
            os.unlink(self.data_file)

    def test_plot_data_only(self):
        with patch('plotly.graph_objects.Figure.show'):
            self.fitter.plot_results()

    def test_plot_without_data_raises_error(self):
        fitter = SANSFitter()
        with self.assertRaises(ValueError):
            fitter.plot_results()

    @patch('plotly.graph_objects.Figure.show')
    def test_plot_with_fit_results(self, _mock_show):
        self.fitter.set_model('sphere')
        self.fitter.set_param('radius', value=20.0, min=10.0, max=30.0, vary=True)
        self.fitter.set_param('scale', value=0.1, min=0.01, max=1.0, vary=True)
        self.fitter.set_param('background', value=0.01, vary=True)
        self.fitter.set_param('sld', value=2.0, vary=False)
        self.fitter.set_param('sld_solvent', value=3.0, vary=False)

        self.fitter.fit(engine='bumps', method='amoeba')

        self.fitter.plot_results(show_residuals=True, log_scale=True)
        self.fitter.plot_results(show_residuals=False, log_scale=False)

    @patch('plotly.graph_objects.Figure.show')
    def test_plot_data_with_resolution_shows_error_x(self, _mock_show):
        data_file = create_loading_test_data_file_with_resolution()
        try:
            fitter = SANSFitter()
            fitter.load_data(data_file)
            fig = fitter.plot_results()
            data_trace = fig.data[0]
            self.assertIsNotNone(data_trace.error_x)
            self.assertTrue(data_trace.error_x.visible)
        finally:
            os.unlink(data_file)

    @patch('plotly.graph_objects.Figure.show')
    def test_plot_data_without_resolution_no_error_x(self, _mock_show):
        fig = self.fitter.plot_results()
        data_trace = fig.data[0]
        self.assertIsNone(data_trace.error_x.array)

    @patch('plotly.graph_objects.Figure.show')
    def test_plot_data_shows_error_y(self, _mock_show):
        fig = self.fitter.plot_results()
        data_trace = fig.data[0]
        self.assertTrue(data_trace.error_y.visible)
        np.testing.assert_array_equal(data_trace.error_y.array, self.fitter.data.dy)

    def test_plot_shows_figure_exactly_once_by_default(self):
        with patch('plotly.graph_objects.Figure.show') as mock_show:
            self.fitter.plot_results()
        self.assertEqual(mock_show.call_count, 1)

    def test_plot_show_false_does_not_display(self):
        with patch('plotly.graph_objects.Figure.show') as mock_show:
            fig = self.fitter.plot_results(show=False)
        mock_show.assert_not_called()
        self.assertIsNotNone(fig)

    def test_plot_default_does_not_display_in_notebook(self):
        with (
            patch('plotly.graph_objects.Figure.show') as mock_show,
            patch('sans_fitter.plotting._running_in_notebook', return_value=True),
        ):
            fig = self.fitter.plot_results()
        mock_show.assert_not_called()
        self.assertIsNotNone(fig)


class TestPreviewContractPlotting(unittest.TestCase):
    """plot_fit rendering a preview contract, including data without dI."""

    def test_preview_title_and_trace_name(self):
        data = examples.simulate('sphere', npoints=15, seed=0)
        fig = plot_fit(
            data=data,
            fit_result=make_preview_contract(data),
            model_name='sphere',
            show=False,
        )
        self.assertIn(PREVIEW_TITLE_PREFIX, fig.layout.title.text)
        self.assertIn('χ² = 1.5000', fig.layout.title.text)
        self.assertIn(PREVIEW_MODEL_TRACE_NAME, [trace.name for trace in fig.data])

    def test_fit_contract_keeps_fit_labels(self):
        data = examples.simulate('sphere', npoints=15, seed=0)
        contract = make_preview_contract(data)
        contract.engine = 'bumps'
        fig = plot_fit(data=data, fit_result=contract, model_name='sphere', show=False)
        self.assertIn('SANS Fit', fig.layout.title.text)
        self.assertIn('Fitted Model', [trace.name for trace in fig.data])

    def test_data_without_dy_column_renders_without_residuals(self):
        q = np.geomspace(0.01, 0.3, 15)
        data = normalize_sans_data(Data1D(x=q, y=np.exp(-q * 10)))
        self.assertIsNone(data.dy)

        fig = plot_fit(
            data=data,
            fit_result=make_preview_contract(data, chisq=float('nan')),
            model_name='sphere',
            show=False,
        )
        self.assertIn('χ² n/a (no dI)', fig.layout.title.text)
        self.assertNotIn('Residuals', [trace.name for trace in fig.data])
        self.assertEqual(fig.layout.yaxis2.title.text, 'Residuals (no dI)')

    def test_zero_uncertainties_render_without_error(self):
        data = examples.simulate('sphere', npoints=15, noise=0, seed=0)
        np.testing.assert_array_equal(data.dy, np.zeros_like(data.dy))

        fig = plot_fit(
            data=data,
            fit_result=make_preview_contract(data, chisq=float('nan')),
            model_name='sphere',
            show=False,
        )
        self.assertIn('χ² n/a (no dI)', fig.layout.title.text)
        self.assertEqual(fig.layout.yaxis2.title.text, 'Residuals (no dI)')


if __name__ == '__main__':
    unittest.main(verbosity=2)

import os
import unittest
from unittest.mock import patch

import numpy as np

from sans_fitter import SANSFitter
from tests.helpers import create_decay_data_file, create_loading_test_data_file_with_resolution


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
        self.assertIsNotNone(data_trace.error_x)
        np.testing.assert_array_equal(data_trace.error_x.array, self.fitter.data.dx)
        self.assertTrue(np.all(data_trace.error_x.array == 0))


if __name__ == '__main__':
    unittest.main(verbosity=2)

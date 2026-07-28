import os
import unittest

import numpy as np

from sans_fitter import SANSFitter
from sans_fitter.data_loader import get_fit_index
from tests.helpers import create_decay_data_file


class QRangeTestCase(unittest.TestCase):
    """Base fixture: loaded decay data with a configured sphere model."""

    def setUp(self):
        self.fitter = SANSFitter()
        self.data_file = create_decay_data_file()
        self.fitter.load_data(self.data_file)
        self.fitter.set_model('sphere')
        self.fitter.set_param('radius', value=20.0, min=10.0, max=30.0, vary=True)
        self.fitter.set_param('scale', value=0.1, min=0.01, max=1.0, vary=True)
        self.fitter.set_param('background', value=0.01, min=0, max=0.1, vary=True)
        self.fitter.set_param('sld', value=2.0, vary=False)
        self.fitter.set_param('sld_solvent', value=3.0, vary=False)

    def tearDown(self):
        if os.path.exists(self.data_file):
            os.unlink(self.data_file)

    def points_in_range(self, qmin, qmax):
        q = np.asarray(self.fitter.data.x)
        return int(((q >= qmin) & (q <= qmax)).sum())


class TestQRangeConfiguration(QRangeTestCase):
    """Test set_q_range / get_q_range / reset_q_range state handling."""

    def test_set_q_range_without_data_raises_error(self):
        fitter = SANSFitter()
        with self.assertRaises(ValueError):
            fitter.set_q_range(0.01, 0.1)

    def test_set_q_range_without_arguments_raises_error(self):
        with self.assertRaises(ValueError):
            self.fitter.set_q_range()

    def test_set_q_range_inverted_bounds_raises_error(self):
        with self.assertRaises(ValueError):
            self.fitter.set_q_range(qmin=0.5, qmax=0.1)

    def test_set_q_range_updates_data_limits(self):
        self.fitter.set_q_range(qmin=0.02, qmax=0.5)
        self.assertEqual(self.fitter.get_q_range(), (0.02, 0.5))

    def test_set_q_range_partial_qmin_only(self):
        full_min, full_max = self.fitter.get_q_range()
        self.fitter.set_q_range(qmin=0.05)
        self.assertEqual(self.fitter.get_q_range(), (0.05, full_max))

    def test_set_q_range_partial_qmax_only(self):
        full_min, full_max = self.fitter.get_q_range()
        self.fitter.set_q_range(qmax=0.5)
        self.assertEqual(self.fitter.get_q_range(), (full_min, 0.5))

    def test_set_q_range_empty_range_raises_and_keeps_previous(self):
        previous = self.fitter.get_q_range()
        with self.assertRaises(ValueError):
            self.fitter.set_q_range(qmin=2.0, qmax=3.0)
        self.assertEqual(self.fitter.get_q_range(), previous)

    def test_reset_q_range_restores_full_range(self):
        full_range = self.fitter.get_q_range()
        self.fitter.set_q_range(qmin=0.05, qmax=0.5)
        self.fitter.reset_q_range()
        self.assertEqual(self.fitter.get_q_range(), full_range)

    def test_get_q_range_without_data_returns_none(self):
        self.assertIsNone(SANSFitter().get_q_range())

    def test_get_fit_index_respects_q_range(self):
        self.fitter.set_q_range(qmin=0.02, qmax=0.5)
        index = get_fit_index(self.fitter.data)
        self.assertEqual(int(index.sum()), self.points_in_range(0.02, 0.5))
        self.assertLess(int(index.sum()), len(self.fitter.data.x))


class TestQRangeFitting(QRangeTestCase):
    """Test that both engines fit only the restricted range."""

    def test_bumps_fit_uses_restricted_range(self):
        self.fitter.set_q_range(qmin=0.02, qmax=0.5)
        expected_points = self.points_in_range(0.02, 0.5)

        result = self.fitter.fit(engine='bumps', method='amoeba')

        self.assertIn('chisq', result)
        contract = self.fitter._fit_contract
        self.assertEqual(len(contract.artifacts.fitted_curve), expected_points)
        self.assertEqual(int(contract.artifacts.fit_index.sum()), expected_points)

    def test_scipy_fit_uses_restricted_range(self):
        self.fitter.set_q_range(qmin=0.02, qmax=0.5)
        expected_points = self.points_in_range(0.02, 0.5)

        result = self.fitter.fit(engine='lmfit', method='leastsq')

        self.assertIn('chisq', result)
        contract = self.fitter._fit_contract
        self.assertEqual(len(contract.artifacts.fitted_curve), expected_points)
        self.assertEqual(int(contract.artifacts.fit_index.sum()), expected_points)

    def test_full_range_fit_still_covers_all_points(self):
        self.fitter.fit(engine='bumps', method='amoeba')
        contract = self.fitter._fit_contract
        self.assertEqual(len(contract.artifacts.fitted_curve), len(self.fitter.data.x))


class TestQRangePostFit(QRangeTestCase):
    """Test plotting and export with a restricted Q range."""

    def setUp(self):
        super().setUp()
        self.fitter.set_q_range(qmin=0.02, qmax=0.5)
        self.expected_points = self.points_in_range(0.02, 0.5)
        self.fitter.fit(engine='bumps', method='amoeba')

    def test_plot_results_with_restricted_range(self):
        fig = self.fitter.plot_results(show_residuals=True, show=False)

        traces = {trace.name: trace for trace in fig.data}
        self.assertIn('Experimental Data', traces)
        self.assertIn('Excluded Data', traces)
        self.assertIn('Fitted Model', traces)
        self.assertEqual(len(traces['Experimental Data'].x), self.expected_points)
        self.assertEqual(len(traces['Fitted Model'].x), self.expected_points)
        n_excluded = len(self.fitter.data.x) - self.expected_points
        self.assertEqual(len(traces['Excluded Data'].x), n_excluded)

    def test_plot_results_without_excluded_points_has_no_excluded_trace(self):
        self.fitter.reset_q_range()
        self.fitter.fit(engine='bumps', method='amoeba')
        fig = self.fitter.plot_results(show=False)
        trace_names = [trace.name for trace in fig.data]
        self.assertNotIn('Excluded Data', trace_names)

    def test_save_results_with_restricted_range(self):
        output_file = self.data_file + '.results.csv'
        try:
            self.fitter.save_results(output_file)

            with open(output_file) as f:
                lines = f.readlines()

            data_rows = [
                line for line in lines if not line.startswith('#') and not line.startswith('Q,')
            ]
            self.assertEqual(len(data_rows), self.expected_points)
            header = ''.join(lines)
            self.assertIn('# Q range: 0.02 to 0.5', header)
            self.assertIn(
                f'# Points fitted: {self.expected_points} of {len(self.fitter.data.x)}',
                header,
            )
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)


if __name__ == '__main__':
    unittest.main()

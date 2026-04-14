import os
import tempfile
import unittest

import numpy as np

from sans_fitter import SANSFitter
from tests.helpers import create_decay_data_file


class TestResultExport(unittest.TestCase):
    """Test result export functionality."""

    def setUp(self):
        self.fitter = SANSFitter()
        self.data_file = create_decay_data_file(num_points=20)
        self.fitter.load_data(self.data_file)
        self.fitter.set_model('sphere')
        self.fitter.set_param('radius', value=20.0, min=10.0, max=30.0, vary=True)
        self.fitter.set_param('scale', value=0.1, min=0.01, max=1.0, vary=True)
        self.fitter.set_param('background', value=0.01, vary=True)
        self.fitter.set_param('sld', value=2.0, vary=False)
        self.fitter.set_param('sld_solvent', value=3.0, vary=False)
        self.fitter.fit(engine='bumps', method='amoeba')

    def tearDown(self):
        if os.path.exists(self.data_file):
            os.unlink(self.data_file)

    def test_save_results_creates_file(self):
        output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        output_file.close()
        output_path = output_file.name

        try:
            self.fitter.save_results(output_path)
            self.assertTrue(os.path.exists(output_path))

            with open(output_path) as handle:
                content = handle.read()
                self.assertIn('SANS Fit Results', content)
                self.assertIn('Model:', content)
                self.assertIn('Chi-squared:', content)
                self.assertIn('Q,I_exp,dI_exp,I_fit,Residuals', content)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_save_results_without_fit_raises_error(self):
        fitter = SANSFitter()
        with self.assertRaises(ValueError):
            fitter.save_results('output.csv')

    def test_save_results_raises_on_mismatched_array_lengths(self):
        output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        output_file.close()
        output_path = output_file.name

        original_curve = self.fitter._fit_contract.artifacts.fitted_curve
        self.fitter._fit_contract.artifacts.fitted_curve = np.asarray(original_curve[:-1])

        try:
            with self.assertRaisesRegex(ValueError, 'mismatched array lengths'):
                self.fitter.save_results(output_path)
        finally:
            self.fitter._fit_contract.artifacts.fitted_curve = original_curve
            if os.path.exists(output_path):
                os.unlink(output_path)


if __name__ == '__main__':
    unittest.main(verbosity=2)

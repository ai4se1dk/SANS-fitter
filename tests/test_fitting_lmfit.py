import os
import unittest

from sans_fitter import SANSFitter
from tests.helpers import create_decay_data_file


class TestLMFitFitting(unittest.TestCase):
    """Test LMFit fitting functionality."""

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

    def test_lmfit_fit_runs(self):
        try:
            result = self.fitter.fit(engine='lmfit', method='leastsq')

            self.assertIsNotNone(result)
            self.assertIn('engine', result)
            self.assertEqual(result['engine'], 'lmfit')
            self.assertIn('chisq', result)
            self.assertIn('parameters', result)
        except ValueError as error:
            if 'lmfit is not installed' in str(error):
                self.skipTest('lmfit not installed')
            raise

    def test_lmfit_fit_returns_parameters(self):
        try:
            result = self.fitter.fit(engine='lmfit', method='leastsq')

            self.assertIn('radius', result['parameters'])
            self.assertIn('scale', result['parameters'])

            for _param_name, param_result in result['parameters'].items():
                self.assertIn('value', param_result)
                self.assertIn('stderr', param_result)
                self.assertIn('formatted', param_result)
        except ValueError as error:
            if 'lmfit is not installed' in str(error):
                self.skipTest('lmfit not installed')
            raise


class TestLMFitFittingWithPolydispersity(unittest.TestCase):
    """Test scipy/lmfit fitting engine with polydispersity parameters."""

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

    def test_fit_with_pd_enabled_fixed_width_leastsq(self):
        self.fitter.set_pd_param('radius', pd_width=0.1, pd_type='gaussian', vary=False)
        self.fitter.enable_polydispersity(True)

        result = self.fitter.fit(engine='lmfit', method='leastsq')

        self.assertIsNotNone(result)
        self.assertIn('chisq', result)
        self.assertIn('parameters', result)

    def test_fit_with_pd_enabled_varying_width_least_squares(self):
        self.fitter.set_pd_param('radius', pd_width=0.05, pd_type='gaussian', vary=True)
        self.fitter.enable_polydispersity(True)

        result = self.fitter.fit(engine='lmfit', method='least_squares')

        self.assertIsNotNone(result)
        self.assertIn('chisq', result)
        self.assertIn('radius_pd', result['parameters'])

    def test_pd_width_updated_after_lmfit(self):
        self.fitter.set_pd_param('radius', pd_width=0.05, pd_type='gaussian', vary=True)
        self.fitter.enable_polydispersity(True)

        self.fitter.fit(engine='lmfit', method='least_squares')

        pd_config = self.fitter.get_pd_param('radius')
        self.assertIsNotNone(pd_config['pd'])


if __name__ == '__main__':
    unittest.main(verbosity=2)

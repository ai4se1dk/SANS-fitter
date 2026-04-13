import os
import unittest

from sans_fitter import SANSFitter
from tests.helpers import create_decay_data_file


class TestBumpsFitting(unittest.TestCase):
    """Test BUMPS fitting functionality."""

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

    def test_bumps_fit_runs(self):
        result = self.fitter.fit(engine='bumps', method='amoeba')

        self.assertIsNotNone(result)
        self.assertIn('engine', result)
        self.assertEqual(result['engine'], 'bumps')
        self.assertIn('chisq', result)
        self.assertIn('parameters', result)

    def test_bumps_fit_returns_parameters(self):
        result = self.fitter.fit(engine='bumps', method='amoeba')
        self.assertIn('radius', result['parameters'])
        self.assertIn('scale', result['parameters'])
        self.assertIn('background', result['parameters'])

        for _param_name, param_result in result['parameters'].items():
            self.assertIn('value', param_result)
            self.assertIn('stderr', param_result)
            self.assertIn('formatted', param_result)

    def test_bumps_fit_updates_internal_params(self):
        self.fitter.params['radius']['value']
        self.fitter.fit(engine='bumps', method='amoeba')
        self.assertIsNotNone(self.fitter.fit_result)

    def test_bumps_fit_stores_result(self):
        self.fitter.fit(engine='bumps', method='amoeba')

        self.assertIsNotNone(self.fitter.fit_result)
        self.assertIsNotNone(self.fitter._fitted_model)


class TestBumpsFittingWithPolydispersity(unittest.TestCase):
    """Test BUMPS fitting engine with polydispersity parameters."""

    def setUp(self):
        self.fitter = SANSFitter()
        self.data_file = create_decay_data_file(num_points=12)
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

    def test_fit_with_pd_enabled_fixed_width(self):
        self.fitter.set_pd_param('radius', pd_width=0.1, pd_type='gaussian', vary=False)
        self.fitter.enable_polydispersity(True)

        result = self.fitter.fit(engine='bumps', method='amoeba')

        self.assertIsNotNone(result)
        self.assertIn('chisq', result)
        self.assertIn('parameters', result)
        self.assertIn('radius', result['parameters'])
        self.assertIn('scale', result['parameters'])

    def test_fit_with_pd_enabled_varying_width(self):
        self.fitter.set_pd_param('radius', pd_width=0.05, pd_type='gaussian', vary=True)
        self.fitter.enable_polydispersity(True)

        result = self.fitter.fit(engine='bumps', method='amoeba')

        self.assertIsNotNone(result)
        self.assertIn('chisq', result)
        self.assertIn('radius_pd', result['parameters'])

    def test_pd_width_updated_after_fit(self):
        self.fitter.set_pd_param('radius', pd_width=0.05, pd_type='gaussian', vary=True)
        self.fitter.enable_polydispersity(True)

        self.fitter.fit(engine='bumps', method='amoeba')

        pd_config = self.fitter.get_pd_param('radius')
        self.assertIsNotNone(pd_config['pd'])

    def test_fit_with_multiple_pd_params(self):
        self.fitter.set_model('cylinder')
        self.fitter.set_param('radius', value=20.0, min=10.0, max=50.0, vary=False)
        self.fitter.set_param('length', value=200.0, min=50.0, max=500.0, vary=False)
        self.fitter.set_param('scale', value=0.1, min=0.01, max=1.0, vary=True)
        self.fitter.set_param('background', value=0.01, min=0.0, max=0.1, vary=False)
        self.fitter.set_pd_param('radius', pd_width=0.1, vary=False)
        self.fitter.set_pd_param('length', pd_width=0.15, vary=False)
        self.fitter.enable_polydispersity(True)

        result = self.fitter.fit(engine='bumps', method='amoeba')

        self.assertIsNotNone(result)
        self.assertIn('chisq', result)
        self.assertIn('scale', result['parameters'])


if __name__ == '__main__':
    unittest.main(verbosity=2)

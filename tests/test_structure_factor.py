import os
import unittest

from sans_fitter import SANSFitter
from tests.helpers import create_concentrated_sphere_data_file


class TestStructureFactorSetup(unittest.TestCase):
    """Test structure factor setup and configuration."""

    def setUp(self):
        self.fitter = SANSFitter()
        self.fitter.set_model('sphere')

    def test_set_structure_factor_hardsphere(self):
        self.fitter.set_structure_factor('hardsphere')

        self.assertEqual(self.fitter.get_structure_factor(), 'hardsphere')
        self.assertIn('volfraction', self.fitter.params)
        self.assertIn('radius_effective', self.fitter.params)

    def test_set_structure_factor_hayter_msa(self):
        self.fitter.set_structure_factor('hayter_msa')

        self.assertEqual(self.fitter.get_structure_factor(), 'hayter_msa')
        self.assertIn('volfraction', self.fitter.params)
        self.assertIn('radius_effective', self.fitter.params)
        self.assertIn('charge', self.fitter.params)

    def test_set_structure_factor_squarewell(self):
        self.fitter.set_structure_factor('squarewell')

        self.assertEqual(self.fitter.get_structure_factor(), 'squarewell')
        self.assertIn('volfraction', self.fitter.params)
        self.assertIn('radius_effective', self.fitter.params)

    def test_set_structure_factor_stickyhardsphere(self):
        self.fitter.set_structure_factor('stickyhardsphere')

        self.assertEqual(self.fitter.get_structure_factor(), 'stickyhardsphere')
        self.assertIn('volfraction', self.fitter.params)
        self.assertIn('radius_effective', self.fitter.params)

    def test_set_structure_factor_without_model_raises_error(self):
        fitter = SANSFitter()
        with self.assertRaises(ValueError) as context:
            fitter.set_structure_factor('hardsphere')
        self.assertIn('No form factor model loaded', str(context.exception))

    def test_set_structure_factor_invalid_name_raises_error(self):
        with self.assertRaises(ValueError) as context:
            self.fitter.set_structure_factor('invalid_structure_factor')
        self.assertIn('Unsupported structure factor', str(context.exception))

    def test_set_structure_factor_invalid_mode_raises_error(self):
        with self.assertRaises(ValueError) as context:
            self.fitter.set_structure_factor('hardsphere', radius_effective_mode='invalid_mode')
        self.assertIn('Invalid radius_effective_mode', str(context.exception))

    def test_structure_factor_preserves_form_factor_params(self):
        self.fitter.set_param('radius', value=50.0, min=10.0, max=100.0)
        self.fitter.set_structure_factor('hardsphere')

        self.assertEqual(self.fitter.params['radius']['value'], 50.0)
        self.assertEqual(self.fitter.params['radius']['min'], 10.0)
        self.assertEqual(self.fitter.params['radius']['max'], 100.0)


class TestStructureFactorLinkRadius(unittest.TestCase):
    """Test radius_effective linking functionality."""

    def setUp(self):
        self.fitter = SANSFitter()
        self.fitter.set_model('sphere')
        self.fitter.set_param('radius', value=50.0)

    def test_link_radius_mode_sets_radius_effective(self):
        self.fitter.set_structure_factor('hardsphere', radius_effective_mode='link_radius')

        self.assertEqual(self.fitter.params['radius_effective']['value'], 50.0)
        self.assertFalse(self.fitter.params['radius_effective']['vary'])

    def test_link_radius_mode_syncs_on_set_param(self):
        self.fitter.set_structure_factor('hardsphere', radius_effective_mode='link_radius')
        self.fitter.set_param('radius', value=75.0)
        self.assertEqual(self.fitter.params['radius_effective']['value'], 75.0)

    def test_unconstrained_mode_allows_independent_radius_effective(self):
        self.fitter.set_structure_factor('hardsphere', radius_effective_mode='unconstrained')
        self.fitter.set_param('radius_effective', value=60.0, vary=True)

        self.assertEqual(self.fitter.params['radius_effective']['value'], 60.0)
        self.assertTrue(self.fitter.params['radius_effective']['vary'])


class TestStructureFactorRemoval(unittest.TestCase):
    """Test structure factor removal functionality."""

    def setUp(self):
        self.fitter = SANSFitter()
        self.fitter.set_model('sphere')
        self.fitter.set_param('radius', value=50.0, min=10.0, max=100.0)

    def test_remove_structure_factor(self):
        self.fitter.set_structure_factor('hardsphere')
        self.assertIsNotNone(self.fitter.get_structure_factor())

        self.fitter.remove_structure_factor()

        self.assertIsNone(self.fitter.get_structure_factor())
        self.assertNotIn('volfraction', self.fitter.params)
        self.assertNotIn('radius_effective', self.fitter.params)

    def test_remove_structure_factor_restores_params(self):
        original_radius = self.fitter.params['radius']['value']

        self.fitter.set_structure_factor('hardsphere')
        self.fitter.remove_structure_factor()

        self.assertEqual(self.fitter.params['radius']['value'], original_radius)

    def test_remove_structure_factor_without_one_raises_error(self):
        with self.assertRaises(ValueError) as context:
            self.fitter.remove_structure_factor()
        self.assertIn('No structure factor is currently set', str(context.exception))


class TestStructureFactorModelSwitching(unittest.TestCase):
    """Test structure factor behavior when switching models."""

    def setUp(self):
        self.fitter = SANSFitter()

    def test_set_model_resets_structure_factor(self):
        self.fitter.set_model('sphere')
        self.fitter.set_structure_factor('hardsphere')
        self.fitter.set_model('cylinder')

        self.assertIsNone(self.fitter.get_structure_factor())
        self.assertNotIn('volfraction', self.fitter.params)


class TestStructureFactorFitting(unittest.TestCase):
    """Test fitting with structure factors."""

    def setUp(self):
        self.fitter = SANSFitter()
        self.data_file = create_concentrated_sphere_data_file()
        self.fitter.load_data(self.data_file)
        self.fitter.set_model('sphere')
        self.fitter.set_param('radius', value=50.0, min=10.0, max=100.0, vary=True)
        self.fitter.set_param('scale', value=0.01, min=0.001, max=1.0, vary=True)
        self.fitter.set_param('background', value=0.001, min=0, max=0.1, vary=True)
        self.fitter.set_param('sld', value=1.0, vary=False)
        self.fitter.set_param('sld_solvent', value=6.0, vary=False)

    def tearDown(self):
        if os.path.exists(self.data_file):
            os.unlink(self.data_file)

    def test_fit_with_hardsphere_bumps(self):
        self.fitter.set_structure_factor('hardsphere')
        self.fitter.set_param('volfraction', value=0.2, min=0.0, max=0.6, vary=True)
        self.fitter.set_param('radius_effective', value=50.0, min=10.0, max=100.0, vary=True)

        result = self.fitter.fit(engine='bumps', method='amoeba')

        self.assertIsNotNone(result)
        self.assertIn('volfraction', result['parameters'])
        self.assertIn('radius_effective', result['parameters'])

    def test_fit_with_hardsphere_link_radius_bumps(self):
        self.fitter.set_structure_factor('hardsphere', radius_effective_mode='link_radius')
        self.fitter.set_param('volfraction', value=0.2, min=0.0, max=0.6, vary=True)

        result = self.fitter.fit(engine='bumps', method='amoeba')

        self.assertIsNotNone(result)
        self.assertIn('volfraction', result['parameters'])

    def test_fit_with_hardsphere_lmfit(self):
        self.fitter.set_structure_factor('hardsphere')
        self.fitter.set_param('volfraction', value=0.2, min=0.01, max=0.6, vary=True)
        self.fitter.set_param('radius_effective', value=50.0, min=10.0, max=100.0, vary=True)

        try:
            result = self.fitter.fit(engine='lmfit', method='least_squares')

            self.assertIsNotNone(result)
            self.assertIn('volfraction', result['parameters'])
            self.assertIn('radius_effective', result['parameters'])
        except ValueError as error:
            if 'scipy is not installed' in str(error):
                self.skipTest('scipy not installed')
            raise


if __name__ == '__main__':
    unittest.main(verbosity=2)

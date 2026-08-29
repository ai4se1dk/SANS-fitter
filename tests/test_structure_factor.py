import os
import unittest
import warnings

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

    def test_link_radius_mode_registers_an_ordinary_link(self):
        # link_radius is not a special case: it is reported by get_links() and
        # reaches the engines through the snapshot like any other link.
        self.fitter.set_structure_factor('hardsphere', radius_effective_mode='link_radius')

        self.assertEqual(self.fitter.get_links(), {'radius_effective': 'radius'})
        snapshot = self.fitter._param_manager.snapshot_fit_state()
        self.assertEqual(snapshot.linked_params, {'radius_effective': 'radius'})
        self.assertNotIn('radius_effective', snapshot.varying_params)

    def test_unconstrained_mode_registers_no_link(self):
        self.fitter.set_structure_factor('hardsphere', radius_effective_mode='unconstrained')
        self.assertEqual(self.fitter.get_links(), {})

    def test_link_radius_mode_rejects_direct_writes(self):
        self.fitter.set_structure_factor('hardsphere', radius_effective_mode='link_radius')

        with self.assertRaises(ValueError):
            self.fitter.set_param('radius_effective', value=60.0)
        with self.assertRaises(ValueError):
            self.fitter.set_param('radius_effective', vary=True)

    def test_link_radius_mode_rejects_a_duplicate_manual_link(self):
        self.fitter.set_structure_factor('hardsphere', radius_effective_mode='link_radius')

        with self.assertRaises(ValueError) as context:
            self.fitter.link_params('radius_effective', to='radius')
        self.assertIn('already linked', str(context.exception))

    def test_switching_mode_drops_the_previous_link(self):
        self.fitter.set_structure_factor('hardsphere', radius_effective_mode='link_radius')
        self.fitter.set_structure_factor('squarewell', radius_effective_mode='unconstrained')

        self.assertEqual(self.fitter.get_links(), {})
        self.fitter.set_param('radius_effective', value=60.0, vary=True)
        self.assertEqual(self.fitter.params['radius_effective']['value'], 60.0)

    def test_removing_structure_factor_retires_the_link_silently(self):
        self.fitter.set_structure_factor('hardsphere', radius_effective_mode='link_radius')

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.fitter.remove_structure_factor()

        self.assertEqual(self.fitter.get_links(), {})
        # The mode-owned link retires with its structure factor; only a link the
        # user made themselves is reported as stale.
        self.assertFalse([w for w in caught if 'link' in str(w.message).lower()])


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
        # radius varies, so this only holds if the link was applied during the
        # fit rather than merely at setup time.
        self.assertEqual(
            self.fitter.params['radius_effective']['value'],
            self.fitter.params['radius']['value'],
        )

    def test_fit_with_hardsphere_link_radius_lmfit(self):
        self.fitter.set_structure_factor('hardsphere', radius_effective_mode='link_radius')
        self.fitter.set_param('volfraction', value=0.2, min=0.01, max=0.6, vary=True)

        try:
            result = self.fitter.fit(engine='lmfit', method='least_squares')
        except ValueError as error:
            if 'scipy is not installed' in str(error):
                self.skipTest('scipy not installed')
            raise

        self.assertIsNotNone(result)
        self.assertEqual(
            self.fitter.params['radius_effective']['value'],
            self.fitter.params['radius']['value'],
        )
        self.assertEqual(
            result['parameters']['radius_effective']['value'],
            result['parameters']['radius']['value'],
        )

    def test_fit_bayesian_with_link_radius(self):
        self.fitter.set_structure_factor('hardsphere', radius_effective_mode='link_radius')
        self.fitter.set_param('volfraction', value=0.2, min=0.0, max=0.6, vary=True)

        result = self.fitter.fit_bayesian(samples=200, burn=10)

        self.assertIsNotNone(result)
        # The follower is constrained away, so it never enters the chain.
        self.assertNotIn('radius_effective', self.fitter.get_posterior().labels)
        self.assertEqual(
            self.fitter.params['radius_effective']['value'],
            self.fitter.params['radius']['value'],
        )

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

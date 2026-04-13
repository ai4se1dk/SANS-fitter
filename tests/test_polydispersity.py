"""Unit tests for polydispersity support."""

import os
import unittest
from unittest.mock import Mock

from sans_fitter import (
    PD_DEFAULTS,
    PD_DISTRIBUTION_TYPES,
    ParameterManager,
    SANSFitter,
)
from tests.helpers import create_decay_data_file


class TestPolydispersityConstants(unittest.TestCase):
    """Test polydispersity constants."""

    def test_pd_defaults_structure(self):
        """Test that PD_DEFAULTS has expected keys."""
        expected_keys = ['pd', 'pd_n', 'pd_nsigma', 'pd_type']
        for key in expected_keys:
            self.assertIn(key, PD_DEFAULTS)

    def test_pd_defaults_values(self):
        """Test default PD values."""
        self.assertEqual(PD_DEFAULTS['pd'], 0.0)  # No polydispersity by default
        self.assertEqual(PD_DEFAULTS['pd_n'], 35)
        self.assertEqual(PD_DEFAULTS['pd_nsigma'], 3.0)
        self.assertEqual(PD_DEFAULTS['pd_type'], 'gaussian')

    def test_pd_distribution_types(self):
        """Test that expected distribution types are available."""
        expected = ['gaussian', 'rectangle', 'lognormal', 'schulz', 'boltzmann']
        for dist_type in expected:
            self.assertIn(dist_type, PD_DISTRIBUTION_TYPES)


class TestParameterManagerPolydispersityDetection(unittest.TestCase):
    """Test polydispersity parameter detection."""

    def test_detect_polydisperse_params_from_mock_kernel(self):
        """Test detection of polydisperse parameters from mock kernel."""
        # Create mock kernel with polydisperse parameter
        mock_param_pd = Mock()
        mock_param_pd.name = 'radius'
        mock_param_pd.default = 20.0
        mock_param_pd.limits = (1.0, 100.0)
        mock_param_pd.description = 'Radius'
        mock_param_pd.polydisperse = True

        mock_param_no_pd = Mock()
        mock_param_no_pd.name = 'sld'
        mock_param_no_pd.default = 1.0
        mock_param_no_pd.limits = (-10.0, 10.0)
        mock_param_no_pd.description = 'SLD'
        mock_param_no_pd.polydisperse = False

        mock_kernel = Mock()
        mock_kernel.info.parameters.kernel_parameters = [mock_param_pd, mock_param_no_pd]

        pm = ParameterManager()
        pm.initialize_from_kernel(mock_kernel, 'sphere')

        pd_params = pm.get_polydisperse_parameters()
        self.assertIn('radius', pd_params)
        self.assertNotIn('sld', pd_params)

    def test_has_polydisperse_parameters_true(self):
        """Test has_polydisperse_parameters returns True when params exist."""
        mock_param = Mock()
        mock_param.name = 'radius'
        mock_param.default = 20.0
        mock_param.limits = (1.0, 100.0)
        mock_param.description = 'Radius'
        mock_param.polydisperse = True

        mock_kernel = Mock()
        mock_kernel.info.parameters.kernel_parameters = [mock_param]

        pm = ParameterManager()
        pm.initialize_from_kernel(mock_kernel, 'sphere')

        self.assertTrue(pm.has_polydisperse_parameters())

    def test_has_polydisperse_parameters_false(self):
        """Test has_polydisperse_parameters returns False when no params."""
        mock_param = Mock()
        mock_param.name = 'sld'
        mock_param.default = 1.0
        mock_param.limits = (-10.0, 10.0)
        mock_param.description = 'SLD'
        mock_param.polydisperse = False

        mock_kernel = Mock()
        mock_kernel.info.parameters.kernel_parameters = [mock_param]

        pm = ParameterManager()
        pm.initialize_from_kernel(mock_kernel, 'test_model')

        self.assertFalse(pm.has_polydisperse_parameters())


class TestParameterManagerPolydispersityInitialization(unittest.TestCase):
    """Test polydispersity parameter initialization."""

    def setUp(self):
        """Set up test fixtures."""
        mock_param = Mock()
        mock_param.name = 'radius'
        mock_param.default = 20.0
        mock_param.limits = (1.0, 100.0)
        mock_param.description = 'Radius'
        mock_param.polydisperse = True

        mock_kernel = Mock()
        mock_kernel.info.parameters.kernel_parameters = [mock_param]

        self.pm = ParameterManager()
        self.pm.initialize_from_kernel(mock_kernel, 'sphere')

    def test_pd_params_initialized_with_defaults(self):
        """Test that PD parameters are initialized with default values."""
        pd_config = self.pm.get_pd_param('radius')

        self.assertEqual(pd_config['pd'], PD_DEFAULTS['pd'])
        self.assertEqual(pd_config['pd_n'], PD_DEFAULTS['pd_n'])
        self.assertEqual(pd_config['pd_nsigma'], PD_DEFAULTS['pd_nsigma'])
        self.assertEqual(pd_config['pd_type'], PD_DEFAULTS['pd_type'])
        self.assertFalse(pd_config['vary'])

    def test_pd_disabled_by_default(self):
        """Test that polydispersity is disabled by default."""
        self.assertFalse(self.pm.is_pd_enabled())


class TestParameterManagerPolydispersityConfiguration(unittest.TestCase):
    """Test polydispersity parameter configuration."""

    def setUp(self):
        """Set up test fixtures."""
        mock_param = Mock()
        mock_param.name = 'radius'
        mock_param.default = 20.0
        mock_param.limits = (1.0, 100.0)
        mock_param.description = 'Radius'
        mock_param.polydisperse = True

        mock_kernel = Mock()
        mock_kernel.info.parameters.kernel_parameters = [mock_param]

        self.pm = ParameterManager()
        self.pm.initialize_from_kernel(mock_kernel, 'sphere')

    def test_set_pd_width(self):
        """Test setting PD width."""
        self.pm.set_pd_param('radius', pd_width=0.1)
        pd_config = self.pm.get_pd_param('radius')
        self.assertEqual(pd_config['pd'], 0.1)

    def test_set_pd_n(self):
        """Test setting PD n points."""
        self.pm.set_pd_param('radius', pd_n=50)
        pd_config = self.pm.get_pd_param('radius')
        self.assertEqual(pd_config['pd_n'], 50)

    def test_set_pd_nsigma(self):
        """Test setting PD nsigma."""
        self.pm.set_pd_param('radius', pd_nsigma=5.0)
        pd_config = self.pm.get_pd_param('radius')
        self.assertEqual(pd_config['pd_nsigma'], 5.0)

    def test_set_pd_type(self):
        """Test setting PD distribution type."""
        self.pm.set_pd_param('radius', pd_type='lognormal')
        pd_config = self.pm.get_pd_param('radius')
        self.assertEqual(pd_config['pd_type'], 'lognormal')

    def test_set_pd_vary(self):
        """Test setting PD vary flag."""
        self.pm.set_pd_param('radius', vary=True)
        pd_config = self.pm.get_pd_param('radius')
        self.assertTrue(pd_config['vary'])

    def test_set_all_pd_params_at_once(self):
        """Test setting all PD parameters at once."""
        self.pm.set_pd_param(
            'radius',
            pd_width=0.15,
            pd_n=40,
            pd_nsigma=4.0,
            pd_type='schulz',
            vary=True,
        )
        pd_config = self.pm.get_pd_param('radius')

        self.assertEqual(pd_config['pd'], 0.15)
        self.assertEqual(pd_config['pd_n'], 40)
        self.assertEqual(pd_config['pd_nsigma'], 4.0)
        self.assertEqual(pd_config['pd_type'], 'schulz')
        self.assertTrue(pd_config['vary'])

    def test_set_pd_param_invalid_param_raises_error(self):
        """Test that setting PD on non-polydisperse param raises error."""
        with self.assertRaises(KeyError):
            self.pm.set_pd_param('sld', pd_width=0.1)

    def test_set_pd_type_invalid_raises_error(self):
        """Test that invalid pd_type raises error."""
        with self.assertRaises(ValueError):
            self.pm.set_pd_param('radius', pd_type='invalid_distribution')


class TestParameterManagerPolydispersityVisibility(unittest.TestCase):
    """Test polydispersity visibility toggling."""

    def setUp(self):
        """Set up test fixtures."""
        mock_param = Mock()
        mock_param.name = 'radius'
        mock_param.default = 20.0
        mock_param.limits = (1.0, 100.0)
        mock_param.description = 'Radius'
        mock_param.polydisperse = True

        mock_kernel = Mock()
        mock_kernel.info.parameters.kernel_parameters = [mock_param]

        self.pm = ParameterManager()
        self.pm.initialize_from_kernel(mock_kernel, 'sphere')

    def test_toggle_pd_visibility_enable(self):
        """Test enabling polydispersity."""
        self.pm.toggle_pd_visibility(True)
        self.assertTrue(self.pm.is_pd_enabled())

    def test_toggle_pd_visibility_disable(self):
        """Test disabling polydispersity."""
        self.pm.toggle_pd_visibility(True)
        self.pm.toggle_pd_visibility(False)
        self.assertFalse(self.pm.is_pd_enabled())

    def test_pd_params_excluded_when_disabled(self):
        """Test that PD params are excluded when PD is disabled."""
        self.pm.set_pd_param('radius', pd_width=0.1)
        self.pm.toggle_pd_visibility(False)

        pd_params = self.pm.get_pd_params_for_fitting()
        self.assertEqual(pd_params, {})

    def test_pd_params_included_when_enabled(self):
        """Test that PD params are included when PD is enabled."""
        self.pm.set_pd_param('radius', pd_width=0.1)
        self.pm.toggle_pd_visibility(True)

        pd_params = self.pm.get_pd_params_for_fitting()
        self.assertIn('radius_pd', pd_params)
        self.assertEqual(pd_params['radius_pd'], 0.1)

    def test_pd_param_values_preserved_across_toggle(self):
        """Test that PD values are preserved when toggling visibility."""
        self.pm.set_pd_param('radius', pd_width=0.2, pd_type='lognormal')
        self.pm.toggle_pd_visibility(True)
        self.pm.toggle_pd_visibility(False)
        self.pm.toggle_pd_visibility(True)

        pd_config = self.pm.get_pd_param('radius')
        self.assertEqual(pd_config['pd'], 0.2)
        self.assertEqual(pd_config['pd_type'], 'lognormal')


class TestParameterManagerPolydispersityFitting(unittest.TestCase):
    """Test polydispersity parameters for fitting."""

    def setUp(self):
        """Set up test fixtures."""
        mock_param1 = Mock()
        mock_param1.name = 'radius'
        mock_param1.default = 20.0
        mock_param1.limits = (1.0, 100.0)
        mock_param1.description = 'Radius'
        mock_param1.polydisperse = True

        mock_param2 = Mock()
        mock_param2.name = 'length'
        mock_param2.default = 400.0
        mock_param2.limits = (10.0, 1000.0)
        mock_param2.description = 'Length'
        mock_param2.polydisperse = True

        mock_kernel = Mock()
        mock_kernel.info.parameters.kernel_parameters = [mock_param1, mock_param2]

        self.pm = ParameterManager()
        self.pm.initialize_from_kernel(mock_kernel, 'cylinder')

    def test_get_pd_params_for_fitting_format(self):
        """Test format of PD params for fitting."""
        self.pm.set_pd_param('radius', pd_width=0.1)
        self.pm.toggle_pd_visibility(True)

        pd_params = self.pm.get_pd_params_for_fitting()

        self.assertIn('radius_pd', pd_params)
        self.assertIn('radius_pd_n', pd_params)
        self.assertIn('radius_pd_nsigma', pd_params)
        self.assertIn('radius_pd_type', pd_params)

    def test_get_varying_pd_params(self):
        """Test getting list of varying PD parameters."""
        self.pm.set_pd_param('radius', vary=True)
        self.pm.set_pd_param('length', vary=False)
        self.pm.toggle_pd_visibility(True)

        varying = self.pm.get_varying_pd_params()
        self.assertIn('radius', varying)
        self.assertNotIn('length', varying)

    def test_get_varying_pd_params_when_disabled(self):
        """Test that no varying PD params returned when PD disabled."""
        self.pm.set_pd_param('radius', vary=True)
        self.pm.toggle_pd_visibility(False)

        varying = self.pm.get_varying_pd_params()
        self.assertEqual(varying, [])


class TestParameterManagerPolydispersityClear(unittest.TestCase):
    """Test that clear resets polydispersity state."""

    def test_clear_resets_pd_state(self):
        """Test that clear() resets all PD state."""
        mock_param = Mock()
        mock_param.name = 'radius'
        mock_param.default = 20.0
        mock_param.limits = (1.0, 100.0)
        mock_param.description = 'Radius'
        mock_param.polydisperse = True

        mock_kernel = Mock()
        mock_kernel.info.parameters.kernel_parameters = [mock_param]

        pm = ParameterManager()
        pm.initialize_from_kernel(mock_kernel, 'sphere')
        pm.set_pd_param('radius', pd_width=0.2)
        pm.toggle_pd_visibility(True)

        pm.clear()

        self.assertEqual(pm.get_polydisperse_parameters(), [])
        self.assertEqual(pm.polydisperse_params, {})
        self.assertFalse(pm.is_pd_enabled())


class TestSANSFitterPolydispersity(unittest.TestCase):
    """Test polydispersity methods in SANSFitter."""

    def setUp(self):
        """Set up test fixtures."""
        self.fitter = SANSFitter()
        self.fitter.set_model('cylinder')

    def test_supports_polydispersity_cylinder(self):
        """Test that cylinder model supports polydispersity."""
        self.assertTrue(self.fitter.supports_polydispersity())

    def test_get_polydisperse_parameters_cylinder(self):
        """Test getting polydisperse parameters for cylinder."""
        pd_params = self.fitter.get_polydisperse_parameters()
        # Cylinder should have radius, length as polydisperse
        self.assertIn('radius', pd_params)
        self.assertIn('length', pd_params)

    def test_set_pd_param(self):
        """Test setting PD parameters through SANSFitter."""
        self.fitter.set_pd_param('radius', pd_width=0.1, vary=True)
        pd_config = self.fitter.get_pd_param('radius')

        self.assertEqual(pd_config['pd'], 0.1)
        self.assertTrue(pd_config['vary'])

    def test_enable_polydispersity(self):
        """Test enabling polydispersity through SANSFitter."""
        self.fitter.enable_polydispersity(True)
        self.assertTrue(self.fitter.is_polydispersity_enabled())

    def test_disable_polydispersity(self):
        """Test disabling polydispersity through SANSFitter."""
        self.fitter.enable_polydispersity(True)
        self.fitter.enable_polydispersity(False)
        self.assertFalse(self.fitter.is_polydispersity_enabled())

    def test_model_switch_resets_pd(self):
        """Test that switching models resets PD configuration."""
        self.fitter.set_pd_param('radius', pd_width=0.1)
        self.fitter.enable_polydispersity(True)

        # Switch to sphere model
        self.fitter.set_model('sphere')

        # PD should be reset
        self.assertFalse(self.fitter.is_polydispersity_enabled())

    def test_sphere_model_polydispersity(self):
        """Test polydispersity support for sphere model."""
        self.fitter.set_model('sphere')
        self.assertTrue(self.fitter.supports_polydispersity())
        pd_params = self.fitter.get_polydisperse_parameters()
        self.assertIn('radius', pd_params)


class TestSANSFitterPolydispersityIntegration(unittest.TestCase):
    """Integration tests for polydispersity with fitting."""

    def setUp(self):
        """Set up test fixtures."""
        self.fitter = SANSFitter()
        self.data_file = create_decay_data_file()
        self.fitter.load_data(self.data_file)
        self.fitter.set_model('sphere')

        # Set up parameters
        self.fitter.set_param('radius', value=20.0, min=10.0, max=30.0, vary=True)
        self.fitter.set_param('scale', value=0.1, min=0.01, max=1.0, vary=True)
        self.fitter.set_param('background', value=0.01, min=0, max=0.1, vary=True)
        self.fitter.set_param('sld', value=2.0, vary=False)
        self.fitter.set_param('sld_solvent', value=3.0, vary=False)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.data_file):
            os.unlink(self.data_file)

    def test_fit_with_pd_disabled(self):
        """Test fitting with polydispersity disabled."""
        self.fitter.set_pd_param('radius', pd_width=0.1)
        self.fitter.enable_polydispersity(False)

        result = self.fitter.fit(engine='bumps', method='amoeba')
        self.assertIsNotNone(result)
        self.assertIn('chisq', result)

    def test_configure_pd_before_fit(self):
        """Test configuring PD parameters before fitting."""
        # Configure PD
        self.fitter.set_pd_param('radius', pd_width=0.1, pd_type='gaussian', vary=False)
        self.fitter.enable_polydispersity(True)

        # Verify configuration
        pd_config = self.fitter.get_pd_param('radius')
        self.assertEqual(pd_config['pd'], 0.1)
        self.assertEqual(pd_config['pd_type'], 'gaussian')
        self.assertTrue(self.fitter.is_polydispersity_enabled())


class TestPolydispersityDistributionTypes(unittest.TestCase):
    """Test different polydispersity distribution types in fitting."""

    def setUp(self):
        """Set up test fixtures."""
        self.fitter = SANSFitter()
        self.data_file = create_decay_data_file()
        self.fitter.load_data(self.data_file)
        self.fitter.set_model('sphere')

        # Set up parameters
        self.fitter.set_param('radius', value=20.0, min=10.0, max=30.0, vary=True)
        self.fitter.set_param('scale', value=0.1, vary=True)
        self.fitter.set_param('background', value=0.01, vary=True)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.data_file):
            os.unlink(self.data_file)

    def test_fit_with_gaussian_distribution(self):
        """Test fitting with Gaussian polydispersity distribution."""
        self.fitter.set_pd_param('radius', pd_width=0.1, pd_type='gaussian')
        self.fitter.enable_polydispersity(True)

        result = self.fitter.fit(engine='bumps', method='amoeba')
        self.assertIsNotNone(result)

    def test_fit_with_lognormal_distribution(self):
        """Test fitting with lognormal polydispersity distribution."""
        self.fitter.set_pd_param('radius', pd_width=0.1, pd_type='lognormal')
        self.fitter.enable_polydispersity(True)

        result = self.fitter.fit(engine='bumps', method='amoeba')
        self.assertIsNotNone(result)

    def test_fit_with_schulz_distribution(self):
        """Test fitting with Schulz polydispersity distribution."""
        self.fitter.set_pd_param('radius', pd_width=0.1, pd_type='schulz')
        self.fitter.enable_polydispersity(True)

        result = self.fitter.fit(engine='bumps', method='amoeba')
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main(verbosity=2)

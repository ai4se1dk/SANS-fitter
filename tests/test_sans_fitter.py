"""Unit tests for the public SANSFitter API."""

import os
import tempfile
import unittest
from unittest.mock import Mock

from sans_fitter import SANSFitter
from tests.helpers import create_decay_data_file


class TestGetAllModels(unittest.TestCase):
    """Test get_all_models function."""

    def test_get_all_models_returns_list(self):
        """Test that get_all_models returns a non-empty list."""
        from sans_fitter import get_all_models

        models = get_all_models()
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)

    def test_get_all_models_contains_common_models(self):
        """Test that common models are in the list."""
        from sans_fitter import get_all_models

        models = get_all_models()
        self.assertIn('sphere', models)
        self.assertIn('cylinder', models)
        self.assertIn('ellipsoid', models)

    def test_get_all_models_is_sorted(self):
        """Test that the model list is sorted."""
        from sans_fitter import get_all_models

        models = get_all_models()
        self.assertEqual(models, sorted(models))


class TestSANSFitterInitialization(unittest.TestCase):
    """Test SANSFitter initialization."""

    def test_init(self):
        """Test that SANSFitter initializes correctly."""
        fitter = SANSFitter()
        self.assertIsNone(fitter.data)
        self.assertIsNone(fitter.kernel)
        self.assertIsNone(fitter.model_name)
        self.assertEqual(fitter.params, {})
        self.assertIsNone(fitter.fit_result)
        self.assertIsNone(fitter._fitted_model)


class TestModelSetup(unittest.TestCase):
    """Test model setup and configuration."""

    def setUp(self):
        """Set up test fixtures."""
        self.fitter = SANSFitter()

    def test_set_model_cylinder(self):
        """Test loading cylinder model."""
        self.fitter.set_model('cylinder')
        self.assertIsNotNone(self.fitter.kernel)
        self.assertEqual(self.fitter.model_name, 'cylinder')
        self.assertGreater(len(self.fitter.params), 0)

        # Check that expected parameters exist
        expected_params = ['radius', 'length', 'sld', 'sld_solvent', 'background', 'scale']
        for param in expected_params:
            self.assertIn(param, self.fitter.params)

    def test_set_model_sphere(self):
        """Test loading sphere model."""
        self.fitter.set_model('sphere')
        self.assertIsNotNone(self.fitter.kernel)
        self.assertEqual(self.fitter.model_name, 'sphere')

        # Check that sphere-specific parameters exist
        self.assertIn('radius', self.fitter.params)
        self.assertNotIn('length', self.fitter.params)  # Sphere doesn't have length

    def test_set_model_invalid(self):
        """Test that invalid model name raises error."""
        with self.assertRaises(ValueError):
            self.fitter.set_model('invalid_model_name_xyz')

    def test_model_parameters_structure(self):
        """Test that parameters have correct structure."""
        self.fitter.set_model('cylinder')

        for _param_name, param_info in self.fitter.params.items():
            self.assertIn('value', param_info)
            self.assertIn('min', param_info)
            self.assertIn('max', param_info)
            self.assertIn('vary', param_info)
            self.assertIn('description', param_info)
            self.assertFalse(param_info['vary'])  # Default should be False


class TestParameterManagement(unittest.TestCase):
    """Test parameter management functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.fitter = SANSFitter()
        self.fitter.set_model('cylinder')

    def test_set_param_value(self):
        """Test setting parameter value."""
        self.fitter.set_param('radius', value=25.0)
        self.assertEqual(self.fitter.params['radius']['value'], 25.0)

    def test_set_param_bounds(self):
        """Test setting parameter bounds."""
        self.fitter.set_param('radius', min=5.0, max=50.0)
        self.assertEqual(self.fitter.params['radius']['min'], 5.0)
        self.assertEqual(self.fitter.params['radius']['max'], 50.0)

    def test_set_param_vary(self):
        """Test setting parameter vary flag."""
        self.fitter.set_param('radius', vary=True)
        self.assertTrue(self.fitter.params['radius']['vary'])

        self.fitter.set_param('radius', vary=False)
        self.assertFalse(self.fitter.params['radius']['vary'])

    def test_set_param_all_at_once(self):
        """Test setting all parameter attributes at once."""
        self.fitter.set_param('radius', value=20.0, min=10.0, max=30.0, vary=True)
        self.assertEqual(self.fitter.params['radius']['value'], 20.0)
        self.assertEqual(self.fitter.params['radius']['min'], 10.0)
        self.assertEqual(self.fitter.params['radius']['max'], 30.0)
        self.assertTrue(self.fitter.params['radius']['vary'])

    def test_set_param_invalid_name(self):
        """Test that setting invalid parameter raises KeyError."""
        with self.assertRaises(KeyError):
            self.fitter.set_param('invalid_param', value=1.0)

    def test_get_params_no_model(self):
        """Test get_params with no model loaded."""
        fitter = SANSFitter()
        # Should not raise error, just print message
        fitter.get_params()


class TestFittingPrerequisites(unittest.TestCase):
    """Test prerequisites for fitting."""

    def setUp(self):
        """Set up test fixtures."""
        self.fitter = SANSFitter()

    def test_fit_without_data_raises_error(self):
        """Test that fitting without data raises error."""
        self.fitter.set_model('cylinder')
        with self.assertRaises(ValueError) as context:
            self.fitter.fit()
        self.assertIn('No data loaded', str(context.exception))

    def test_fit_without_model_raises_error(self):
        """Test that fitting without model raises error."""
        # Create mock data
        self.fitter.data = Mock()
        with self.assertRaises(ValueError) as context:
            self.fitter.fit()
        self.assertIn('No model loaded', str(context.exception))

    def test_fit_invalid_engine_raises_error(self):
        """Test that invalid engine name raises error."""
        self.fitter.data = Mock()
        self.fitter.kernel = Mock()
        with self.assertRaises(ValueError) as context:
            self.fitter.fit(engine='invalid_engine')
        self.assertIn('Unknown engine', str(context.exception))


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""

    def test_complete_workflow_bumps(self):
        """Test complete workflow with BUMPS."""
        fitter = SANSFitter()
        data_file = create_decay_data_file(num_points=20)

        try:
            # Load data
            fitter.load_data(data_file)
            self.assertIsNotNone(fitter.data)

            # Set model
            fitter.set_model('sphere')
            self.assertEqual(fitter.model_name, 'sphere')

            # Configure parameters
            fitter.set_param('radius', value=20.0, min=10.0, max=30.0, vary=True)
            fitter.set_param('scale', value=0.1, min=0.01, max=1.0, vary=True)
            fitter.set_param('background', value=0.01, vary=True)
            fitter.set_param('sld', value=2.0, vary=False)
            fitter.set_param('sld_solvent', value=3.0, vary=False)

            # Fit
            result = fitter.fit(engine='bumps', method='amoeba')
            self.assertIsNotNone(result)
            self.assertIn('chisq', result)

            # Save results
            output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            output_file.close()
            output_path = output_file.name

            try:
                fitter.save_results(output_path)
                self.assertTrue(os.path.exists(output_path))
            finally:
                if os.path.exists(output_path):
                    os.unlink(output_path)

        finally:
            if os.path.exists(data_file):
                os.unlink(data_file)

    def test_model_switching(self):
        """Test switching between different models."""
        fitter = SANSFitter()

        # Load cylinder model
        fitter.set_model('cylinder')
        self.assertIn('length', fitter.params)

        # Switch to sphere model
        fitter.set_model('sphere')
        self.assertNotIn('length', fitter.params)
        self.assertIn('radius', fitter.params)


if __name__ == '__main__':
    unittest.main(verbosity=2)

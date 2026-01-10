"""
Unit tests for the SANSFitter class.

Tests cover:
- Data loading and validation
- Model initialization
- Parameter management
- Fitting with BUMPS engine
- Fitting with LMFit engine
- Result visualization and export
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np

# Add parent directory to path to import sans_fitter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sans_fitter import SANSFitter


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


class TestDataLoading(unittest.TestCase):
    """Test data loading functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.fitter = SANSFitter()

    def create_test_data_file(self):
        """Create a temporary CSV data file for testing."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write('Q,I,dI\n')
        for i in range(10):
            q = 0.01 * (i + 1)
            intensity = 100 * np.exp(-q * 10) + 0.1
            error = intensity * 0.1
            temp_file.write(f'{q},{intensity},{error}\n')
        temp_file.close()
        return temp_file.name

    def test_load_data_success(self):
        """Test successful data loading."""
        data_file = self.create_test_data_file()
        try:
            self.fitter.load_data(data_file)
            self.assertIsNotNone(self.fitter.data)
            self.assertTrue(hasattr(self.fitter.data, 'x'))
            self.assertTrue(hasattr(self.fitter.data, 'y'))
            self.assertTrue(hasattr(self.fitter.data, 'dy'))
            self.assertIsNotNone(self.fitter.data.qmin)
            self.assertIsNotNone(self.fitter.data.qmax)
        finally:
            os.unlink(data_file)

    def test_load_data_file_not_found(self):
        """Test that loading non-existent file raises error."""
        with self.assertRaises(ValueError):
            self.fitter.load_data('nonexistent_file.csv')

    def test_load_data_sets_mask(self):
        """Test that data loading sets up mask correctly."""
        data_file = self.create_test_data_file()
        try:
            self.fitter.load_data(data_file)
            self.assertTrue(hasattr(self.fitter.data, 'mask'))
            self.assertIsInstance(self.fitter.data.mask, np.ndarray)
        finally:
            os.unlink(data_file)


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


class TestBumpsFitting(unittest.TestCase):
    """Test BUMPS fitting functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.fitter = SANSFitter()
        self.data_file = self.create_test_data_file()
        self.fitter.load_data(self.data_file)
        self.fitter.set_model('sphere')

        # Set up basic parameters
        self.fitter.set_param('radius', value=20.0, min=10.0, max=30.0, vary=True)
        self.fitter.set_param('scale', value=0.1, min=0.01, max=1.0, vary=True)
        self.fitter.set_param('background', value=0.01, min=0, max=0.1, vary=True)
        self.fitter.set_param('sld', value=2.0, vary=False)
        self.fitter.set_param('sld_solvent', value=3.0, vary=False)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.data_file):
            os.unlink(self.data_file)

    def create_test_data_file(self):
        """Create synthetic SANS data for a sphere."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write('Q,I,dI\n')

        # Generate synthetic sphere scattering data
        q = np.logspace(-2, 0, 30)
        # Simplified form factor for testing
        intensity = 0.1 * (1 / (1 + q**2)) + 0.01
        d_intensity = intensity * 0.1

        for qi, intensity_i, d_intensity_i in zip(q, intensity, d_intensity):
            temp_file.write(f'{qi},{intensity_i},{d_intensity_i}\n')
        temp_file.close()
        return temp_file.name

    def test_bumps_fit_runs(self):
        """Test that BUMPS fit executes without error."""
        result = self.fitter.fit(engine='bumps', method='amoeba')

        self.assertIsNotNone(result)
        self.assertIn('engine', result)
        self.assertEqual(result['engine'], 'bumps')
        self.assertIn('chisq', result)
        self.assertIn('parameters', result)

    def test_bumps_fit_returns_parameters(self):
        """Test that BUMPS fit returns fitted parameters."""
        result = self.fitter.fit(engine='bumps', method='amoeba')
        # Check that varied parameters are in results
        self.assertIn('radius', result['parameters'])
        self.assertIn('scale', result['parameters'])
        self.assertIn('background', result['parameters'])

        # Check parameter structure
        for _param_name, param_result in result['parameters'].items():
            self.assertIn('value', param_result)
            self.assertIn('stderr', param_result)
            self.assertIn('formatted', param_result)

    def test_bumps_fit_updates_internal_params(self):
        """Test that fit updates internal parameter values."""
        self.fitter.params['radius']['value']
        self.fitter.fit(engine='bumps', method='amoeba')

        # After fitting, internal params should be updated
        self.assertIsNotNone(self.fitter.fit_result)

    def test_bumps_fit_stores_result(self):
        """Test that fit result is stored properly."""
        self.fitter.fit(engine='bumps', method='amoeba')

        self.assertIsNotNone(self.fitter.fit_result)
        self.assertIsNotNone(self.fitter._fitted_model)


class TestLMFitFitting(unittest.TestCase):
    """Test LMFit fitting functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.fitter = SANSFitter()
        self.data_file = self.create_test_data_file()
        self.fitter.load_data(self.data_file)
        self.fitter.set_model('sphere')

        # Set up basic parameters
        self.fitter.set_param('radius', value=20.0, min=10.0, max=30.0, vary=True)
        self.fitter.set_param('scale', value=0.1, min=0.01, max=1.0, vary=True)
        self.fitter.set_param('background', value=0.01, min=0, max=0.1, vary=True)
        self.fitter.set_param('sld', value=2.0, vary=False)
        self.fitter.set_param('sld_solvent', value=3.0, vary=False)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.data_file):
            os.unlink(self.data_file)

    def create_test_data_file(self):
        """Create synthetic SANS data for a sphere."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write('Q,I,dI\n')

        q = np.logspace(-2, 0, 30)
        intensity = 0.1 * (1 / (1 + q**2)) + 0.01
        d_intensity = intensity * 0.1

        for qi, intensity_i, d_intensity_i in zip(q, intensity, d_intensity):
            temp_file.write(f'{qi},{intensity_i},{d_intensity_i}\n')
        temp_file.close()
        return temp_file.name

    def test_lmfit_fit_runs(self):
        """Test that LMFit fit executes without error."""
        try:
            result = self.fitter.fit(engine='lmfit', method='leastsq')

            self.assertIsNotNone(result)
            self.assertIn('engine', result)
            self.assertEqual(result['engine'], 'lmfit')
            self.assertIn('chisq', result)
            self.assertIn('parameters', result)
        except ValueError as e:
            if 'lmfit is not installed' in str(e):
                self.skipTest('lmfit not installed')
            raise

    def test_lmfit_fit_returns_parameters(self):
        """Test that LMFit fit returns fitted parameters."""
        try:
            result = self.fitter.fit(engine='lmfit', method='leastsq')

            # Check that varied parameters are in results
            self.assertIn('radius', result['parameters'])
            self.assertIn('scale', result['parameters'])

            # Check parameter structure
            for _param_name, param_result in result['parameters'].items():
                self.assertIn('value', param_result)
                self.assertIn('stderr', param_result)
                self.assertIn('formatted', param_result)
        except ValueError as e:
            if 'lmfit is not installed' in str(e):
                self.skipTest('lmfit not installed')
            raise


class TestVisualization(unittest.TestCase):
    """Test visualization functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.fitter = SANSFitter()
        self.data_file = self.create_test_data_file()
        self.fitter.load_data(self.data_file)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.data_file):
            os.unlink(self.data_file)

    def create_test_data_file(self):
        """Create synthetic SANS data."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write('Q,I,dI\n')
        q = np.logspace(-2, 0, 20)
        intensity = 0.1 * (1 / (1 + q**2)) + 0.01
        d_intensity = intensity * 0.1
        for qi, intensity_i, d_intensity_i in zip(q, intensity, d_intensity):
            temp_file.write(f'{qi},{intensity_i},{d_intensity_i}\n')
        temp_file.close()
        return temp_file.name

    def test_plot_data_only(self):
        """Test plotting without fit results."""
        with patch('plotly.graph_objects.Figure.show'):
            # Should not raise error
            self.fitter.plot_results()

    def test_plot_without_data_raises_error(self):
        """Test that plotting without data raises error."""
        fitter = SANSFitter()
        with self.assertRaises(ValueError):
            fitter.plot_results()

    @patch('plotly.graph_objects.Figure.show')
    def test_plot_with_fit_results(self, mock_show):
        """Test plotting with fit results."""
        self.fitter.set_model('sphere')
        self.fitter.set_param('radius', value=20.0, min=10.0, max=30.0, vary=True)
        self.fitter.set_param('scale', value=0.1, min=0.01, max=1.0, vary=True)
        self.fitter.set_param('background', value=0.01, vary=True)
        self.fitter.set_param('sld', value=2.0, vary=False)
        self.fitter.set_param('sld_solvent', value=3.0, vary=False)

        self.fitter.fit(engine='bumps', method='amoeba')

        # Should not raise error
        self.fitter.plot_results(show_residuals=True, log_scale=True)
        self.fitter.plot_results(show_residuals=False, log_scale=False)


class TestResultExport(unittest.TestCase):
    """Test result export functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.fitter = SANSFitter()
        self.data_file = self.create_test_data_file()
        self.fitter.load_data(self.data_file)
        self.fitter.set_model('sphere')

        self.fitter.set_param('radius', value=20.0, min=10.0, max=30.0, vary=True)
        self.fitter.set_param('scale', value=0.1, min=0.01, max=1.0, vary=True)
        self.fitter.set_param('background', value=0.01, vary=True)
        self.fitter.set_param('sld', value=2.0, vary=False)
        self.fitter.set_param('sld_solvent', value=3.0, vary=False)

        self.fitter.fit(engine='bumps', method='amoeba')

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.data_file):
            os.unlink(self.data_file)

    def create_test_data_file(self):
        """Create synthetic SANS data."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write('Q,I,dI\n')
        q = np.logspace(-2, 0, 20)
        intensity = 0.1 * (1 / (1 + q**2)) + 0.01
        d_intensity = intensity * 0.1
        for qi, intensity_i, d_intensity_i in zip(q, intensity, d_intensity):
            temp_file.write(f'{qi},{intensity_i},{d_intensity_i}\n')
        temp_file.close()
        return temp_file.name

    def test_save_results_creates_file(self):
        """Test that save_results creates output file."""
        output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        output_file.close()
        output_path = output_file.name

        try:
            self.fitter.save_results(output_path)
            self.assertTrue(os.path.exists(output_path))

            # Check file content
            with open(output_path) as f:
                content = f.read()
                self.assertIn('SANS Fit Results', content)
                self.assertIn('Model:', content)
                self.assertIn('Chi-squared:', content)
                self.assertIn('Q,I_exp,dI_exp,I_fit,Residuals', content)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_save_results_without_fit_raises_error(self):
        """Test that saving without fit results raises error."""
        fitter = SANSFitter()
        with self.assertRaises(ValueError):
            fitter.save_results('output.csv')


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows."""

    def test_complete_workflow_bumps(self):
        """Test complete workflow with BUMPS."""
        fitter = SANSFitter()
        data_file = self.create_test_data_file()

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

    def create_test_data_file(self):
        """Create synthetic SANS data."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write('Q,I,dI\n')
        q = np.logspace(-2, 0, 20)
        intensity = 0.1 * (1 / (1 + q**2)) + 0.01
        d_intensity = intensity * 0.1
        for qi, intensity_i, d_intensity_i in zip(q, intensity, d_intensity):
            temp_file.write(f'{qi},{intensity_i},{d_intensity_i}\n')
        temp_file.close()
        return temp_file.name


class TestStructureFactorSetup(unittest.TestCase):
    """Test structure factor setup and configuration."""

    def setUp(self):
        """Set up test fixtures."""
        self.fitter = SANSFitter()
        self.fitter.set_model('sphere')

    def test_set_structure_factor_hardsphere(self):
        """Test setting hardsphere structure factor."""
        self.fitter.set_structure_factor('hardsphere')

        self.assertEqual(self.fitter.get_structure_factor(), 'hardsphere')
        self.assertIn('volfraction', self.fitter.params)
        self.assertIn('radius_effective', self.fitter.params)

    def test_set_structure_factor_hayter_msa(self):
        """Test setting hayter_msa structure factor."""
        self.fitter.set_structure_factor('hayter_msa')

        self.assertEqual(self.fitter.get_structure_factor(), 'hayter_msa')
        self.assertIn('volfraction', self.fitter.params)
        self.assertIn('radius_effective', self.fitter.params)
        self.assertIn('charge', self.fitter.params)

    def test_set_structure_factor_squarewell(self):
        """Test setting squarewell structure factor."""
        self.fitter.set_structure_factor('squarewell')

        self.assertEqual(self.fitter.get_structure_factor(), 'squarewell')
        self.assertIn('volfraction', self.fitter.params)
        self.assertIn('radius_effective', self.fitter.params)

    def test_set_structure_factor_stickyhardsphere(self):
        """Test setting stickyhardsphere structure factor."""
        self.fitter.set_structure_factor('stickyhardsphere')

        self.assertEqual(self.fitter.get_structure_factor(), 'stickyhardsphere')
        self.assertIn('volfraction', self.fitter.params)
        self.assertIn('radius_effective', self.fitter.params)

    def test_set_structure_factor_without_model_raises_error(self):
        """Test that setting structure factor without model raises error."""
        fitter = SANSFitter()
        with self.assertRaises(ValueError) as context:
            fitter.set_structure_factor('hardsphere')
        self.assertIn('No form factor model loaded', str(context.exception))

    def test_set_structure_factor_invalid_name_raises_error(self):
        """Test that invalid structure factor name raises error."""
        with self.assertRaises(ValueError) as context:
            self.fitter.set_structure_factor('invalid_structure_factor')
        self.assertIn('Unsupported structure factor', str(context.exception))

    def test_set_structure_factor_invalid_mode_raises_error(self):
        """Test that invalid radius_effective_mode raises error."""
        with self.assertRaises(ValueError) as context:
            self.fitter.set_structure_factor('hardsphere', radius_effective_mode='invalid_mode')
        self.assertIn('Invalid radius_effective_mode', str(context.exception))

    def test_structure_factor_preserves_form_factor_params(self):
        """Test that form factor parameters are preserved when adding structure factor."""
        # Set a specific radius value
        self.fitter.set_param('radius', value=50.0, min=10.0, max=100.0)

        # Add structure factor
        self.fitter.set_structure_factor('hardsphere')

        # Check that radius value is preserved
        self.assertEqual(self.fitter.params['radius']['value'], 50.0)
        self.assertEqual(self.fitter.params['radius']['min'], 10.0)
        self.assertEqual(self.fitter.params['radius']['max'], 100.0)


class TestStructureFactorLinkRadius(unittest.TestCase):
    """Test radius_effective linking functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.fitter = SANSFitter()
        self.fitter.set_model('sphere')
        self.fitter.set_param('radius', value=50.0)

    def test_link_radius_mode_sets_radius_effective(self):
        """Test that link_radius mode sets radius_effective to radius value."""
        self.fitter.set_structure_factor('hardsphere', radius_effective_mode='link_radius')

        self.assertEqual(self.fitter.params['radius_effective']['value'], 50.0)
        self.assertFalse(self.fitter.params['radius_effective']['vary'])

    def test_link_radius_mode_syncs_on_set_param(self):
        """Test that changing radius updates radius_effective in link_radius mode."""
        self.fitter.set_structure_factor('hardsphere', radius_effective_mode='link_radius')

        # Update radius
        self.fitter.set_param('radius', value=75.0)

        # radius_effective should be synced
        self.assertEqual(self.fitter.params['radius_effective']['value'], 75.0)

    def test_unconstrained_mode_allows_independent_radius_effective(self):
        """Test that unconstrained mode allows setting radius_effective independently."""
        self.fitter.set_structure_factor('hardsphere', radius_effective_mode='unconstrained')

        # Set radius_effective to a different value
        self.fitter.set_param('radius_effective', value=60.0, vary=True)

        # Should be independent of radius
        self.assertEqual(self.fitter.params['radius_effective']['value'], 60.0)
        self.assertTrue(self.fitter.params['radius_effective']['vary'])


class TestStructureFactorRemoval(unittest.TestCase):
    """Test structure factor removal functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.fitter = SANSFitter()
        self.fitter.set_model('sphere')
        self.fitter.set_param('radius', value=50.0, min=10.0, max=100.0)

    def test_remove_structure_factor(self):
        """Test removing a structure factor."""
        self.fitter.set_structure_factor('hardsphere')
        self.assertIsNotNone(self.fitter.get_structure_factor())

        self.fitter.remove_structure_factor()

        self.assertIsNone(self.fitter.get_structure_factor())
        self.assertNotIn('volfraction', self.fitter.params)
        self.assertNotIn('radius_effective', self.fitter.params)

    def test_remove_structure_factor_restores_params(self):
        """Test that removing structure factor restores original parameters."""
        original_radius = self.fitter.params['radius']['value']

        self.fitter.set_structure_factor('hardsphere')
        self.fitter.remove_structure_factor()

        self.assertEqual(self.fitter.params['radius']['value'], original_radius)

    def test_remove_structure_factor_without_one_raises_error(self):
        """Test that removing when no structure factor is set raises error."""
        with self.assertRaises(ValueError) as context:
            self.fitter.remove_structure_factor()
        self.assertIn('No structure factor is currently set', str(context.exception))


class TestStructureFactorModelSwitching(unittest.TestCase):
    """Test structure factor behavior when switching models."""

    def setUp(self):
        """Set up test fixtures."""
        self.fitter = SANSFitter()

    def test_set_model_resets_structure_factor(self):
        """Test that setting a new model resets the structure factor."""
        self.fitter.set_model('sphere')
        self.fitter.set_structure_factor('hardsphere')

        # Switch to cylinder model
        self.fitter.set_model('cylinder')

        # Structure factor should be reset
        self.assertIsNone(self.fitter.get_structure_factor())
        self.assertNotIn('volfraction', self.fitter.params)


class TestStructureFactorFitting(unittest.TestCase):
    """Test fitting with structure factors."""

    def setUp(self):
        """Set up test fixtures."""
        self.fitter = SANSFitter()
        self.data_file = self.create_test_data_file()
        self.fitter.load_data(self.data_file)
        self.fitter.set_model('sphere')

        # Set up form factor parameters
        self.fitter.set_param('radius', value=50.0, min=10.0, max=100.0, vary=True)
        self.fitter.set_param('scale', value=0.01, min=0.001, max=1.0, vary=True)
        self.fitter.set_param('background', value=0.001, min=0, max=0.1, vary=True)
        self.fitter.set_param('sld', value=1.0, vary=False)
        self.fitter.set_param('sld_solvent', value=6.0, vary=False)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.data_file):
            os.unlink(self.data_file)

    def create_test_data_file(self):
        """Create synthetic SANS data for concentrated spheres."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write('Q,I,dI\n')

        q = np.logspace(-2, 0, 30)
        # Approximate concentrated sphere scattering
        intensity = (
            0.01 * (1 / (1 + (q * 50) ** 2)) * (1 - 0.2 * np.sin(q * 100) / (q * 100 + 1e-10))
        )
        intensity = np.maximum(intensity, 0.001)
        d_intensity = intensity * 0.1

        for qi, intensity_i, d_intensity_i in zip(q, intensity, d_intensity):
            temp_file.write(f'{qi},{intensity_i},{d_intensity_i}\n')
        temp_file.close()
        return temp_file.name

    def test_fit_with_hardsphere_bumps(self):
        """Test fitting with hardsphere structure factor using BUMPS."""
        self.fitter.set_structure_factor('hardsphere')
        self.fitter.set_param('volfraction', value=0.2, min=0.0, max=0.6, vary=True)
        self.fitter.set_param('radius_effective', value=50.0, min=10.0, max=100.0, vary=True)

        result = self.fitter.fit(engine='bumps', method='amoeba')

        self.assertIsNotNone(result)
        self.assertIn('volfraction', result['parameters'])
        self.assertIn('radius_effective', result['parameters'])

    def test_fit_with_hardsphere_link_radius_bumps(self):
        """Test fitting with hardsphere and linked radius using BUMPS."""
        self.fitter.set_structure_factor('hardsphere', radius_effective_mode='link_radius')
        self.fitter.set_param('volfraction', value=0.2, min=0.0, max=0.6, vary=True)

        result = self.fitter.fit(engine='bumps', method='amoeba')

        self.assertIsNotNone(result)
        self.assertIn('volfraction', result['parameters'])

    def test_fit_with_hardsphere_lmfit(self):
        """Test fitting with hardsphere structure factor using LMFit."""
        self.fitter.set_structure_factor('hardsphere')
        self.fitter.set_param('volfraction', value=0.2, min=0.01, max=0.6, vary=True)
        self.fitter.set_param('radius_effective', value=50.0, min=10.0, max=100.0, vary=True)

        try:
            result = self.fitter.fit(engine='lmfit', method='least_squares')

            self.assertIsNotNone(result)
            self.assertIn('volfraction', result['parameters'])
            self.assertIn('radius_effective', result['parameters'])
        except ValueError as e:
            if 'scipy is not installed' in str(e):
                self.skipTest('scipy not installed')
            raise


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)

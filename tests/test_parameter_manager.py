"""
Unit tests for the ParameterManager class.

Tests cover:
- Parameter initialization
- Parameter validation and setting
- Structure factor handling
- Parameter backup and restore
"""

import os
import sys
import unittest
from unittest.mock import Mock

import numpy as np

# Add parent directory to path to import sans_fitter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sans_fitter import ParameterManager


class TestParameterManagerInitialization(unittest.TestCase):
    """Test ParameterManager initialization."""

    def test_init(self):
        """Test that ParameterManager initializes correctly."""
        pm = ParameterManager()
        self.assertEqual(pm.params, {})
        self.assertIsNone(pm.model_name)
        self.assertIsNone(pm.get_structure_factor())
        self.assertEqual(pm.get_radius_effective_mode(), 'unconstrained')

    def test_initialize_from_kernel(self):
        """Test parameter initialization from a mock kernel."""
        # Create a mock kernel
        mock_param = Mock()
        mock_param.name = 'radius'
        mock_param.default = 20.0
        mock_param.limits = (1.0, 100.0)
        mock_param.description = 'Radius of the sphere'

        mock_kernel = Mock()
        mock_kernel.info.parameters.kernel_parameters = [mock_param]

        pm = ParameterManager()
        pm.initialize_from_kernel(mock_kernel, 'sphere')

        # Check that parameters were initialized
        self.assertIn('radius', pm.params)
        self.assertIn('scale', pm.params)
        self.assertIn('background', pm.params)
        self.assertEqual(pm.params['radius']['value'], 20.0)
        self.assertEqual(pm.params['radius']['min'], 1.0)
        self.assertEqual(pm.params['radius']['max'], 100.0)
        self.assertFalse(pm.params['radius']['vary'])


class TestParameterManagement(unittest.TestCase):
    """Test parameter management operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.pm = ParameterManager()
        # Manually add some test parameters
        self.pm.params = {
            'radius': {
                'value': 20.0,
                'min': 1.0,
                'max': 100.0,
                'vary': False,
                'description': 'Radius',
            },
            'length': {
                'value': 400.0,
                'min': 10.0,
                'max': 1000.0,
                'vary': False,
                'description': 'Length',
            },
        }
        self.pm.model_name = 'cylinder'

    def test_set_param_value(self):
        """Test setting parameter value."""
        self.pm.set_param('radius', value=25.0)
        self.assertEqual(self.pm.params['radius']['value'], 25.0)

    def test_set_param_bounds(self):
        """Test setting parameter bounds."""
        self.pm.set_param('radius', min=5.0, max=50.0)
        self.assertEqual(self.pm.params['radius']['min'], 5.0)
        self.assertEqual(self.pm.params['radius']['max'], 50.0)

    def test_set_param_vary(self):
        """Test setting parameter vary flag."""
        self.pm.set_param('radius', vary=True)
        self.assertTrue(self.pm.params['radius']['vary'])

    def test_set_param_invalid_name(self):
        """Test that setting invalid parameter raises error."""
        with self.assertRaises(KeyError):
            self.pm.set_param('invalid_param', value=10.0)

    def test_validate_param(self):
        """Test parameter validation."""
        self.assertTrue(self.pm.validate_param('radius'))
        self.assertFalse(self.pm.validate_param('invalid'))

    def test_get_param_values(self):
        """Test getting parameter values dictionary."""
        values = self.pm.get_param_values()
        self.assertEqual(values['radius'], 20.0)
        self.assertEqual(values['length'], 400.0)

    def test_get_varying_params(self):
        """Test getting list of varying parameters."""
        self.pm.set_param('radius', vary=True)
        varying = self.pm.get_varying_params()
        self.assertIn('radius', varying)
        self.assertNotIn('length', varying)

    def test_update_param_value(self):
        """Test updating a parameter value."""
        self.pm.update_param_value('radius', 30.0)
        self.assertEqual(self.pm.params['radius']['value'], 30.0)


class TestParameterBackupRestore(unittest.TestCase):
    """Test parameter backup and restore functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.pm = ParameterManager()
        self.pm.params = {
            'radius': {
                'value': 20.0,
                'min': 1.0,
                'max': 100.0,
                'vary': False,
                'description': 'Radius',
            }
        }

    def test_backup_params(self):
        """Test backing up parameters."""
        self.assertFalse(self.pm.has_backed_up_params())
        self.pm.backup_params()
        self.assertTrue(self.pm.has_backed_up_params())

    def test_restore_params(self):
        """Test restoring backed up parameters."""
        self.pm.backup_params()
        self.pm.params['radius']['value'] = 30.0
        self.pm.restore_params()
        self.assertEqual(self.pm.params['radius']['value'], 20.0)
        self.assertFalse(self.pm.has_backed_up_params())

    def test_get_backed_up_params(self):
        """Test getting backed up parameters."""
        self.pm.backup_params()
        backed_up = self.pm.get_backed_up_params()
        self.assertEqual(backed_up['radius']['value'], 20.0)


class TestStructureFactorManagement(unittest.TestCase):
    """Test structure factor management."""

    def setUp(self):
        """Set up test fixtures."""
        self.pm = ParameterManager()
        self.pm.params = {
            'radius': {
                'value': 20.0,
                'min': 1.0,
                'max': 100.0,
                'vary': False,
                'description': 'Radius',
            }
        }
        self.pm.model_name = 'sphere'

    def test_remove_structure_factor_without_one(self):
        """Test that removing SF without one set raises error."""
        with self.assertRaises(ValueError):
            self.pm.remove_structure_factor()

    def test_get_structure_factor_none(self):
        """Test getting structure factor when none is set."""
        self.assertIsNone(self.pm.get_structure_factor())


class TestParameterClear(unittest.TestCase):
    """Test clearing all parameters."""

    def test_clear(self):
        """Test that clear resets all state."""
        pm = ParameterManager()
        pm.params = {'radius': {'value': 20.0}}
        pm.model_name = 'sphere'
        pm.backup_params()

        pm.clear()

        self.assertEqual(pm.params, {})
        self.assertIsNone(pm.model_name)
        self.assertIsNone(pm.get_structure_factor())
        self.assertFalse(pm.has_backed_up_params())


if __name__ == '__main__':
    unittest.main()

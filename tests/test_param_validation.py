"""Parameter validation and read-only state (issue #32).

Three invariants are pinned here:

* ``set_param`` rejects inverted bounds and out-of-range values *up front*,
  instead of letting them surface as a cryptic optimizer failure;
* a ``radius_effective`` that follows ``radius`` cannot be given a value or a
  vary flag, since both engines overwrite it anyway;
* ``SANSFitter.params`` and ``model_name`` are read-only, so no caller can
  bypass those checks by writing into the dictionary.
"""

import unittest

from sans_fitter import SANSFitter


class TestBoundsValidation(unittest.TestCase):
    def setUp(self):
        self.fitter = SANSFitter()
        self.fitter.set_model('sphere')
        self.fitter.set_param('radius', value=50.0, min=10.0, max=100.0)

    def test_inverted_bounds_in_one_call_are_rejected(self):
        with self.assertRaises(ValueError) as context:
            self.fitter.set_param('radius', min=50.0, max=10.0)
        self.assertIn('greater than max', str(context.exception))

    def test_bounds_inverted_by_a_partial_update_are_rejected(self):
        with self.assertRaises(ValueError):
            self.fitter.set_param('radius', min=200.0)

    def test_value_outside_existing_bounds_is_rejected(self):
        with self.assertRaises(ValueError) as context:
            self.fitter.set_param('radius', value=1000.0)
        self.assertIn('outside its bounds', str(context.exception))

    def test_value_outside_new_bounds_is_rejected(self):
        with self.assertRaises(ValueError):
            self.fitter.set_param('radius', value=5.0, min=10.0, max=100.0)

    def test_value_and_widened_bounds_in_one_call_are_accepted(self):
        self.fitter.set_param('radius', value=1000.0, min=10.0, max=2000.0)
        self.assertEqual(self.fitter.params['radius']['value'], 1000.0)
        self.assertEqual(self.fitter.params['radius']['max'], 2000.0)

    def test_value_on_a_bound_is_accepted(self):
        self.fitter.set_param('radius', value=100.0)
        self.assertEqual(self.fitter.params['radius']['value'], 100.0)

    def test_a_rejected_call_writes_nothing(self):
        with self.assertRaises(ValueError):
            self.fitter.set_param('radius', value=1000.0, min=20.0, vary=True)
        entry = self.fitter.params['radius']
        self.assertEqual(entry['value'], 50.0)
        self.assertEqual(entry['min'], 10.0)
        self.assertFalse(entry['vary'])

    def test_infinite_default_bounds_still_accept_values(self):
        self.fitter.set_param('scale', value=12.5)
        self.assertEqual(self.fitter.params['scale']['value'], 12.5)

    def test_fitted_values_bypass_the_bounds_check(self):
        """leastsq ignores bounds; a fit that leaves them must not be discarded."""
        manager = self.fitter._param_manager
        manager.apply_fitted_values({'radius': 5000.0})
        self.assertEqual(self.fitter.params['radius']['value'], 5000.0)


class TestLinkedRadiusEffectiveGuard(unittest.TestCase):
    def setUp(self):
        self.fitter = SANSFitter()
        self.fitter.set_model('sphere')
        self.fitter.set_param('radius', value=50.0, min=10.0, max=100.0)

    def test_vary_on_a_linked_radius_effective_is_rejected(self):
        self.fitter.set_structure_factor('hardsphere', radius_effective_mode='link_radius')
        with self.assertRaises(ValueError) as context:
            self.fitter.set_param('radius_effective', vary=True)
        self.assertIn('link_radius', str(context.exception))
        self.assertFalse(self.fitter.params['radius_effective']['vary'])

    def test_value_on_a_linked_radius_effective_is_rejected(self):
        self.fitter.set_structure_factor('hardsphere', radius_effective_mode='link_radius')
        with self.assertRaises(ValueError):
            self.fitter.set_param('radius_effective', value=80.0)
        self.assertEqual(self.fitter.params['radius_effective']['value'], 50.0)

    def test_bounds_and_vary_false_remain_settable(self):
        self.fitter.set_structure_factor('hardsphere', radius_effective_mode='link_radius')
        self.fitter.set_param('radius_effective', min=1.0, max=500.0, vary=False)
        self.assertEqual(self.fitter.params['radius_effective']['min'], 1.0)

    def test_the_target_still_drives_the_follower(self):
        self.fitter.set_structure_factor('hardsphere', radius_effective_mode='link_radius')
        self.fitter.set_param('radius', value=75.0)
        self.assertEqual(self.fitter.params['radius_effective']['value'], 75.0)

    def test_unconstrained_mode_is_unaffected(self):
        self.fitter.set_structure_factor('hardsphere', radius_effective_mode='unconstrained')
        self.fitter.set_param('radius_effective', value=60.0, min=10.0, max=100.0, vary=True)
        self.assertEqual(self.fitter.params['radius_effective']['value'], 60.0)
        self.assertTrue(self.fitter.params['radius_effective']['vary'])


class TestReadOnlyState(unittest.TestCase):
    def setUp(self):
        self.fitter = SANSFitter()
        self.fitter.set_model('sphere')

    def test_params_cannot_be_replaced(self):
        with self.assertRaises(AttributeError) as context:
            self.fitter.params = {}
        self.assertIn('set_param', str(context.exception))

    def test_a_parameter_entry_cannot_be_replaced(self):
        with self.assertRaises(TypeError):
            self.fitter.params['radius'] = {'value': 1.0}

    def test_a_parameter_field_cannot_be_written(self):
        with self.assertRaises(TypeError):
            self.fitter.params['radius']['vary'] = True
        self.assertFalse(self.fitter.params['radius']['vary'])

    def test_model_name_cannot_be_assigned(self):
        with self.assertRaises(AttributeError) as context:
            self.fitter.model_name = 'cylinder'
        self.assertIn('set_model', str(context.exception))
        self.assertEqual(self.fitter.model_name, 'sphere')

    def test_reads_still_work(self):
        self.assertIn('radius', self.fitter.params)
        self.assertEqual(self.fitter.model_name, 'sphere')
        self.assertEqual(sorted(self.fitter.params)[0], 'background')
        self.assertEqual(dict(self.fitter.params['radius'])['value'], 50.0)

    def test_the_view_reflects_later_changes(self):
        self.fitter.set_param('radius', value=42.0)
        self.assertEqual(self.fitter.params['radius']['value'], 42.0)

    def test_an_empty_fitter_still_compares_equal_to_an_empty_dict(self):
        self.assertEqual(SANSFitter().params, {})


if __name__ == '__main__':
    unittest.main()

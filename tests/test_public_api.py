"""Locks in the public API surface preserved across the 0.3 restructuring.

The package layout may move (data/, modeling/, inversion/), but everything
asserted here is a stability guarantee for downstream users: the names in
``__all__`` and the ``data_ops``/``pr_inversion`` module aliases must keep
working from the package root.
"""

import unittest

import sans_fitter


class TestPublicApi(unittest.TestCase):
    def test_all_names_are_importable(self):
        for name in sans_fitter.__all__:
            self.assertTrue(hasattr(sans_fitter, name), f'missing public name: {name}')

    def test_expected_names_in_all(self):
        expected = {
            'SANSFitter',
            'ParameterManager',
            'PD_DEFAULTS',
            'PD_DISTRIBUTION_TYPES',
            'get_all_models',
            'FitResultContract',
            'PosteriorSummary',
            'data_ops',
            'examples',
            'pr_inversion',
            'PrResult',
            'InsufficientDataError',
            'PrEstimationError',
        }
        self.assertLessEqual(expected, set(sans_fitter.__all__))

    def test_module_aliases_resolve_to_canonical_modules(self):
        from sans_fitter import data_ops, pr_inversion

        self.assertEqual(data_ops.__name__, 'sans_fitter.data.ops')
        self.assertEqual(pr_inversion.__name__, 'sans_fitter.inversion')

    def test_pr_inversion_alias_exposes_public_entry_points(self):
        from sans_fitter import pr_inversion

        for name in ('invert', 'auto_invert', 'estimate_alpha', 'estimate_n_terms', 'explore_dmax'):
            self.assertTrue(callable(getattr(pr_inversion, name)), name)


if __name__ == '__main__':
    unittest.main()

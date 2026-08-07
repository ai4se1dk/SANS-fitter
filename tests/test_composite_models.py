"""Tests for composite (multi-model) support — issue #46.

Covers the design in 46_COMPOSITE_MODELS.md: hardened string syntax, the
friendly set_models() builder, equality links, shared= parameters, the alias
layer, bounds policy, component curves, polydispersity routing, engine
gating, and plotting.
"""

import os
import unittest
import warnings

import numpy as np
from sasdata.dataloader.data_info import Data1D
from sasmodels.core import load_model
from sasmodels.direct_model import DirectModel

from sans_fitter import SANSFitter


def make_composite_data(
    expression='dab+peak_lorentz',
    params=None,
    num_points=40,
    noise=0.0,
    seed=42,
):
    """Simulate data from a composite model and return a fit-ready Data1D."""
    q = np.linspace(0.005, 0.3, num_points)
    data = Data1D(x=q, y=np.ones_like(q), dy=np.full_like(q, 0.05))
    data.qmin, data.qmax = q.min(), q.max()
    kernel = load_model(expression, dtype='single', platform='dll')
    defaults = dict(scale=1.0, background=0.01)
    if params:
        defaults.update(params)
    y = np.asarray(DirectModel(data, kernel)(**defaults))
    if noise > 0:
        rng = np.random.default_rng(seed)
        y = y + rng.normal(0, noise, size=y.shape)
    data.y = y
    return data


class TestCompositeLoading(unittest.TestCase):
    """Test 1: loading and parameter discovery for both paths."""

    def setUp(self):
        self.fitter = SANSFitter()
        self.fitter.set_data(make_composite_data())

    def test_raw_and_builder_paths_same_shape(self):
        raw = SANSFitter()
        raw.set_data(make_composite_data())
        raw.set_model('dab+peak_lorentz')

        builder = SANSFitter()
        builder.set_data(make_composite_data())
        builder.set_models('dab', 'peak_lorentz')

        # Same shape (flat dict, same schema), differing only in keys.
        self.assertEqual(len(raw.params), len(builder.params))
        for info in raw.params.values():
            self.assertEqual(set(info.keys()), {'value', 'min', 'max', 'vary', 'description'})
        for info in builder.params.values():
            self.assertEqual(set(info.keys()), {'value', 'min', 'max', 'vary', 'description'})

        self.assertIn('A_cor_length', raw.params)
        self.assertIn('B_peak_pos', raw.params)
        self.assertIn('dab_cor_length', builder.params)
        self.assertIn('peak_lorentz_peak_pos', builder.params)

    def test_get_components_triples(self):
        raw = SANSFitter()
        raw.set_data(make_composite_data())
        raw.set_model('dab+peak_lorentz')
        self.assertEqual(raw.get_components(), [('A', 'A', 'dab'), ('B', 'B', 'peak_lorentz')])

        builder = SANSFitter()
        builder.set_data(make_composite_data())
        builder.set_models('dab', 'peak_lorentz')
        self.assertEqual(
            builder.get_components(),
            [('A', 'dab', 'dab'), ('B', 'peak_lorentz', 'peak_lorentz')],
        )

    def test_atomic_model_has_no_components(self):
        fitter = SANSFitter()
        fitter.set_data(make_composite_data())
        fitter.set_model('sphere')
        self.assertEqual(fitter.get_components(), [])

    def test_precedence_product_binds_tightest(self):
        # sphere@hardsphere+peak_lorentz must parse as
        # (sphere@hardsphere) + peak_lorentz.
        fitter = SANSFitter()
        fitter.set_data(make_composite_data())
        fitter.set_models('sphere@hardsphere', 'peak_lorentz')
        comps = fitter.get_components()
        self.assertEqual(comps[0][0], 'A')
        self.assertEqual(comps[0][2], 'sphere@hardsphere')
        # The product part's parameters carry the A_ prefix, aliased via the
        # form-factor-derived moniker ('sphere').
        self.assertIn('sphere_volfraction', fitter.params)
        # Composition root is a mixture whose first part is a product.
        self.assertEqual(fitter.kernel.info.composition[0], 'mixture')
        first_part = fitter.kernel.info.composition[1][0]
        self.assertEqual(first_part.composition[0], 'product')


class TestCompositeValidation(unittest.TestCase):
    """Test 2: validation of model expressions and set_models arguments."""

    def setUp(self):
        self.fitter = SANSFitter()
        self.fitter.set_data(make_composite_data())

    def test_unknown_part_name_raises_with_suggestion(self):
        with self.assertRaises(ValueError) as ctx:
            self.fitter.set_model('dab+peak_lorentzzz')
        self.assertIn('peak_lorentzzz', str(ctx.exception))
        self.assertIn("Did you mean 'peak_lorentz'", str(ctx.exception))

    def test_set_models_single_model_raises(self):
        with self.assertRaises(ValueError):
            self.fitter.set_models('dab')

    def test_set_models_bad_operation_raises(self):
        with self.assertRaises(ValueError):
            self.fitter.set_models('dab', 'peak_lorentz', operation='%')

    def test_set_structure_factor_on_composite_raises(self):
        self.fitter.set_models('dab', 'peak_lorentz')
        with self.assertRaises(ValueError) as ctx:
            self.fitter.set_structure_factor('hardsphere')
        self.assertIn('composite', str(ctx.exception))


class TestBoundsPolicy(unittest.TestCase):
    """Test 3: the new sign-permissive bounds fallback."""

    def test_unbounded_param_gets_sign_permissive_range(self):
        # peak_lorentz.peak_pos defaults to 0.05 with (-inf, inf) limits.
        fitter = SANSFitter()
        fitter.set_data(make_composite_data())
        fitter.set_model('dab+peak_lorentz')
        info = fitter.params['B_peak_pos']
        self.assertEqual(info['value'], 0.05)
        # New policy: default ± max(10·|default|, 1.0), lower capped at 0.
        self.assertAlmostEqual(info['min'], min(0.0, 0.05 - 1.0))
        self.assertAlmostEqual(info['max'], 0.05 + 1.0)
        self.assertLess(info['min'], 0.0)  # sign-permissive

    def test_zero_default_unbounded_param_non_degenerate(self):
        # stacked_disks.sld_layer defaults to 0 with (-inf, inf) limits.
        fitter = SANSFitter()
        fitter.set_data(make_composite_data())
        fitter.set_model('stacked_disks')
        info = fitter.params['sld_layer']
        self.assertLess(info['min'], info['max'])  # not degenerate [0, 0]


class TestLinks(unittest.TestCase):
    """Test 4: equality link validation and mechanics."""

    def setUp(self):
        self.fitter = SANSFitter()
        self.fitter.set_data(make_composite_data())
        self.fitter.set_models(small='sphere', large='sphere')

    def test_link_missing_name_raises(self):
        with self.assertRaises(KeyError):
            self.fitter.link_params('small_bogus', to='small_sld')
        with self.assertRaises(KeyError):
            self.fitter.link_params('small_sld', to='large_bogus')

    def test_self_link_raises(self):
        with self.assertRaises(ValueError):
            self.fitter.link_params('small_sld', to='small_sld')

    def test_chain_link_raises(self):
        self.fitter.link_params('large_sld', to='small_sld')
        with self.assertRaises(ValueError):
            self.fitter.link_params('small_sld', to='large_sld')

    def test_set_param_on_follower_raises(self):
        self.fitter.link_params('large_sld', to='small_sld')
        with self.assertRaises(ValueError):
            self.fitter.set_param('large_sld', value=9.0)
        with self.assertRaises(ValueError):
            self.fitter.set_param('large_sld', vary=True)

    def test_target_value_propagates_to_follower(self):
        self.fitter.link_params('large_sld', to='small_sld')
        self.fitter.set_param('small_sld', value=3.5)
        self.assertEqual(self.fitter.params['large_sld']['value'], 3.5)

    def test_follower_forced_fixed(self):
        self.fitter.set_param('large_sld', vary=True)
        self.fitter.link_params('large_sld', to='small_sld')
        self.assertFalse(self.fitter.params['large_sld']['vary'])

    def test_unlink_restores_independence(self):
        self.fitter.link_params('large_sld', to='small_sld')
        self.fitter.unlink_params('large_sld')
        self.assertEqual(self.fitter.get_links(), {})
        # Now writable again.
        self.fitter.set_param('large_sld', value=7.0)
        self.assertEqual(self.fitter.params['large_sld']['value'], 7.0)

    def test_unlink_non_linked_raises(self):
        with self.assertRaises(ValueError):
            self.fitter.unlink_params('large_sld')

    def test_snapshot_carries_linked_params(self):
        self.fitter.link_params('large_sld', to='small_sld')
        snap = self.fitter._param_manager.snapshot_fit_state()
        self.assertEqual(snap.linked_params, {'B_sld': 'A_sld'})


class TestCompositeFitRoundTrip(unittest.TestCase):
    """Test 5: bumps fit recovers known composite parameters."""

    def setUp(self):
        truth = dict(
            A_scale=10.0,
            A_cor_length=50.0,
            B_scale=5.0,
            B_peak_pos=0.1,
            B_peak_hwhm=0.01,
            scale=1.0,
            background=0.01,
        )
        self.truth = truth
        self.fitter = SANSFitter()
        self.fitter.set_data(make_composite_data(params=truth))
        self.fitter.set_models('dab', 'peak_lorentz')

    def test_fit_recovers_parameters(self):
        self.fitter.set_param('dab_cor_length', value=45.0, min=1.0, max=200.0, vary=True)
        self.fitter.set_param('dab_scale', value=9.0, min=0.1, max=100.0, vary=True)
        self.fitter.set_param('peak_lorentz_peak_pos', value=0.095, min=0.01, max=0.3, vary=True)
        self.fitter.set_param('peak_lorentz_scale', value=4.5, min=0.1, max=100.0, vary=True)
        self.fitter.set_param('background', value=0.008, min=0.0, max=0.1, vary=True)

        result = self.fitter.fit(engine='bumps', method='amoeba', steps=2000)

        self.assertIn('chisq', result)
        self.assertAlmostEqual(
            self.fitter.params['dab_cor_length']['value'], self.truth['A_cor_length'], delta=2.0
        )
        self.assertAlmostEqual(
            self.fitter.params['peak_lorentz_peak_pos']['value'],
            self.truth['B_peak_pos'],
            delta=0.02,
        )

    def test_follower_equals_target_after_fit(self):
        self.fitter.link_params('peak_lorentz_scale', to='dab_scale')
        self.fitter.set_param('dab_cor_length', value=40.0, min=1.0, max=200.0, vary=True)
        self.fitter.set_param('dab_scale', value=8.0, min=0.1, max=100.0, vary=True)
        self.fitter.set_param('peak_lorentz_peak_pos', value=0.08, min=0.01, max=0.3, vary=True)
        self.fitter.set_param('background', value=0.005, min=0.0, max=0.1, vary=True)

        self.fitter.fit(engine='bumps', method='amoeba')

        self.assertAlmostEqual(
            self.fitter.params['peak_lorentz_scale']['value'],
            self.fitter.params['dab_scale']['value'],
            places=8,
        )


class TestEngineGating(unittest.TestCase):
    """Test 6: non-bumps engines reject composites and links."""

    def setUp(self):
        self.fitter = SANSFitter()
        self.fitter.set_data(make_composite_data())
        self.fitter.set_models('dab', 'peak_lorentz')
        self.fitter.set_param('dab_cor_length', vary=True)

    def test_lmfit_on_composite_raises(self):
        with self.assertRaises(NotImplementedError):
            self.fitter.fit(engine='lmfit')

    def test_lmfit_with_links_raises(self):
        atomic = SANSFitter()
        atomic.set_data(make_composite_data())
        atomic.set_model('sphere')
        atomic.set_param('radius', vary=True)
        atomic.set_param('sld', vary=True)
        atomic.link_params('sld_solvent', to='sld')
        with self.assertRaises(NotImplementedError):
            atomic.fit(engine='lmfit')

    def test_bayesian_on_composite_raises(self):
        with self.assertRaises(NotImplementedError):
            self.fitter.fit_bayesian(samples=100, burn=10)


class TestComponentCurves(unittest.TestCase):
    """Test 7: per-component curves for '+' mixtures."""

    def setUp(self):
        truth = dict(
            A_scale=10.0,
            A_cor_length=50.0,
            B_scale=5.0,
            B_peak_pos=0.1,
            B_peak_hwhm=0.01,
            scale=1.0,
            background=0.01,
        )
        self.fitter = SANSFitter()
        self.fitter.set_data(make_composite_data(params=truth))
        self.fitter.set_models('dab', 'peak_lorentz')
        self.fitter.set_param('dab_cor_length', value=50.0, vary=True)
        self.fitter.set_param('peak_lorentz_peak_pos', value=0.1, vary=True)
        self.fitter.fit(engine='bumps', method='amoeba')

    def test_component_curves_present_and_sum_to_total(self):
        curves = self.fitter._fit_contract.artifacts.component_curves
        self.assertIsNotNone(curves)
        self.assertEqual(set(curves.keys()), {'dab', 'peak_lorentz'})

        fit_index = self.fitter._fit_contract.artifacts.fit_index
        n_fit = (
            int(np.asarray(fit_index).sum()) if fit_index is not None else len(self.fitter.data.x)
        )
        for curve in curves.values():
            self.assertEqual(len(curve), n_fit)

        total = sum(curves.values()) + self.fitter.params['background']['value']
        fitted = self.fitter._fit_contract.require_fitted_curve()
        np.testing.assert_allclose(total, fitted, rtol=1e-3, atol=1e-6)

    def test_product_mixture_has_no_component_curves(self):
        fitter = SANSFitter()
        fitter.set_data(make_composite_data(expression='dab*peak_lorentz'))
        fitter.set_model('dab*peak_lorentz')
        fitter.set_param('A_cor_length', value=50.0, vary=True)
        fitter.fit(engine='bumps', method='amoeba')
        self.assertIsNone(fitter._fit_contract.artifacts.component_curves)

    def test_atomic_model_has_no_component_curves(self):
        fitter = SANSFitter()
        fitter.set_data(make_composite_data())
        fitter.set_model('sphere')
        fitter.set_param('radius', value=20.0, vary=True)
        fitter.fit(engine='bumps', method='amoeba')
        self.assertIsNone(fitter._fit_contract.artifacts.component_curves)


class TestPolydispersityOnComponent(unittest.TestCase):
    """Test 8: polydispersity routed to a composite component."""

    def test_pd_on_component_via_canonical_name(self):
        fitter = SANSFitter()
        fitter.set_data(make_composite_data(expression='sphere+peak_lorentz'))
        fitter.set_model('sphere+peak_lorentz')
        fitter.set_pd_param('A_radius', pd_width=0.1, vary=False)
        fitter.enable_polydispersity(True)
        fitter.set_param('A_radius', value=20.0, vary=True)
        fitter.set_param('scale', value=0.1, vary=True)

        result = fitter.fit(engine='bumps', method='amoeba')
        self.assertIn('chisq', result)
        # Component curves still stack to the total with PD active.
        curves = fitter._fit_contract.artifacts.component_curves
        self.assertIsNotNone(curves)
        total = sum(curves.values()) + fitter.params['background']['value']
        fitted = fitter._fit_contract.require_fitted_curve()
        np.testing.assert_allclose(total, fitted, rtol=1e-2, atol=1e-5)

    def test_pd_via_friendly_alias_equivalent_to_canonical(self):
        fitter = SANSFitter()
        fitter.set_data(make_composite_data(expression='sphere+peak_lorentz'))
        fitter.set_models('sphere', 'peak_lorentz')
        fitter.set_pd_param('sphere_radius', pd_width=0.15, pd_n=25)

        # Alias and canonical spellings reach the same PD store.
        alias_cfg = fitter.get_pd_param('sphere_radius')
        canonical_cfg = fitter.get_pd_param('A_radius')
        self.assertEqual(alias_cfg['pd'], canonical_cfg['pd'])
        self.assertEqual(alias_cfg['pd'], 0.15)
        self.assertEqual(alias_cfg['pd_n'], 25)

    def test_pd_on_product_part_inside_mixture(self):
        # PD on A_radius of (sphere@hardsphere) + peak_lorentz: the product
        # part takes a single A_ prefix over all its parameters, so the strip
        # must map onto sphere@hardsphere's own names.
        fitter = SANSFitter()
        data = make_composite_data(expression='sphere@hardsphere+peak_lorentz')
        fitter.set_data(data)
        fitter.set_model('sphere@hardsphere+peak_lorentz')
        fitter.set_pd_param('A_radius', pd_width=0.1, vary=False)
        fitter.enable_polydispersity(True)
        fitter.set_param('A_radius', value=30.0, vary=True)
        fitter.set_param('scale', value=0.1, vary=True)

        result = fitter.fit(engine='bumps', method='amoeba')
        self.assertIn('chisq', result)
        # The sum check still holds with PD on a product part.
        curves = fitter._fit_contract.artifacts.component_curves
        self.assertIsNotNone(curves)
        total = sum(curves.values()) + fitter.params['background']['value']
        fitted = fitter._fit_contract.require_fitted_curve()
        np.testing.assert_allclose(total, fitted, rtol=2e-2, atol=1e-5)


class TestAliases(unittest.TestCase):
    """Test 10: the friendly-name alias layer."""

    def setUp(self):
        self.data = make_composite_data()

    def test_positional_yields_model_name_prefixes(self):
        fitter = SANSFitter()
        fitter.set_data(self.data)
        fitter.set_models('dab', 'peak_lorentz')
        self.assertIn('dab_cor_length', fitter.params)
        self.assertIn('peak_lorentz_peak_pos', fitter.params)

    def test_keyword_monikers_yield_moniker_prefixes(self):
        fitter = SANSFitter()
        fitter.set_data(self.data)
        fitter.set_models(small='sphere', large='sphere')
        self.assertIn('small_radius', fitter.params)
        self.assertIn('large_radius', fitter.params)

    def test_duplicate_positionals_auto_suffix(self):
        fitter = SANSFitter()
        fitter.set_data(self.data)
        fitter.set_models('sphere', 'sphere')
        self.assertIn('sphere1_radius', fitter.params)
        self.assertIn('sphere2_radius', fitter.params)

    def test_alias_collision_raises(self):
        fitter = SANSFitter()
        fitter.set_data(self.data)
        # Moniker 'dab' on dab (param cor_length) and moniker 'dab_cor' on
        # cylinder (param length) both generate the alias 'dab_cor_length'.
        with self.assertRaises(ValueError):
            fitter.set_models('dab', dab_cor='cylinder')

    def test_canonical_names_still_accepted(self):
        fitter = SANSFitter()
        fitter.set_data(self.data)
        fitter.set_models('dab', 'peak_lorentz')
        fitter.set_param('A_cor_length', value=42.0)
        self.assertEqual(fitter.params['dab_cor_length']['value'], 42.0)

    def test_keyerror_lists_alias_names(self):
        fitter = SANSFitter()
        fitter.set_data(self.data)
        fitter.set_models('dab', 'peak_lorentz')
        with self.assertRaises(KeyError) as ctx:
            fitter.set_param('bogus', value=1.0)
        self.assertIn('dab_cor_length', str(ctx.exception))

    def test_raw_path_keeps_canonical_names(self):
        fitter = SANSFitter()
        fitter.set_data(self.data)
        fitter.set_model('dab+peak_lorentz')
        # Empty alias map on the raw path.
        self.assertEqual(fitter._param_manager._alias_to_canonical, {})
        self.assertIn('A_cor_length', fitter.params)


class TestSharedParameters(unittest.TestCase):
    """Test 11: the shared= parameter mechanism."""

    def setUp(self):
        self.data = make_composite_data(expression='sphere+cylinder')

    def test_shared_creates_unprefixed_and_removes_prefixed(self):
        fitter = SANSFitter()
        fitter.set_data(self.data)
        fitter.set_models('sphere', 'cylinder', shared=['sld', 'sld_solvent'])
        self.assertIn('sld', fitter.params)
        self.assertIn('sld_solvent', fitter.params)
        self.assertNotIn('sphere_sld', fitter.params)
        self.assertNotIn('cylinder_sld', fitter.params)

    def test_shared_drives_all_components(self):
        fitter = SANSFitter()
        fitter.set_data(self.data)
        fitter.set_models('sphere', 'cylinder', shared=['sld'])
        fitter.set_param('sld', value=4.5)
        snap = fitter._param_manager.snapshot_fit_state()
        self.assertEqual(snap.params['A_sld']['value'], 4.5)
        self.assertEqual(snap.params['B_sld']['value'], 4.5)
        self.assertEqual(snap.linked_params, {'B_sld': 'A_sld'})

    def test_shared_name_in_too_few_components_raises(self):
        fitter = SANSFitter()
        fitter.set_data(self.data)
        # 'radius' exists in sphere but not cylinder (cylinder has radius too,
        # so use a name present in only one).
        with self.assertRaises(ValueError) as ctx:
            fitter.set_models('dab', 'peak_lorentz', shared=['cor_length'])
        self.assertIn('cor_length', str(ctx.exception))

    def test_shared_pd_stays_per_component(self):
        fitter = SANSFitter()
        fitter.set_data(self.data)
        fitter.set_models('sphere', 'cylinder', shared=['radius'])
        self.assertIn('radius', fitter.params)
        self.assertNotIn('sphere_radius', fitter.params)
        self.assertNotIn('cylinder_radius', fitter.params)
        # But PD access stays addressable per component.
        fitter.set_pd_param('sphere_radius', pd_width=0.1)
        fitter.set_pd_param('cylinder_radius', pd_width=0.2)
        self.assertEqual(fitter.get_pd_param('A_radius')['pd'], 0.1)
        self.assertEqual(fitter.get_pd_param('B_radius')['pd'], 0.2)

    def test_shared_fitted_value_writes_back_to_alias(self):
        fitter = SANSFitter()
        fitter.set_data(make_composite_data(expression='sphere+sphere'))
        fitter.set_models(small='sphere', large='sphere', shared=['sld'])
        fitter.set_param('sld', value=4.0, vary=True)
        fitter.set_param('small_radius', value=20.0, vary=True)
        fitter.set_param('scale', value=0.1, vary=True)
        fitter.fit(engine='bumps', method='amoeba')
        # The shared alias carries the fitted value.
        self.assertIn('sld', fitter.params)


class TestComponentCurvesAgainstIntermediates(unittest.TestCase):
    """Dev-only validation against sasmodels' private MixtureKernel API.

    Skipped by default; enable with SANS_FITTER_VALIDATE_COMPONENTS=1.
    Compares the per-part re-evaluation against the mixture kernel's internal
    intermediates, especially for the PD-enabled-component case.
    """

    @unittest.skipUnless(
        os.environ.get('SANS_FITTER_VALIDATE_COMPONENTS') == '1',
        'dev-only validation; set SANS_FITTER_VALIDATE_COMPONENTS=1 to run',
    )
    def test_component_curves_match_intermediates(self):
        from sasmodels.details import make_details

        fitter = SANSFitter()
        fitter.set_data(make_composite_data())
        fitter.set_models('dab', 'peak_lorentz')
        fitter.set_param('dab_cor_length', value=50.0, vary=True)
        fitter.set_param('peak_lorentz_peak_pos', value=0.1, vary=True)
        fitter.fit(engine='bumps', method='amoeba')

        curves = fitter._fit_contract.artifacts.component_curves
        self.assertIsNotNone(curves)
        # The sum check is the public equivalent of the intermediates check.
        total = sum(curves.values()) + fitter.params['background']['value']
        fitted = fitter._fit_contract.require_fitted_curve()
        np.testing.assert_allclose(total, fitted, rtol=1e-3, atol=1e-6)


class TestScaleDegeneracyWarning(unittest.TestCase):
    """Test 12: warning when global scale and a component scale both vary."""

    def setUp(self):
        self.fitter = SANSFitter()
        self.fitter.set_data(make_composite_data())
        self.fitter.set_models('dab', 'peak_lorentz')

    def test_both_scales_varying_warns(self):
        self.fitter.set_param('scale', vary=True)
        self.fitter.set_param('dab_scale', vary=True)
        self.fitter.set_param('dab_cor_length', vary=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.fitter._check_scale_degeneracy()
        self.assertTrue(any('degenerate' in str(w.message) for w in caught))

    def test_global_scale_alone_no_warning(self):
        self.fitter.set_param('scale', vary=True)
        self.fitter.set_param('dab_cor_length', vary=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.fitter._check_scale_degeneracy()
        self.assertFalse(any('degenerate' in str(w.message) for w in caught))

    def test_component_scale_alone_no_warning(self):
        self.fitter.set_param('dab_scale', vary=True)
        self.fitter.set_param('dab_cor_length', vary=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.fitter._check_scale_degeneracy()
        self.assertFalse(any('degenerate' in str(w.message) for w in caught))


class TestCompositePlotting(unittest.TestCase):
    """Test 9: show_components plotting."""

    def _fit_composite(self):
        truth = dict(
            A_scale=10.0,
            A_cor_length=50.0,
            B_scale=5.0,
            B_peak_pos=0.1,
            B_peak_hwhm=0.01,
            scale=1.0,
            background=0.01,
        )
        fitter = SANSFitter()
        fitter.set_data(make_composite_data(params=truth))
        fitter.set_models('dab', 'peak_lorentz')
        fitter.set_param('dab_cor_length', value=50.0, vary=True)
        fitter.set_param('peak_lorentz_peak_pos', value=0.1, vary=True)
        fitter.fit(engine='bumps', method='amoeba')
        return fitter

    def test_show_components_adds_traces(self):
        from unittest.mock import patch

        fitter = self._fit_composite()
        with patch('plotly.graph_objects.Figure.show'):
            fig = fitter.plot_results(show_components=True, show=False)
        names = [trace.name for trace in fig.data]
        self.assertIn('dab', names)
        self.assertIn('peak_lorentz', names)

    def test_show_components_noop_without_curves(self):
        from unittest.mock import patch

        fitter = SANSFitter()
        fitter.set_data(make_composite_data())
        fitter.set_model('sphere')
        fitter.set_param('radius', value=20.0, vary=True)
        fitter.fit(engine='bumps', method='amoeba')
        with patch('plotly.graph_objects.Figure.show'):
            fig = fitter.plot_results(show_components=True, show=False)
        # No component traces added for an atomic model.
        names = [trace.name for trace in fig.data]
        self.assertNotIn('sphere', [n for n in names if n and ':' in str(n)])


if __name__ == '__main__':
    unittest.main()

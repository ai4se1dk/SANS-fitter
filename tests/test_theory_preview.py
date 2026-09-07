"""Integration tests for the theory preview API: calculate/plot_model/compare."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
from sasdata.dataloader.data_info import Data1D
from sasmodels.data import empty_data1D
from sasmodels.direct_model import DirectModel

from sans_fitter import SANSFitter, examples
from sans_fitter.data.loader import normalize_sans_data
from sans_fitter.fitting.bumps_engine import _build_bumps_problem
from sans_fitter.plotting import PREVIEW_MODEL_TRACE_NAME, PREVIEW_TITLE_PREFIX


def make_sphere_fitter(data=None, **simulate_kwargs):
    """A sphere fitter on simulated data with radius free."""
    fitter = SANSFitter()
    fitter.set_data(data if data is not None else examples.simulate('sphere', **simulate_kwargs))
    fitter.set_model('sphere')
    fitter.set_param('radius', value=40.0, min=1.0, max=100.0, vary=True)
    return fitter


class TestCalculate(unittest.TestCase):
    def setUp(self):
        self.fitter = make_sphere_fitter(radius=45, npoints=25, seed=0)

    def test_matches_direct_model_on_the_data_grid(self):
        result = self.fitter.calculate()
        expected = DirectModel(self.fitter.data, self.fitter.kernel)(
            **{name: info['value'] for name, info in self.fitter.params.items()}
        )

        self.assertEqual(len(result), len(self.fitter.data.x))
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_allclose(result, expected)

    def test_excluded_points_are_nan(self):
        self.fitter.set_q_range(qmin=0.05)
        result = self.fitter.calculate()

        excluded = np.asarray(self.fitter.data.x) < 0.05
        np.testing.assert_array_equal(np.isnan(result), excluded)

    def test_explicit_q_grid(self):
        q = np.geomspace(0.01, 0.4, 12)
        self.assertEqual(len(self.fitter.calculate(q=q)), 12)

    def test_explicit_q_grid_is_unsmeared_by_default(self):
        fitter = make_sphere_fitter(radius=45, npoints=25, seed=0, dq=0.1)
        on_data = fitter.calculate()
        on_same_q = fitter.calculate(q=np.asarray(fitter.data.x))

        # The data path applies the dataset's dQ column; the explicit grid
        # does not, so the two must differ.
        self.assertFalse(np.allclose(on_data, on_same_q))

    def test_dq_is_a_relative_width(self):
        q = np.geomspace(0.01, 0.4, 12)
        values = {name: info['value'] for name, info in self.fitter.params.items()}
        expected = DirectModel(empty_data1D(q, resolution=0.1), self.fitter.kernel)(**values)

        smeared = self.fitter.calculate(q=q, dq=0.1)
        np.testing.assert_allclose(smeared, expected)
        self.assertFalse(np.allclose(smeared, self.fitter.calculate(q=q)))

    def test_explicit_q_works_without_data(self):
        fitter = SANSFitter()
        fitter.set_model('sphere')
        self.assertEqual(len(fitter.calculate(q=np.geomspace(0.01, 0.4, 8))), 8)

    def test_without_data_and_without_q_raises(self):
        fitter = SANSFitter()
        fitter.set_model('sphere')
        with self.assertRaises(ValueError):
            fitter.calculate()

    def test_without_model_raises(self):
        fitter = SANSFitter()
        fitter.set_data(examples.simulate('sphere', npoints=10))
        with self.assertRaises(ValueError):
            fitter.calculate()

    def test_dq_without_q_raises(self):
        with self.assertRaises(ValueError):
            self.fitter.calculate(dq=0.05)

    def test_invalid_dq_raises(self):
        with self.assertRaises(ValueError):
            self.fitter.calculate(q=np.geomspace(0.01, 0.4, 8), dq=-0.1)

    def test_polydispersity_changes_the_curve(self):
        monodisperse = self.fitter.calculate()

        self.fitter.set_pd_param('radius', pd_width=0.2)
        self.fitter.enable_polydispersity(True)
        polydisperse = self.fitter.calculate()
        self.assertFalse(np.allclose(monodisperse, polydisperse))

        # Disabling globally is the same switch the fitting engines honour.
        self.fitter.enable_polydispersity(False)
        np.testing.assert_allclose(self.fitter.calculate(), monodisperse)

    def test_parameter_links_are_applied(self):
        fitter = SANSFitter()
        fitter.set_data(examples.simulate('core_shell_sphere', npoints=20, seed=0))
        fitter.set_model('core_shell_sphere')
        fitter.set_param('sld_core', value=2.0)
        fitter.link_params('sld_shell', to='sld_core')

        linked = fitter.calculate()
        fitter.set_param('sld_core', value=5.0)
        self.assertFalse(np.allclose(linked, fitter.calculate()))

        expected = DirectModel(fitter.data, fitter.kernel)(
            **{**{n: i['value'] for n, i in fitter.params.items()}, 'sld_shell': 5.0}
        )
        np.testing.assert_allclose(fitter.calculate(), expected)

    def test_shared_parameters_drive_every_component(self):
        fitter = SANSFitter()
        fitter.set_data(examples.simulate('sphere', npoints=20, seed=0))
        fitter.set_models(small='sphere', large='sphere', shared=['sld'])
        fitter.set_param('sld', value=3.5)
        fitter.set_param('small_radius', value=20.0)
        fitter.set_param('large_radius', value=60.0)

        canonical = fitter._param_manager.get_canonical_param_values()
        self.assertEqual(canonical['A_sld'], 3.5)
        self.assertEqual(canonical['B_sld'], 3.5)
        expected = DirectModel(fitter.data, fitter.kernel)(**canonical)
        np.testing.assert_allclose(fitter.calculate(), expected, rtol=1e-5)

    def test_structure_factor_radius_link(self):
        fitter = SANSFitter()
        fitter.set_data(examples.simulate('sphere', npoints=20, seed=0))
        fitter.set_model('sphere')
        fitter.set_structure_factor('hardsphere', radius_effective_mode='link_radius')
        fitter.set_param('radius', value=60.0)

        linked = fitter.calculate()
        fitter.set_param('radius_effective', value=20.0)
        # radius_effective follows radius, so writing it must not matter.
        np.testing.assert_allclose(fitter.calculate(), linked)


class TestPlotModel(unittest.TestCase):
    def setUp(self):
        self.fitter = make_sphere_fitter(radius=45, npoints=25, seed=0)

    def test_preview_labels_and_untouched_fit_state(self):
        fig = self.fitter.plot_model(show=False)

        self.assertIn(PREVIEW_TITLE_PREFIX, fig.layout.title.text)
        self.assertIn(PREVIEW_MODEL_TRACE_NAME, [trace.name for trace in fig.data])
        self.assertIsNone(self.fitter.fit_result)
        self.assertIsNone(self.fitter._fit_contract)

    def test_plot_results_still_reports_no_fit(self):
        self.fitter.plot_model(show=False)
        with patch('builtins.print') as mock_print:
            self.fitter.plot_results(show=False)
        printed = ' '.join(str(call) for call in mock_print.call_args_list)
        self.assertIn('No fit results available', printed)

    def test_matches_the_fit_curve_after_fitting(self):
        self.fitter.fit(engine='bumps', method='amoeba')
        preview = self.fitter.plot_model(show=False)
        fitted = self.fitter.plot_results(show=False)

        np.testing.assert_allclose(preview.data[1].y, fitted.data[1].y, rtol=1e-6)

    def test_chisq_matches_the_bumps_initial_value(self):
        problem, _ = _build_bumps_problem(
            self.fitter.data, self.fitter.kernel, self.fitter._param_manager.snapshot_fit_state()
        )
        fig = self.fitter.plot_model(show=False)
        self.assertIn(f'χ² = {problem.chisq():.4f}', fig.layout.title.text)

    def test_data_without_uncertainties_reports_no_chisq(self):
        q = np.geomspace(0.01, 0.3, 15)
        data = normalize_sans_data(Data1D(x=q, y=np.exp(-q * 10) + 0.1))
        fitter = make_sphere_fitter(data=data)

        fig = fitter.plot_model(show=False)

        self.assertIn('χ² n/a (no dI)', fig.layout.title.text)
        self.assertIsNone(fitter.data.dy)

    def test_zero_uncertainties_report_no_chisq(self):
        fitter = make_sphere_fitter(npoints=15, noise=0, seed=0)
        self.assertIn('χ² n/a (no dI)', fitter.plot_model(show=False).layout.title.text)

    def test_chisq_stays_finite_when_free_parameters_outnumber_points(self):
        self.fitter.set_q_range(qmin=self.fitter.data.x[-2])  # two fitted points
        self.fitter.set_param('scale', vary=True)
        self.fitter.set_param('background', vary=True)
        self.assertIn('χ² = ', self.fitter.plot_model(show=False).layout.title.text)

    def test_components_overlay_before_fitting(self):
        fitter = SANSFitter()
        fitter.set_data(examples.simulate('dab+peak_lorentz', npoints=25, seed=0))
        fitter.set_models('dab', 'peak_lorentz')

        names = [trace.name for trace in fitter.plot_model(show_components=True, show=False).data]
        self.assertIn('dab', names)
        self.assertIn('peak_lorentz', names)

    def test_without_data_raises(self):
        fitter = SANSFitter()
        fitter.set_model('sphere')
        with self.assertRaises(ValueError):
            fitter.plot_model(show=False)

    def test_without_model_raises(self):
        fitter = SANSFitter()
        fitter.set_data(examples.simulate('sphere', npoints=10))
        with self.assertRaises(ValueError):
            fitter.plot_model(show=False)


class TestCompare(unittest.TestCase):
    def setUp(self):
        self.fitter = make_sphere_fitter(radius=45, npoints=25, seed=0)

    def test_sweep_form(self):
        fig = self.fitter.compare(radius=[20, 30, 40], show=False)
        names = [trace.name for trace in fig.data]
        self.assertEqual(names, ['Experimental Data', 'radius = 20', 'radius = 30', 'radius = 40'])

    def test_cases_form(self):
        fig = self.fitter.compare({'monodisperse': {}, '20% PD': {'radius_pd': 0.2}}, show=False)
        self.assertEqual(
            [trace.name for trace in fig.data],
            ['Experimental Data', 'monodisperse', '20% PD'],
        )
        self.assertFalse(np.allclose(fig.data[1].y, fig.data[2].y))

    def test_curves_align_with_the_plotted_q_vector_after_q_range(self):
        self.fitter.set_q_range(qmin=0.05)
        fig = self.fitter.compare(radius=[20, 40], show=False)

        excluded = np.asarray(self.fitter.data.x) < 0.05
        for trace in fig.data[1:]:
            self.assertEqual(len(trace.y), len(trace.x))
            self.assertEqual(len(trace.y), len(self.fitter.data.x))
            np.testing.assert_array_equal(np.isnan(np.asarray(trace.y, dtype=float)), excluded)

    def test_show_data_false_drops_the_data_trace(self):
        fig = self.fitter.compare(radius=[20, 40], show_data=False, show=False)
        self.assertEqual([trace.name for trace in fig.data], ['radius = 20', 'radius = 40'])

    def test_explicit_q_grid_with_data(self):
        q = np.geomspace(0.01, 0.4, 12)
        fig = self.fitter.compare(radius=[20, 40], q=q, show=False)

        self.assertEqual(len(fig.data[0].x), len(self.fitter.data.x))
        self.assertEqual(len(fig.data[1].x), 12)

    def test_parameters_are_unchanged(self):
        before = copy.deepcopy(self.fitter.params)
        self.fitter.compare(radius=[20, 30], show=False)
        self.assertEqual(self.fitter.params, before)

    def test_without_data_and_without_q_raises(self):
        fitter = SANSFitter()
        fitter.set_model('sphere')
        with self.assertRaises(ValueError):
            fitter.compare(radius=[20, 40], show=False)

    def test_without_model_raises(self):
        fitter = SANSFitter()
        fitter.set_data(examples.simulate('sphere', npoints=10))
        with self.assertRaises(ValueError):
            fitter.compare(radius=[20, 40], show=False)

    def test_two_sweeps_raise(self):
        with self.assertRaises(ValueError):
            self.fitter.compare(radius=[20, 40], scale=[1.0, 2.0], show=False)

    def test_cases_plus_sweep_raises(self):
        with self.assertRaises(ValueError):
            self.fitter.compare({'a': {}}, radius=[20, 40], show=False)

    def test_nothing_to_compare_raises(self):
        with self.assertRaises(ValueError):
            self.fitter.compare(show=False)

    def test_unknown_parameter_raises_keyerror(self):
        with self.assertRaises(KeyError):
            self.fitter.compare({'a': {'nonexistent': 1.0}}, show=False)

    def test_unknown_pd_parameter_raises_keyerror(self):
        with self.assertRaises(KeyError):
            self.fitter.compare({'a': {'scale_pd': 0.1}}, show=False)

    def test_overriding_a_link_follower_raises(self):
        fitter = SANSFitter()
        fitter.set_data(examples.simulate('core_shell_sphere', npoints=20, seed=0))
        fitter.set_model('core_shell_sphere')
        fitter.link_params('sld_shell', to='sld_core')

        with self.assertRaises(ValueError):
            fitter.compare({'a': {'sld_shell': 3.0}}, show=False)

    def test_alias_and_canonical_overrides_agree(self):
        fitter = SANSFitter()
        fitter.set_data(examples.simulate('sphere', npoints=20, seed=0))
        fitter.set_models(small='sphere', large='sphere')

        alias = fitter.compare({'a': {'small_radius': 25.0}}, show=False)
        canonical = fitter.compare({'a': {'A_radius': 25.0}}, show=False)
        np.testing.assert_allclose(alias.data[1].y, canonical.data[1].y)


class TestPreviewDoesNotPrintFromCalculate(unittest.TestCase):
    def test_calculate_is_silent(self):
        fitter = make_sphere_fitter(npoints=10, seed=0)
        with patch('builtins.print') as mock_print:
            fitter.calculate()
        mock_print.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)

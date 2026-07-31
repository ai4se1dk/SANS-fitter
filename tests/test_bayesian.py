import os
import tempfile
import unittest

import numpy as np
import plotly.graph_objects as go

from sans_fitter import PosteriorSummary, SANSFitter
from sans_fitter.plotting import (
    plot_param_correlations,
    plot_param_distribution,
    plot_posterior_pairs,
    plot_posterior_predictive,
    plot_trace,
)
from sans_fitter.results import FitArtifacts, FitResultContract
from tests.helpers import create_decay_data_file

# Keep the DREAM run tiny so the whole module stays fast.
TINY_DREAM = {'samples': 300, 'burn': 10, 'thin': 1, 'pop': 4}


def make_synthetic_posterior(n_samples=500, with_chains=True, with_diagnostics=True):
    rng = np.random.default_rng(42)
    labels = ['radius', 'scale']
    samples = np.column_stack(
        [
            rng.normal(20.0, 1.0, n_samples),
            rng.normal(0.1, 0.01, n_samples),
        ]
    )
    percentiles = np.percentile(samples, [16, 84, 2.5, 97.5], axis=0)
    chains = None
    if with_chains:
        chains = samples[: (n_samples // 4) * 4].reshape(-1, 4, 2)
    diagnostics = None
    if with_diagnostics:
        diagnostics = {name: {'r_hat': 1.01, 'ess': 400.0} for name in labels}
    return PosteriorSummary(
        labels=labels,
        samples=samples,
        logp=rng.normal(-10.0, 1.0, n_samples),
        chains=chains,
        best={name: float(np.median(samples[:, i])) for i, name in enumerate(labels)},
        mean={name: float(samples[:, i].mean()) for i, name in enumerate(labels)},
        median={name: float(np.median(samples[:, i])) for i, name in enumerate(labels)},
        std={name: float(samples[:, i].std()) for i, name in enumerate(labels)},
        ci_68={
            name: (float(percentiles[0, i]), float(percentiles[1, i]))
            for i, name in enumerate(labels)
        },
        ci_95={
            name: (float(percentiles[2, i]), float(percentiles[3, i]))
            for i, name in enumerate(labels)
        },
        diagnostics=diagnostics,
    )


class FakeData:
    def __init__(self, n=20):
        self.x = np.logspace(-2, 0, n)
        self.y = 0.1 / (1 + self.x**2) + 0.01
        self.dy = self.y * 0.1
        self.dx = np.zeros(n)


class TestPosteriorSummary(unittest.TestCase):
    def setUp(self):
        self.posterior = make_synthetic_posterior()

    def test_shape_properties(self):
        self.assertEqual(self.posterior.n_samples, 500)
        self.assertEqual(self.posterior.n_params, 2)

    def test_index_of_unknown_param_raises(self):
        with self.assertRaises(KeyError):
            self.posterior.index_of('nonexistent')

    def test_format_summary_lists_parameters(self):
        text = self.posterior.format_summary()
        self.assertIn('radius', text)
        self.assertIn('scale', text)
        self.assertIn('95% CI', text)
        self.assertIn('R-hat', text)

    def test_format_summary_omits_diagnostics_when_absent(self):
        posterior = make_synthetic_posterior(with_diagnostics=False)
        self.assertNotIn('R-hat', posterior.format_summary())

    def test_save_posterior_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'posterior.csv')
            self.posterior.save_posterior_csv(path)
            with open(path) as f:
                lines = f.readlines()
            self.assertEqual(lines[0].strip(), 'radius,scale,logp')
            self.assertEqual(len(lines), 1 + self.posterior.n_samples)

    def test_require_posterior_raises_without_posterior(self):
        contract = FitResultContract(
            engine='bumps', method='amoeba', chisq=1.0, parameters={}, artifacts=FitArtifacts()
        )
        with self.assertRaises(ValueError):
            contract.require_posterior()

    def test_require_posterior_returns_posterior(self):
        contract = FitResultContract(
            engine='bumps',
            method='dream',
            chisq=1.0,
            parameters={},
            artifacts=FitArtifacts(posterior=self.posterior),
        )
        self.assertIs(contract.require_posterior(), self.posterior)


class TestPosteriorPlots(unittest.TestCase):
    """Plot primitives over a synthetic posterior (no sampling)."""

    def setUp(self):
        self.posterior = make_synthetic_posterior()

    def test_plot_posterior_pairs(self):
        fig = plot_posterior_pairs(self.posterior, show=False)
        self.assertIsInstance(fig, go.Figure)
        # 2 diagonal histograms + 1 scatter + 1 contour for the lower panel
        self.assertEqual(len(fig.data), 4)

    def test_plot_posterior_pairs_without_contours(self):
        fig = plot_posterior_pairs(self.posterior, show_contours=False, show=False)
        self.assertEqual(len(fig.data), 3)

    def test_plot_posterior_pairs_single_param_raises(self):
        with self.assertRaises(ValueError) as ctx:
            plot_posterior_pairs(self.posterior, params=['radius'], show=False)
        self.assertIn('plot_param_distribution', str(ctx.exception))

    def test_plot_param_distribution(self):
        fig = plot_param_distribution(self.posterior, 'radius', show=False)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1)
        self.assertEqual(fig.data[0].type, 'histogram')

    def test_plot_param_distribution_unknown_param_raises(self):
        with self.assertRaises(KeyError):
            plot_param_distribution(self.posterior, 'nonexistent', show=False)

    def test_plot_posterior_predictive_band(self):
        data = FakeData()

        def model_eval(pars):
            return pars['scale'] / (1 + (data.x * pars['radius'] / 20.0) ** 2)

        fig = plot_posterior_predictive(
            data, self.posterior, model_eval, style='band', n_draws=10, show=False
        )
        names = [trace.name for trace in fig.data]
        self.assertIn('95% credible interval', names)
        self.assertIn('Best posterior sample', names)
        self.assertIn('Measured', names)

    def test_plot_posterior_predictive_draws(self):
        data = FakeData()

        def model_eval(pars):
            return np.full_like(data.x, pars['scale'])

        fig = plot_posterior_predictive(
            data, self.posterior, model_eval, style='draws', n_draws=5, show=False
        )
        # 5 draw curves + best + measured
        self.assertEqual(len(fig.data), 7)

    def test_plot_posterior_predictive_invalid_style_raises(self):
        with self.assertRaises(ValueError):
            plot_posterior_predictive(
                FakeData(), self.posterior, lambda p: np.zeros(20), style='bogus', show=False
            )

    def test_plot_posterior_predictive_no_data_raises(self):
        with self.assertRaises(ValueError):
            plot_posterior_predictive(None, self.posterior, lambda p: np.zeros(20), show=False)

    def test_plot_param_correlations(self):
        fig = plot_param_correlations(self.posterior, show=False)
        self.assertIsInstance(fig, go.Figure)
        z = np.asarray(fig.data[0].z, dtype=float)
        # Upper triangle and diagonal are masked; lower triangle is real.
        self.assertTrue(np.isnan(z[0, 0]) and np.isnan(z[0, 1]) and np.isnan(z[1, 1]))
        self.assertFalse(np.isnan(z[1, 0]))

    def test_plot_trace_with_chains(self):
        fig = plot_trace(self.posterior, show=False)
        # 4 chains per parameter x 2 parameters
        self.assertEqual(len(fig.data), 8)
        self.assertNotIn('combined', fig.layout.title.text)

    def test_plot_trace_combined_fallback(self):
        posterior = make_synthetic_posterior(with_chains=False)
        fig = plot_trace(posterior, show=False)
        self.assertEqual(len(fig.data), 2)
        self.assertIn('combined', fig.layout.title.text)


class TestFitBayesian(unittest.TestCase):
    """End-to-end DREAM fit through the public API (one shared tiny run)."""

    fitter = None

    @classmethod
    def setUpClass(cls):
        cls.data_file = create_decay_data_file()
        cls.fitter = SANSFitter()
        cls.fitter.load_data(cls.data_file)
        cls.fitter.set_model('sphere')
        cls.fitter.set_param('radius', value=20.0, min=10.0, max=30.0, vary=True)
        cls.fitter.set_param('scale', value=0.1, min=0.01, max=1.0, vary=True)
        cls.fitter.set_param('background', value=0.01, min=0, max=0.1, vary=False)
        cls.fitter.set_param('sld', value=2.0, vary=False)
        cls.fitter.set_param('sld_solvent', value=3.0, vary=False)
        cls.result = cls.fitter.fit_bayesian(**TINY_DREAM)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.data_file):
            os.unlink(cls.data_file)

    def test_result_has_legacy_shape(self):
        self.assertEqual(self.result['engine'], 'bumps')
        self.assertEqual(self.result['method'], 'dream')
        self.assertIn('chisq', self.result)
        self.assertIn('radius', self.result['parameters'])
        self.assertIn('scale', self.result['parameters'])
        for info in self.result['parameters'].values():
            self.assertIn('value', info)
            self.assertIn('stderr', info)
            self.assertIn('formatted', info)

    def test_posterior_populated(self):
        posterior = self.fitter.get_posterior()
        problem = self.fitter._fit_contract.artifacts.runtime_handle
        self.assertEqual(posterior.labels, list(problem.labels()))
        self.assertEqual(posterior.samples.shape[1], len(posterior.labels))
        self.assertGreaterEqual(posterior.samples.shape[0], 2)
        self.assertEqual(posterior.logp.shape, (posterior.n_samples,))

    def test_ci_95_brackets_median(self):
        posterior = self.fitter.get_posterior()
        for name in posterior.labels:
            lo, hi = posterior.ci_95[name]
            self.assertLessEqual(lo, posterior.median[name])
            self.assertGreaterEqual(hi, posterior.median[name])

    def test_map_curve_tracks_data(self):
        """Point-estimate re-evaluation: the fitted curve must track the data."""
        contract = self.fitter._fit_contract
        curve = contract.require_fitted_curve()
        data = self.fitter.data
        residuals = (np.asarray(data.y) - curve) / np.asarray(data.dy)
        # DREAM's best point on this fixture fits well within a few sigma.
        self.assertLess(np.abs(residuals).mean(), 5.0)
        self.assertTrue(np.isfinite(contract.chisq))

    def test_format_summary_runs(self):
        text = self.fitter.get_posterior().format_summary()
        self.assertIn('radius', text)

    def test_all_posterior_plots_return_figures(self):
        self.assertIsInstance(self.fitter.plot_posterior_pairs(show=False), go.Figure)
        self.assertIsInstance(self.fitter.plot_param_distribution('radius', show=False), go.Figure)
        self.assertIsInstance(
            self.fitter.plot_posterior_predictive(n_draws=5, show=False), go.Figure
        )
        self.assertIsInstance(self.fitter.plot_param_correlations(show=False), go.Figure)
        self.assertIsInstance(self.fitter.plot_trace(show=False), go.Figure)

    def test_plot_results_after_bayesian_fit(self):
        fig = self.fitter.plot_results(show=False)
        self.assertIsInstance(fig, go.Figure)

    def test_save_results_includes_credible_intervals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'results.csv')
            self.fitter.save_results(path)
            with open(path) as f:
                content = f.read()
            self.assertIn('Posterior credible intervals', content)
            self.assertIn('95% CI', content)


class TestFitBayesianErrors(unittest.TestCase):
    def test_fit_bayesian_without_data_raises(self):
        with self.assertRaises(ValueError):
            SANSFitter().fit_bayesian()

    def test_fit_bayesian_without_model_raises(self):
        fitter = SANSFitter()
        data_file = create_decay_data_file()
        try:
            fitter.load_data(data_file)
            with self.assertRaises(ValueError):
                fitter.fit_bayesian()
        finally:
            os.unlink(data_file)

    def test_fit_bayesian_without_varying_params_raises(self):
        fitter = SANSFitter()
        data_file = create_decay_data_file()
        try:
            fitter.load_data(data_file)
            fitter.set_model('sphere')
            for name in fitter.params:
                fitter.set_param(name, vary=False)
            with self.assertRaises(ValueError) as ctx:
                fitter.fit_bayesian(**TINY_DREAM)
            self.assertIn('varying parameter', str(ctx.exception))
        finally:
            os.unlink(data_file)

    def test_posterior_plots_without_fit_raise(self):
        fitter = SANSFitter()
        with self.assertRaises(ValueError):
            fitter.plot_posterior_pairs(show=False)
        with self.assertRaises(ValueError):
            fitter.plot_trace(show=False)

    def test_posterior_plots_after_non_bayesian_fit_raise(self):
        fitter = SANSFitter()
        data_file = create_decay_data_file()
        try:
            fitter.load_data(data_file)
            fitter.set_model('sphere')
            fitter.set_param('radius', value=20.0, min=10.0, max=30.0, vary=True)
            fitter.set_param('scale', value=0.1, min=0.01, max=1.0, vary=True)
            fitter.fit(engine='bumps', method='amoeba')
            with self.assertRaises(ValueError) as ctx:
                fitter.plot_posterior_pairs(show=False)
            self.assertIn('fit_bayesian', str(ctx.exception))
        finally:
            os.unlink(data_file)


if __name__ == '__main__':
    unittest.main(verbosity=2)

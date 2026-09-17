"""Fit-quality reporting (issue #77): the chi-squared vocabulary, convergence,
covariance, bound detection and the FitReport renderers."""

import json
import math
import warnings

import numpy as np
import pytest

from sans_fitter import SANSFitter, examples
from sans_fitter.fitting.base import (
    at_bound,
    correlation_matrix,
    normalize_message,
    reduced_chisq,
    validate_covariance,
)
from sans_fitter.fitting.bumps_engine import _build_bumps_problem, _configured_budget
from sans_fitter.report import FitReport, escape_markdown_cell, json_safe

FIXED = {'sld': 4.0, 'sld_solvent': 1.0, 'scale': 1.0, 'background': 0.001}


def sphere_fitter(npoints=40, seed=0, radius=50.0, noise=0.02, **kwargs):
    """A sphere fitter on simulated data with the radius free."""
    fitter = SANSFitter()
    fitter.set_data(
        examples.simulate('sphere', npoints=npoints, seed=seed, noise=noise, radius=radius, **FIXED)
    )
    fitter.set_model('sphere')
    fitter.set_param('radius', value=radius * 0.9, min=10.0, max=200.0, vary=True)
    for name, value in FIXED.items():
        fitter.set_param(name, value=value, vary=False)
    for name, value in kwargs.items():
        fitter.set_param(name, **value)
    return fitter


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestReducedChisq:
    def test_divides_by_dof(self):
        assert reduced_chisq(100.0, 50) == pytest.approx(2.0)

    @pytest.mark.parametrize('dof', [0, -1, -10])
    def test_no_degrees_of_freedom_is_not_a_number(self, dof):
        assert math.isnan(reduced_chisq(100.0, dof))

    def test_propagates_an_unavailable_chisq(self):
        assert math.isnan(reduced_chisq(float('nan'), 10))


class TestCorrelationMatrix:
    def test_unit_diagonal_and_symmetry(self):
        cov = np.array([[4.0, 1.0, 0.0], [1.0, 9.0, -3.0], [0.0, -3.0, 16.0]])
        corr = correlation_matrix(cov)
        np.testing.assert_allclose(np.diag(corr), 1.0)
        np.testing.assert_allclose(corr, corr.T)

    def test_reproduces_a_known_coefficient(self):
        # rho = cov_01 / (sigma_0 * sigma_1) = 1 / (2 * 3)
        corr = correlation_matrix(np.array([[4.0, 1.0], [1.0, 9.0]]))
        assert corr[0, 1] == pytest.approx(1.0 / 6.0)

    def test_bumps_own_helper_would_fail_this(self):
        """Guards against anyone replacing the helper with bumps.lsqerror.corr.

        That function's Dinv is 1-D, so its double np.dot contracts both axes and
        returns a scalar despite a correct docstring.
        """
        corr = correlation_matrix(np.array([[4.0, 1.0], [1.0, 9.0]]))
        assert corr.shape == (2, 2)

    def test_zero_variance_row_is_nan_not_inf(self):
        corr = correlation_matrix(np.array([[4.0, 0.0], [0.0, 0.0]]))
        assert np.isnan(corr[1, 1])
        assert np.isnan(corr[0, 1])
        assert not np.isinf(corr).any()

    def test_single_parameter(self):
        np.testing.assert_allclose(correlation_matrix(np.array([[2.5]])), [[1.0]])


class TestAtBound:
    def test_exact_hit(self):
        assert at_bound(100.0, 100.0)

    def test_within_relative_tolerance(self):
        assert at_bound(100.0 - 1e-3, 100.0)

    def test_zero_bound_hit_exactly(self):
        assert at_bound(0.0, 0.0)

    def test_ordinary_small_scale_is_not_on_its_lower_bound(self):
        """The rule a span fraction got wrong: 1e-3 is a normal SANS scale."""
        assert not at_bound(1e-3, 1e-5)

    def test_infinite_bound_is_never_hit(self):
        assert not at_bound(1e300, float('inf'))
        assert not at_bound(-1e300, float('-inf'))

    def test_far_from_bound(self):
        assert not at_bound(50.0, 100.0)


class TestValidateCovariance:
    def test_accepts_an_aligned_matrix(self):
        matrix = validate_covariance([[1.0, 0.0], [0.0, 2.0]], ['a', 'b'])
        assert matrix.shape == (2, 2)

    def test_rejects_a_shape_mismatch_naming_both_shapes(self):
        with pytest.raises(ValueError, match=r'\(3, 3\)'):
            validate_covariance(np.eye(3), ['a', 'b'])

    def test_rejects_a_scalar(self):
        with pytest.raises(ValueError):
            validate_covariance(np.array(1.0), ['a'])

    def test_rejects_infinities(self):
        with pytest.raises(ValueError, match='infinite'):
            validate_covariance([[float('inf')]], ['a'])

    def test_nan_is_allowed(self):
        matrix = validate_covariance([[float('nan')]], ['a'])
        assert np.isnan(matrix[0, 0])


class TestNormalizeMessage:
    def test_collapses_a_wrapped_scipy_message(self):
        assert normalize_message('both actual\n  and predicted') == 'both actual and predicted'

    def test_none_becomes_empty(self):
        assert normalize_message(None) == ''


# ---------------------------------------------------------------------------
# The chi-squared vocabulary
# ---------------------------------------------------------------------------


class TestChisqVocabulary:
    def test_bumps_reduced_matches_the_engines_own_value(self):
        fitter = sphere_fitter()
        result = fitter.fit(engine='bumps', method='amoeba')
        problem = fitter._fitted_model
        assert result['dof'] > 0
        assert result['reduced_chisq'] == pytest.approx(problem.chisq(), rel=1e-9)
        assert result['chisq'] == pytest.approx(result['reduced_chisq'] * result['dof'])

    def test_dof_is_points_minus_free_on_every_engine(self):
        for engine, method in (('bumps', 'amoeba'), ('lmfit', 'leastsq')):
            result = sphere_fitter().fit(engine=engine, method=method)
            assert result['dof'] == result['n_points'] - result['n_free']
            assert result['n_free'] == 1

    def test_n_points_honours_the_q_range(self):
        fitter = sphere_fitter(npoints=40)
        fitter.set_q_range(qmin=float(fitter.data.x[10]))
        result = fitter.fit(engine='bumps', method='amoeba')
        assert result['n_points'] == 30

    def test_n_points_honours_a_masked_row(self):
        data = examples.simulate('sphere', npoints=40, seed=1, noise=0.02, radius=50.0, **FIXED)
        data.y[5] = np.nan
        fitter = SANSFitter()
        fitter.set_data(data)
        fitter.set_model('sphere')
        fitter.set_param('radius', value=45.0, min=10.0, max=200.0, vary=True)
        result = fitter.fit(engine='bumps', method='amoeba')
        assert result['n_points'] == 39

    def test_n_free_counts_a_varied_pd_width(self):
        fitter = sphere_fitter()
        fitter.enable_polydispersity(True)
        fitter.set_pd_param('radius', pd_width=0.1, vary=True)
        result = fitter.fit(engine='bumps', method='amoeba')
        assert result['n_free'] == 2
        assert result['dof'] == result['n_points'] - 2

    def test_chisq_equals_the_sum_of_stored_residuals(self):
        for engine, method in (
            ('bumps', 'amoeba'),
            ('lmfit', 'leastsq'),
            ('lmfit', 'least_squares'),
        ):
            fitter = sphere_fitter()
            result = fitter.fit(engine=engine, method=method)
            residuals = fitter._fit_contract.artifacts.residuals
            assert residuals is not None, f'{engine}/{method}'
            assert result['chisq'] == pytest.approx(float(np.sum(residuals**2)))

    def test_no_degrees_of_freedom_reports_an_unavailable_reduced_value(self):
        fitter = sphere_fitter(npoints=40)
        fitter.set_q_range(qmin=float(fitter.data.x[-2]))  # two fitted points
        fitter.set_param('scale', value=1.0, min=0.1, max=10.0, vary=True)
        fitter.set_param('background', value=1e-3, min=0.0, max=1.0, vary=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = fitter.fit(engine='bumps', method='amoeba')
        assert result['n_points'] == 2
        assert result['n_free'] == 3
        assert result['dof'] <= 0
        assert math.isnan(result['reduced_chisq'])
        assert math.isfinite(result['chisq'])

    def test_legacy_dict_carries_every_quality_key(self):
        result = sphere_fitter().fit(engine='bumps', method='amoeba')
        for key in (
            'chisq',
            'reduced_chisq',
            'n_points',
            'n_free',
            'dof',
            'converged',
            'message',
            'weighting_note',
            'cov',
            'cov_labels',
            'cov_source',
            'on_bounds',
        ):
            assert key in result, key


class TestPreviewSeam:
    def test_preview_matches_the_initial_chisq_under_custom_resolution(self):
        fitter = sphere_fitter()
        fitter.set_resolution('pinhole', dq_over_q=0.1)
        problem, _ = _build_bumps_problem(
            fitter._evaluation_data(warn=False),
            fitter.kernel,
            fitter._param_manager.snapshot_fit_state(),
        )
        title = fitter.plot_model(show=False).layout.title.text
        assert f'χ²/dof = {problem.chisq():.4f}' in title

    def test_preview_does_not_create_a_fit_result(self):
        fitter = sphere_fitter()
        fitter.plot_model(show=False)
        assert fitter._fit_contract is None
        with pytest.raises(ValueError, match='No fit result available'):
            fitter.get_fit_report()


class TestCsvExport:
    def test_header_carries_the_quality_block(self, tmp_path):
        fitter = sphere_fitter()
        fitter.fit(engine='bumps', method='amoeba')
        path = tmp_path / 'fit.csv'
        fitter.save_results(str(path))
        text = path.read_text(encoding='utf-8')
        for line in (
            '# Chi-squared:',
            '# Reduced chi-squared:',
            '# Free parameters:',
            '# Degrees of freedom:',
            '# Weighting:',
        ):
            assert line in text, line

    def test_mixed_zero_di_exports_finite_residuals_that_match_chisq(self, tmp_path):
        """The scipy engine unit-weights zero-dI points; the export must agree.

        Re-deriving (y - fit) / dy at export time wrote inf for exactly these
        points, and their squares then did not sum to the reported chi-squared.
        """
        data = examples.simulate('sphere', npoints=30, seed=2, noise=0.02, radius=50.0, **FIXED)
        data.dy[3] = 0.0
        data.dy[7] = 0.0
        fitter = SANSFitter()
        fitter.set_data(data)
        fitter.set_model('sphere')
        fitter.set_param('radius', value=45.0, min=10.0, max=200.0, vary=True)
        with pytest.warns(UserWarning, match='zero intensity'):
            result = fitter.fit(engine='lmfit', method='leastsq')

        assert '2 of 30 points unit-weighted' in result['weighting_note']

        path = tmp_path / 'fit.csv'
        fitter.save_results(str(path))
        rows = [
            line
            for line in path.read_text(encoding='utf-8').splitlines()
            if not line.startswith('#')
        ]
        residuals = np.array([float(row.split(',')[-1]) for row in rows[1:]])
        assert np.isfinite(residuals).all()
        assert float(np.sum(residuals**2)) == pytest.approx(result['chisq'], rel=1e-6)


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


class TestConvergence:
    def test_least_squares_out_of_budget_does_not_converge_and_warns(self):
        fitter = sphere_fitter()
        with pytest.warns(UserWarning, match='did not report convergence'):
            result = fitter.fit(engine='lmfit', method='least_squares', max_nfev=1)
        assert result['converged'] is False
        assert result['message']

    def test_leastsq_on_a_well_posed_problem_converges(self):
        result = sphere_fitter().fit(engine='lmfit', method='leastsq')
        assert result['converged'] is True
        assert '\n' not in result['message']

    def test_bumps_reports_no_verdict_and_does_not_warn_about_one(self):
        fitter = sphere_fitter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = fitter.fit(engine='bumps', method='amoeba')
        assert result['converged'] is None
        assert 'iterations reported' in result['message']
        assert 'configured maximum' in result['message']
        # Other warnings (numpy, sasmodels) are fine; a convergence one is not.
        assert not [w for w in caught if 'convergence' in str(w.message)]

    def test_bumps_budget_reflects_an_explicit_step_count(self):
        result = sphere_fitter().fit(engine='bumps', method='amoeba', steps=50)
        assert 'configured maximum: 50' in result['message']

    def test_bumps_budget_accounts_for_starts(self):
        """max_steps multiplies steps by starts, which a bare settings read misses."""
        result = sphere_fitter().fit(engine='bumps', method='amoeba', steps=50, starts=3)
        assert 'configured maximum: 150' in result['message']


class TestBudgetWithoutMaxSteps:
    """bumps 1.0.3 has no FitBase.max_steps, and `bumps>=1.0` still admits it.

    The fallback reproduces the generic default (defaults overridden by the
    caller's options, then steps x starts) from the `settings` every 1.0.x
    exposes. These tests stub the registry so they hold on any installed version.
    """

    @staticmethod
    def _stub_registry(monkeypatch, *fitters):
        import bumps.fitters

        monkeypatch.setattr(bumps.fitters, 'FITTERS', fitters)

    def test_derives_steps_times_starts_from_settings(self, monkeypatch):
        class LegacyAmoeba:
            id = 'amoeba'
            settings = [('steps', 1000), ('starts', 1)]

        self._stub_registry(monkeypatch, LegacyAmoeba)
        assert _configured_budget('amoeba', None, {'steps': 50, 'starts': 3}) == 150
        assert _configured_budget('amoeba', None, {'steps': 50}) == 50
        assert _configured_budget('amoeba', None, {}) == 1000

    def test_a_sampler_budget_is_not_approximated(self, monkeypatch):
        """DREAM's budget is a function of samples and pop; only bumps should derive it."""

        class LegacyDream:
            id = 'dream'
            settings = [('samples', 10000), ('burn', 100), ('pop', 10), ('steps', 0)]

        self._stub_registry(monkeypatch, LegacyDream)
        assert _configured_budget('dream', None, {'samples': 600, 'burn': 40}) is None

    def test_an_unknown_method_has_no_budget(self, monkeypatch):
        self._stub_registry(monkeypatch)
        assert _configured_budget('amoeba', None, {}) is None

    def test_dream_message_survives_a_missing_budget(self, monkeypatch):
        """The sampler settings still reach the message when the maximum is unknown."""
        import sans_fitter.fitting.bumps_engine as engine

        monkeypatch.setattr(engine, '_configured_budget', lambda *args, **kwargs: None)
        fitter = sphere_fitter(npoints=25)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = fitter.fit_bayesian(samples=600, burn=40)
        assert '600 samples' in result['message']
        assert 'configured maximum' not in result['message']

    def test_point_estimate_message_survives_a_missing_budget(self, monkeypatch):
        import sans_fitter.fitting.bumps_engine as engine

        monkeypatch.setattr(engine, '_configured_budget', lambda *args, **kwargs: None)
        result = sphere_fitter().fit(engine='bumps', method='amoeba')
        assert 'configured maximum: unknown' in result['message']


# ---------------------------------------------------------------------------
# Covariance
# ---------------------------------------------------------------------------


class TestCovariance:
    def test_bumps_cov_diagonal_reproduces_the_reported_stderr(self):
        fitter = sphere_fitter()
        result = fitter.fit(engine='bumps', method='amoeba')
        cov = result['cov']
        assert cov.shape == (result['n_free'], result['n_free'])
        assert result['cov_source'] == 'jacobian'
        stderr = [result['parameters'][name]['stderr'] for name in result['cov_labels']]
        np.testing.assert_allclose(np.sqrt(np.diag(cov)), stderr, rtol=1e-6)

    def test_leastsq_cov_diagonal_reproduces_the_reported_stderr(self):
        result = sphere_fitter().fit(engine='lmfit', method='leastsq')
        assert result['cov_source'] == 'scipy cov_x'
        stderr = [result['parameters'][name]['stderr'] for name in result['cov_labels']]
        np.testing.assert_allclose(np.sqrt(np.diag(result['cov'])), stderr, rtol=1e-6)

    def test_differential_evolution_has_no_covariance(self):
        fitter = sphere_fitter()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = fitter.fit(engine='lmfit', method='differential_evolution', maxiter=3, seed=0)
        assert result['cov'] is None
        assert result['cov_source'] is None
        report = fitter.get_fit_report()
        assert report.corr is None
        assert report.strongly_correlated() == []
        assert 'Correlations' not in str(report)

    def test_one_free_parameter_is_a_one_by_one_matrix(self):
        for engine, method in (('bumps', 'amoeba'), ('lmfit', 'leastsq')):
            result = sphere_fitter().fit(engine=engine, method=method)
            assert result['cov'].shape == (1, 1)

    def test_labels_are_user_facing_on_the_set_models_path(self):
        fitter = SANSFitter()
        fitter.set_data(examples.simulate('dab+peak_lorentz', npoints=30, seed=0))
        fitter.set_models('dab', 'peak_lorentz')
        fitter.set_param('dab_cor_length', value=40.0, min=5.0, max=200.0, vary=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = fitter.fit(engine='bumps', method='amoeba')
        assert result['cov_labels'] == ['dab_cor_length']
        assert not any(name.startswith('A_') for name in result['cov_labels'])


class TestDreamCovariance:
    @classmethod
    def setup_class(cls):
        fitter = sphere_fitter(npoints=25)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cls.result = fitter.fit_bayesian(samples=600, burn=40)
        cls.fitter = fitter

    def test_one_varied_parameter_gives_a_one_by_one_matrix(self):
        """np.cov on an (n, 1) sample returns a 0-d scalar without atleast_2d."""
        assert self.result['cov'].shape == (1, 1)
        assert self.result['cov_source'] == 'posterior sample'
        np.testing.assert_allclose(self.fitter.get_fit_report().corr, [[1.0]])

    def test_matches_the_sample_covariance(self):
        posterior = self.fitter.get_posterior()
        expected = np.atleast_2d(np.cov(np.asarray(posterior.samples, dtype=float), rowvar=False))
        np.testing.assert_allclose(self.result['cov'], expected)
        assert self.result['cov_labels'] == list(posterior.labels)

    def test_message_talks_about_samples_not_steps(self):
        assert '600 samples' in self.result['message']
        assert '40 burn-in generations' in self.result['message']
        assert self.result['converged'] is None


class TestStronglyCorrelated:
    @classmethod
    def setup_class(cls):
        """Two analytically proportional parameters: I ∝ scale · (sld − sld_solvent)².

        Only two are free, with the solvent fixed, so the degeneracy is exact
        rather than a three-way one that would make the test seed-sensitive.
        """
        fitter = sphere_fitter()
        fitter.set_param('scale', value=1.0, min=0.1, max=10.0, vary=True)
        fitter.set_param('sld', value=4.0, min=1.5, max=8.0, vary=True)
        fitter.set_param('sld_solvent', value=1.0, vary=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fitter.fit(engine='bumps', method='amoeba')
        cls.report = fitter.get_fit_report()

    def test_finds_the_degenerate_pair(self):
        pairs = self.report.strongly_correlated(threshold=0.9)
        assert pairs, self.report.corr
        names = {frozenset((a, b)) for a, b, _ in pairs}
        assert frozenset(('scale', 'sld')) in names

    def test_reports_each_pair_once_with_a_signed_coefficient(self):
        n = len(self.report.cov_labels)
        pairs = self.report.strongly_correlated(threshold=0.0)
        # Strict upper triangle: every unordered pair exactly once.
        assert len(pairs) == n * (n - 1) // 2
        assert len({frozenset((a, b)) for a, b, _ in pairs}) == len(pairs)
        assert all(-1.0 <= rho <= 1.0 for _, _, rho in pairs)

    @pytest.mark.parametrize('threshold', [-0.1, 1.5])
    def test_rejects_a_threshold_outside_the_unit_interval(self, threshold):
        with pytest.raises(ValueError, match='between 0 and 1'):
            self.report.strongly_correlated(threshold=threshold)

    def test_ignores_non_finite_cells(self):
        report = FitReport(
            model='sphere',
            engine='bumps',
            method='amoeba',
            resolution='none',
            weighting_note='dI',
            chisq=1.0,
            reduced_chisq=0.5,
            n_points=10,
            n_free=2,
            dof=8,
            converged=None,
            message='',
            parameters={},
            cov_labels=['a', 'b'],
            cov=np.array([[1.0, 0.0], [0.0, 0.0]]),
        )
        assert report.strongly_correlated(threshold=0.0) == []


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


class TestOnBounds:
    @staticmethod
    def _walled_fitter():
        """A fit whose optimum lies below a lower bound, so the bound is active.

        The truth is a 20 A radius and the lower bound is 40 A. least_squares is
        strictly bounded, so it settles exactly on the wall.
        """
        fitter = SANSFitter()
        fitter.set_data(
            examples.simulate('sphere', npoints=30, seed=0, noise=0.01, radius=20.0, **FIXED)
        )
        fitter.set_model('sphere')
        fitter.set_param('radius', value=45.0, min=40.0, max=200.0, vary=True)
        for name, value in FIXED.items():
            fitter.set_param(name, value=value, vary=False)
        return fitter

    def test_a_parameter_driven_past_its_bound_is_reported(self):
        fitter = self._walled_fitter()
        with pytest.warns(UserWarning, match='may be constrained'):
            result = fitter.fit(engine='lmfit', method='least_squares')

        assert ('radius', 'min') in result['on_bounds']
        report = fitter.get_fit_report()
        rows = {name: status for name, _, status in report._parameter_rows()}
        assert rows['radius'] == 'fitted, on bound (min)'

    def test_warning_names_the_parameter_and_the_side(self):
        fitter = self._walled_fitter()
        with pytest.warns(UserWarning, match=r'radius = 40 \(min\)'):
            fitter.fit(engine='lmfit', method='least_squares')

    def test_an_ordinary_small_scale_is_not_reported(self):
        """A value three orders of magnitude above min must not count as on it."""
        fitter = sphere_fitter()
        fitter.set_param('scale', value=1e-3, min=1e-5, max=1.0, vary=False)
        result = fitter.fit(engine='bumps', method='amoeba')
        assert not any(name == 'scale' for name, _ in result['on_bounds'])

        # And directly, without relying on where the optimizer lands.
        assert not at_bound(1e-3, 1e-5)

    def test_fixed_and_linked_parameters_are_never_reported(self):
        fitter = sphere_fitter()
        # background is fixed at exactly its own lower bound.
        fitter.set_param('background', value=0.0, min=0.0, max=1.0, vary=False)
        result = fitter.fit(engine='bumps', method='amoeba')
        assert not any(name == 'background' for name, _ in result['on_bounds'])

    def test_a_varied_parameter_at_exactly_zero_hits_a_zero_bound(self):
        """A purely relative rule would miss this: 1e-4 * 0 is 0, so the floor decides.

        Driven through the detector rather than through an optimizer, because no
        optimizer reliably lands on exactly 0.0.
        """
        fitter = sphere_fitter()
        fitter.set_param('background', value=1e-3, min=0.0, max=1.0, vary=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fitter.fit(engine='bumps', method='amoeba')

        fitter._fit_contract.parameters['background']['value'] = 0.0
        assert ('background', 'min') in fitter._find_parameters_on_bounds()

    def test_an_infinite_bound_is_never_reported(self):
        fitter = sphere_fitter()
        fitter.set_param('radius', value=45.0, min=10.0, max=float('inf'), vary=True)
        result = fitter.fit(engine='bumps', method='amoeba')
        assert not any(side == 'max' for _, side in result['on_bounds'])


# ---------------------------------------------------------------------------
# FitReport
# ---------------------------------------------------------------------------


def hostile_report(**overrides):
    """A report carrying text and numbers that would break a naive renderer."""
    defaults = {
        'model': 'a|b <script>&',
        'engine': 'bumps',
        'method': 'amoeba',
        'resolution': 'none',
        'weighting_note': 'dI',
        'chisq': 12.5,
        'reduced_chisq': 0.5,
        'n_points': 26,
        'n_free': 2,
        'dof': 24,
        'converged': False,
        'message': 'line one\nline | two <b>&',
        'parameters': {
            'a|b': {
                'value': 1.0,
                'stderr': 0.1,
                'formatted': '1.0(1)',
                'fixed': False,
                'linked_to': None,
            },
            'c': {
                'value': 2.0,
                'stderr': 0.0,
                'formatted': '2 (fixed)',
                'fixed': True,
                'linked_to': None,
            },
        },
        'cov_labels': ['a|b', 'c<d'],
        'cov': np.array([[1.0, 0.99], [0.99, 1.0]]),
        'cov_source': 'jacobian',
        'on_bounds': [('a|b', 'max')],
    }
    defaults.update(overrides)
    return FitReport(**defaults)


class TestFitReportRendering:
    def test_text_report_uses_the_console_glyph_constants(self, monkeypatch):
        """print(report) must survive a console that cannot encode chi-squared.

        The glyph constants resolve once at import, so the ASCII fallback is
        exercised by substituting them rather than by swapping stdout.
        """
        import sans_fitter.report as report_module

        monkeypatch.setattr(report_module, 'CHI_SQUARED', 'chi^2')
        monkeypatch.setattr(report_module, 'ARROW', '->')
        text = str(hostile_report())
        text.encode('cp1252', errors='strict')
        assert 'chi^2/dof' in text

    def test_markdown_keeps_the_unicode_symbol(self, monkeypatch):
        """A Markdown string is placed by the caller, not written to a console."""
        import sans_fitter.report as report_module

        monkeypatch.setattr(report_module, 'CHI_SQUARED', 'chi^2')
        assert 'chi-squared per dof' not in hostile_report().to_markdown()
        assert '\u03c7\u00b2/dof' in hostile_report().to_markdown()

    def test_markdown_keeps_its_column_count(self):
        markdown = hostile_report().to_markdown()
        assert 'χ²/dof' in markdown
        table_rows = [line for line in markdown.splitlines() if line.startswith('| ')]
        assert table_rows
        for row in table_rows:
            # An unescaped pipe or newline in a cell would change the column count.
            assert '\n' not in row

    def test_markdown_escapes_a_pipe(self):
        assert escape_markdown_cell('a|b') == r'a\|b'
        assert '\\|' in hostile_report().to_markdown()

    def test_html_escapes_markup(self):
        html = hostile_report()._repr_html_()
        assert '<script>' not in html
        assert '&lt;script&gt;' in html
        assert html.count('<table') == 3  # quality, parameters, correlations

    def test_no_correlation_table_for_a_single_parameter(self):
        report = hostile_report(cov_labels=['a'], cov=np.array([[1.0]]), n_free=1)
        assert report._repr_html_().count('<table') == 2
        assert 'Correlations' not in str(report)

    def test_nan_cells_never_render_as_the_token_nan(self):
        report = hostile_report(
            cov=np.array([[1.0, 0.0], [0.0, 0.0]]),
            chisq=float('nan'),
            reduced_chisq=float('nan'),
        )
        for rendered in (str(report), report.to_markdown(), report._repr_html_()):
            assert 'nan' not in rendered.lower().replace('n/a', '')

    def test_estimate_column_does_not_repeat_the_status(self):
        rows = {
            name: (estimate, status)
            for name, estimate, status in hostile_report()._parameter_rows()
        }
        estimate, status = rows['c']
        assert status == 'fixed'
        assert '(fixed)' not in estimate

    def test_fitted_rows_keep_the_engine_formatting(self):
        rows = {name: estimate for name, estimate, _ in hostile_report()._parameter_rows()}
        assert rows['a|b'] == '1.0(1)'

    def test_header_states_model_engine_resolution_and_weighting(self):
        text = str(hostile_report())
        assert 'bumps/amoeba' in text
        assert 'weighting: dI' in text


class TestFitReportSerialization:
    def test_round_trips_through_strict_json(self):
        report = hostile_report(
            chisq=float('nan'),
            reduced_chisq=float('nan'),
            dof=0,
            cov=np.array([[1.0, 0.0], [0.0, 0.0]]),
            parameters={
                'a': {
                    'value': np.float64('nan'),
                    'stderr': np.float32(0.5),
                    'formatted': 'n/a',
                    'fixed': False,
                    'linked_to': None,
                },
            },
        )
        payload = json.dumps(report.to_dict(), allow_nan=False)
        restored = json.loads(payload)
        assert restored['chisq'] is None
        assert restored['parameters']['a']['value'] is None
        assert restored['parameters']['a']['stderr'] == pytest.approx(0.5)
        assert restored['corr'][1][1] is None

    def test_sanitizer_handles_numpy_scalars_and_arrays(self):
        payload = json_safe(
            {
                'int': np.int64(3),
                'float': np.float64(1.5),
                'bool': np.bool_(True),
                'array': np.array([1.0, float('inf')]),
                'nested': {'deep': [float('nan')]},
            }
        )
        json.dumps(payload, allow_nan=False)
        assert payload['int'] == 3
        assert payload['bool'] is True
        assert payload['array'][1] is None
        assert payload['nested']['deep'][0] is None

    def test_posterior_diagnostics_with_a_non_finite_r_hat_serialize(self):
        fitter = sphere_fitter(npoints=25)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fitter.fit_bayesian(samples=600, burn=40)
        posterior = fitter.get_posterior()
        if posterior.diagnostics:
            name = next(iter(posterior.diagnostics))
            posterior.diagnostics[name]['r_hat'] = float('nan')
        report = fitter.get_fit_report()
        payload = json.dumps(report.to_dict(), allow_nan=False)
        assert '"posterior"' in payload


class TestFitReportAccess:
    def test_raises_before_any_fit(self):
        fitter = sphere_fitter()
        with pytest.raises(ValueError, match='No fit result available'):
            fitter.get_fit_report()

    def test_is_a_snapshot(self):
        fitter = sphere_fitter()
        fitter.fit(engine='bumps', method='amoeba')
        report = fitter.get_fit_report()
        original = report.parameters['radius']['value']

        fitter.fit_result['parameters']['radius']['value'] = 999.0
        fitter._fit_contract.parameters['radius']['value'] = 999.0

        assert report.parameters['radius']['value'] == original

    def test_covariance_is_copied(self):
        fitter = sphere_fitter()
        fitter.fit(engine='bumps', method='amoeba')
        report = fitter.get_fit_report()
        before = float(report.cov[0, 0])
        fitter._fit_contract.cov[0, 0] = 123.0
        assert float(report.cov[0, 0]) == before

    def test_rejects_a_covariance_that_does_not_match_its_labels(self):
        with pytest.raises(ValueError, match=r'\(2, 2\)'):
            hostile_report(cov_labels=['only-one'])

    def test_reports_state_the_weighting_on_both_engines(self):
        for engine, method in (('bumps', 'amoeba'), ('lmfit', 'leastsq')):
            fitter = sphere_fitter()
            fitter.fit(engine=engine, method=method)
            assert 'weighting: dI' in str(fitter.get_fit_report())

    def test_bayesian_report_includes_the_posterior_block(self):
        fitter = sphere_fitter(npoints=25)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fitter.fit_bayesian(samples=600, burn=40)
        assert 'Posterior summary:' in str(fitter.get_fit_report())

    def test_the_posterior_is_snapshotted_too(self):
        """A Bayesian report must not alias the fitter's live PosteriorSummary.

        Its samples array and its per-parameter statistic and diagnostic
        dictionaries are all mutable, so a caller holding the object from
        get_posterior() could otherwise rewrite a report already handed out.
        """
        fitter = sphere_fitter(npoints=25)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fitter.fit_bayesian(samples=600, burn=40)

        report = fitter.get_fit_report()
        assert report.posterior is not None
        name = report.posterior.labels[0]
        before = {
            'summary': report.posterior.format_summary(),
            'samples': np.array(report.posterior.samples, copy=True),
            'mean': report.posterior.mean[name],
            'diagnostics': None
            if not report.posterior.diagnostics
            else dict(report.posterior.diagnostics[name]),
        }

        live = fitter.get_posterior()
        assert report.posterior is not live
        live.samples[:] = 0.0
        live.mean[name] = 12345.0
        live.median[name] = 12345.0
        live.ci_68[name] = (0.0, 0.0)
        if live.diagnostics:
            live.diagnostics[name]['r_hat'] = 99.0

        assert report.posterior.mean[name] == before['mean']
        np.testing.assert_array_equal(report.posterior.samples, before['samples'])
        assert report.posterior.format_summary() == before['summary']
        if before['diagnostics'] is not None:
            assert report.posterior.diagnostics[name] == before['diagnostics']
        assert 'Posterior summary:' in str(report)

    def test_public_import_path_is_stable(self):
        from sans_fitter.report import FitReport as Imported

        assert Imported is FitReport


# ---------------------------------------------------------------------------
# Legacy adapters
# ---------------------------------------------------------------------------


class TestLegacyAdapters:
    """A result dict from before 0.4 has a recoverable chi-squared meaning.

    bumps stored chi-squared per degree of freedom and the scipy engine stored the
    raw sum, so the adapter converts rather than guesses — and never labels a
    historical value as the wrong one.
    """

    def _legacy_fitter(self, engine, method):
        fitter = sphere_fitter()
        result = fitter.fit(engine=engine, method=method)
        dof = result['dof']
        legacy = {
            'engine': result['engine'],
            'method': result['method'],
            'chisq': result['reduced_chisq'] if engine == 'bumps' else result['chisq'],
            'parameters': result['parameters'],
        }
        fitter._fit_contract = None
        fitter.fit_result = legacy
        return fitter, result, dof

    def test_legacy_bumps_dict_recovers_both_values(self):
        fitter, original, dof = self._legacy_fitter('bumps', 'amoeba')
        contract = fitter._get_active_fit_contract()
        assert contract.dof == dof
        assert contract.reduced_chisq == pytest.approx(original['reduced_chisq'])
        assert contract.chisq == pytest.approx(original['chisq'], rel=1e-6)
        assert 'legacy result' in contract.message

    def test_legacy_lmfit_dict_recovers_both_values(self):
        fitter, original, dof = self._legacy_fitter('lmfit', 'leastsq')
        contract = fitter._get_active_fit_contract()
        assert contract.dof == dof
        assert contract.chisq == pytest.approx(original['chisq'])
        assert contract.reduced_chisq == pytest.approx(original['reduced_chisq'], rel=1e-6)

    def test_plot_title_is_the_reduced_value_on_a_legacy_dict(self):
        fitter, original, _ = self._legacy_fitter('bumps', 'amoeba')
        title = fitter.plot_results(show=False).layout.title.text
        assert f'χ²/dof = {original["reduced_chisq"]:.4f}' in title

    def test_report_still_refuses_a_reconstructed_dict(self):
        fitter, _, _ = self._legacy_fitter('bumps', 'amoeba')
        with pytest.raises(ValueError, match='No fit result available'):
            fitter.get_fit_report()

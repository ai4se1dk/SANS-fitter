"""Fitting several datasets together: truth recovery, identities, uncertainties.

The numerical half. Fixtures are deterministic (fixed seeds, ``lm``), and the
assertions are mostly identities rather than tolerances — the objective must
reconstruct from its per-dataset parts, a shared parameter must count once, an
exported curve must add back up to the reported total — because those are the
things a joint fit can get quietly wrong.
"""

import copy
import math
import os
import warnings

import numpy as np
import pytest

from sans_fitter import MultiFitter, SANSFitter, examples

TRUTH = {'radius': 45.0, 'scale': 0.02, 'sld': 4.0}


@pytest.fixture(scope='module')
def contrast():
    """Two contrasts of one particle: same radius, different solvent and background."""
    return {
        'h2o': examples.simulate(
            'sphere',
            radius=TRUTH['radius'],
            scale=TRUTH['scale'],
            background=0.01,
            sld=TRUTH['sld'],
            sld_solvent=-0.56,
            noise=0.02,
            seed=11,
            npoints=60,
        ),
        'd2o': examples.simulate(
            'sphere',
            radius=TRUTH['radius'],
            scale=TRUTH['scale'],
            background=0.03,
            sld=TRUTH['sld'],
            sld_solvent=6.34,
            noise=0.02,
            seed=12,
            qmin=0.008,
            qmax=0.3,
            npoints=45,
        ),
    }


SOLVENTS = {'h2o': -0.56, 'd2o': 6.34}


def read_curve(path):
    """Read an exported curve file, skipping its '#' provenance header."""
    lines = [
        line
        for line in path.read_text(encoding='utf-8').splitlines()
        if line and not line.startswith('#')
    ]
    columns = lines[0].split(',')
    values = np.array([[float(cell) for cell in line.split(',')] for line in lines[1:]])
    return dict(zip(columns, values.T, strict=True))


def contrast_fit(contrast, share=('radius', 'scale')):
    fit = MultiFitter()
    for name, data in contrast.items():
        fit.add(name, copy.deepcopy(data), model='sphere')
        entry = fit[name]
        entry.set_param('radius', value=40, min=10, max=100, vary=True)
        entry.set_param('scale', value=0.02, min=0.001, max=0.1, vary=True)
        entry.set_param('background', value=0.02, min=0.0, max=0.1, vary=True)
        entry.set_param('sld', value=TRUTH['sld'], vary=False)
        fit.constrain(f'{name}.sld_solvent', SOLVENTS[name])
    if share:
        fit.share(*share)
    return fit


# =========================================================================
# Truth recovery and counting
# =========================================================================


class TestJointFit:
    def test_shared_geometry_recovers_the_truth(self, contrast):
        fit = contrast_fit(contrast)
        result = fit.fit(method='lm')
        radius = result.parameters['h2o.radius']
        assert abs(radius.value - TRUTH['radius']) < 5 * radius.stderr
        assert result.parameters['h2o.scale'].value == pytest.approx(TRUTH['scale'], rel=0.05)
        assert result.reduced_chisq < 2.0

    def test_a_shared_parameter_counts_once(self, contrast):
        fit = contrast_fit(contrast)
        result = fit.fit(method='lm')
        # 6 parameters over two datasets, minus one shared radius, one shared
        # scale and two constrained solvent SLDs, minus two fixed slds.
        assert result.n_free == 4
        assert sorted(result.cov_labels) == [
            'd2o.background',
            'd2o.radius',
            'd2o.scale',
            'h2o.background',
        ]
        assert result.dof == result.n_points - result.n_free

    def test_members_of_a_shared_group_agree_exactly(self, contrast):
        fit = contrast_fit(contrast)
        result = fit.fit(method='lm')
        assert result.parameters['h2o.radius'].value == result.parameters['d2o.radius'].value
        assert result.parameters['h2o.radius'].stderr == result.parameters['d2o.radius'].stderr
        assert result.parameters['d2o.radius'].status == 'shared'
        assert result.parameters['d2o.radius'].root == 'd2o.radius'

    def test_constrained_parameters_stay_exact(self, contrast):
        fit = contrast_fit(contrast)
        result = fit.fit(method='lm')
        for name, value in SOLVENTS.items():
            entry = result.parameters[f'{name}.sld_solvent']
            assert entry.value == value
            assert entry.status == 'fixed'
            assert entry.stderr == 0.0

    def test_fitted_values_are_committed_to_every_dataset(self, contrast):
        fit = contrast_fit(contrast)
        result = fit.fit(method='lm')
        for name in ('h2o', 'd2o'):
            assert fit[name].params['radius']['value'] == pytest.approx(
                result.parameters['h2o.radius'].value
            )

    def test_subset_sharing_keeps_other_datasets_independent(self, contrast):
        third = examples.simulate(
            'sphere',
            radius=70,
            scale=0.02,
            background=0.01,
            sld=4.0,
            sld_solvent=1.0,
            noise=0.02,
            seed=13,
            npoints=40,
        )
        fit = contrast_fit(contrast)
        fit.add('big', third, model='sphere')
        fit['big'].set_param('radius', value=65, min=10, max=120, vary=True)
        fit['big'].set_param('scale', value=0.02, min=0.001, max=0.1, vary=True)
        fit['big'].set_param('background', value=0.01, min=0.0, max=0.1, vary=True)
        fit['big'].set_param('sld', value=4.0, vary=False)
        fit.constrain('big.sld_solvent', 1.0)
        result = fit.fit(method='lm')
        assert result.parameters['big.radius'].value == pytest.approx(70, rel=0.1)
        assert result.parameters['big.radius'].value != result.parameters['h2o.radius'].value

    def test_different_models_can_be_related(self, contrast):
        cylinder = examples.simulate(
            'cylinder',
            radius=45,
            length=400,
            scale=0.02,
            background=0.01,
            noise=0.02,
            seed=14,
            npoints=40,
        )
        fit = contrast_fit(contrast, share=('radius',))
        fit.add('cyl', cylinder, model='cylinder')
        fit['cyl'].set_param('radius', value=40, min=10, max=100, vary=True)
        fit['cyl'].set_param('length', value=400, min=100, max=900, vary=True)
        fit['cyl'].set_param('scale', value=0.02, min=0.001, max=0.1, vary=True)
        fit['cyl'].set_param('background', value=0.01, min=0.0, max=0.1, vary=True)
        fit.link_params('cyl.radius', to='h2o.radius')
        result = fit.fit(method='lm')
        assert result.parameters['cyl.radius'].value == result.parameters['h2o.radius'].value
        assert result.parameters['cyl.radius'].status == 'shared'
        assert 'cyl.length' in result.cov_labels

    def test_unequal_grids_keep_their_own_points(self, contrast):
        fit = contrast_fit(contrast)
        fit['h2o'].set_q_range(qmin=0.02, qmax=0.25)
        result = fit.fit(method='lm')
        assert result.datasets['h2o'].n_points < len(contrast['h2o'].x)
        assert result.datasets['d2o'].n_points == len(contrast['d2o'].x)
        assert result.n_points == sum(d.n_points for d in result.datasets.values())
        for entry in result.datasets.values():
            assert len(entry.fitted_curve) == entry.n_points
            assert len(entry.residuals) == entry.n_points

    def test_mixed_resolution_modes_are_respected(self, contrast):
        fit = contrast_fit(contrast)
        fit['h2o'].set_resolution('pinhole', dq_over_q=0.12)
        fit['d2o'].set_resolution('none')
        result = fit.fit(method='lm')
        assert 'pinhole' in result.datasets['h2o'].resolution
        assert 'No' in result.datasets['d2o'].resolution or 'none' in (
            result.datasets['d2o'].resolution.lower()
        )

    def test_dataset_order_does_not_change_the_optimum(self, contrast):
        forward = contrast_fit(contrast).fit(method='lm')
        reversed_curves = {name: contrast[name] for name in reversed(list(contrast))}
        backward = contrast_fit(reversed_curves).fit(method='lm')
        assert forward.chisq == pytest.approx(backward.chisq, rel=1e-6)
        assert sorted(forward.cov_labels) == sorted(backward.cov_labels)
        for label in forward.cov_labels:
            assert forward.parameters[label].value == pytest.approx(
                backward.parameters[label].value, rel=1e-4
            )


# =========================================================================
# One-entry equivalence (the A0 gate)
# =========================================================================


class TestSingleEntryEquivalence:
    @staticmethod
    def _single(data):
        fitter = SANSFitter()
        fitter.set_data(copy.deepcopy(data))
        fitter.set_model('sphere')
        fitter.set_param('radius', value=40, min=10, max=100, vary=True)
        fitter.set_param('scale', value=0.02, min=0.001, max=0.1, vary=True)
        fitter.set_param('background', value=0.02, min=0.0, max=0.1, vary=True)
        fitter.set_param('sld', value=4.0, vary=False)
        fitter.set_param('sld_solvent', value=-0.56, vary=False)
        return fitter

    @staticmethod
    def _multi(data):
        fit = MultiFitter()
        fit.add('one', copy.deepcopy(data), model='sphere')
        entry = fit['one']
        entry.set_param('radius', value=40, min=10, max=100, vary=True)
        entry.set_param('scale', value=0.02, min=0.001, max=0.1, vary=True)
        entry.set_param('background', value=0.02, min=0.0, max=0.1, vary=True)
        entry.set_param('sld', value=4.0, vary=False)
        entry.set_param('sld_solvent', value=-0.56, vary=False)
        return fit

    @pytest.fixture(scope='class')
    def pair(self, contrast):
        single = self._single(contrast['h2o'])
        single_result = single.fit(engine='bumps', method='lm')
        multi = self._multi(contrast['h2o'])
        multi_result = multi.fit(method='lm')
        return single_result, multi_result

    def test_fitted_values_agree(self, pair):
        single, multi = pair
        for name in ('radius', 'scale', 'background'):
            assert multi.parameters[f'one.{name}'].value == pytest.approx(
                single['parameters'][name]['value'], rel=1e-4
            )

    def test_uncertainties_agree(self, pair):
        single, multi = pair
        for name in ('radius', 'scale', 'background'):
            assert multi.parameters[f'one.{name}'].stderr == pytest.approx(
                single['parameters'][name]['stderr'], rel=1e-2
            )

    def test_goodness_of_fit_agrees(self, pair):
        single, multi = pair
        assert multi.chisq == pytest.approx(single['chisq'], rel=1e-6)
        assert multi.n_points == single['n_points']
        assert multi.n_free == single['n_free']
        assert multi.dof == single['dof']

    def test_covariance_agrees(self, pair):
        single, multi = pair
        order = [multi.cov_labels.index(f'one.{name}') for name in single['cov_labels']]
        reordered = np.asarray(multi.cov)[np.ix_(order, order)]
        np.testing.assert_allclose(reordered, single['cov'], rtol=2e-2)

    def test_the_objective_equals_chi_squared_without_weights(self, pair):
        _single, multi = pair
        assert multi.objective == pytest.approx(multi.chisq)
        assert not multi.weighted


class TestLinkedUncertaintyCorrection:
    """A0: an equality follower reports its target's error, not zero."""

    def test_a_single_fit_follower_inherits_the_target_error(self, contrast):
        fitter = SANSFitter()
        fitter.set_data(copy.deepcopy(contrast['h2o']))
        fitter.set_model('sphere')
        fitter.set_param('radius', value=40, min=10, max=100, vary=True)
        fitter.set_param('scale', value=0.02, min=0.001, max=0.1, vary=True)
        fitter.set_param('sld', value=4.0, vary=False)
        fitter.set_param('sld_solvent', value=-0.56, vary=False)
        fitter.link_params('background', to='scale')
        result = fitter.fit(engine='bumps', method='lm')

        follower = result['parameters']['background']
        target = result['parameters']['scale']
        assert follower['linked_to'] == 'scale'
        assert follower['fixed'] is True
        assert follower['stderr'] == target['stderr'] > 0
        assert follower['formatted'] == target['formatted']

    def test_a_genuinely_fixed_parameter_still_reports_zero(self, contrast):
        fitter = SANSFitter()
        fitter.set_data(copy.deepcopy(contrast['h2o']))
        fitter.set_model('sphere')
        fitter.set_param('radius', value=40, min=10, max=100, vary=True)
        fitter.set_param('sld', value=4.0, vary=False)
        result = fitter.fit(engine='bumps', method='lm')
        assert result['parameters']['sld']['stderr'] == 0.0
        assert result['parameters']['sld']['linked_to'] is None


# =========================================================================
# Weighting and covariance
# =========================================================================


class TestWeighting:
    def test_a_public_factor_scales_residuals_by_its_square_root(self, contrast):
        fit = contrast_fit(contrast)
        fit.set_dataset_weight('d2o', 0.25)
        result = fit.fit(method='lm')
        entry = result.datasets['d2o']
        np.testing.assert_allclose(entry.objective_residuals, 0.5 * entry.residuals)

    def test_the_objective_reconstructs_from_its_parts(self, contrast):
        fit = contrast_fit(contrast)
        fit.set_dataset_weight('d2o', 0.25)
        result = fit.fit(method='lm')
        assert result.objective == pytest.approx(
            sum(entry.objective_contribution for entry in result.datasets.values())
        )
        assert result.chisq == pytest.approx(sum(entry.chisq for entry in result.datasets.values()))
        concatenated = np.concatenate(
            [entry.objective_residuals for entry in result.datasets.values()]
        )
        assert result.objective == pytest.approx(float(np.dot(concatenated, concatenated)))

    def test_raw_and_weighted_statistics_are_reported_separately(self, contrast):
        fit = contrast_fit(contrast)
        fit.set_dataset_weight('d2o', 0.25)
        result = fit.fit(method='lm')
        assert result.weighted
        assert result.objective != result.chisq
        assert result.reduced_objective == pytest.approx(result.objective / result.dof)
        assert 'Weighted objective' in str(fit.get_fit_report())

    def test_weighted_covariance_is_labelled_as_such(self, contrast):
        fit = contrast_fit(contrast)
        fit.set_dataset_weight('d2o', 0.25)
        result = fit.fit(method='lm')
        assert result.cov_source == 'weighted-sandwich-known-dI'

    def test_rescaling_every_factor_leaves_the_covariance_unchanged(self, contrast):
        def run(scale):
            fit = contrast_fit(contrast)
            fit.set_dataset_weight('h2o', 1.0 * scale)
            fit.set_dataset_weight('d2o', 0.25 * scale)
            return fit.fit(method='lm')

        base, rescaled = run(1.0), run(9.0)
        assert base.cov_labels == rescaled.cov_labels
        np.testing.assert_allclose(base.cov, rescaled.cov, rtol=1e-5)

    def test_the_sandwich_matches_the_analytic_value_for_a_constant_model(self, contrast):
        """A pure-background model is exactly linear, so the answer is closed form.

        With ``scale`` fixed at zero the model is ``I = background`` and the
        residual is ``(b - I_j)/dI_j``, whose derivative is ``1/dI_j``. For one
        shared background over two datasets the sandwich reduces to
        ``(Σ a_d² S_d) / (Σ a_d S_d)²`` with ``S_d = Σ_j dI_dj⁻²`` — which is not
        ``1/Σ a_d S_d``, the inverse curvature, unless every ``a_d`` is equal.
        """
        weights = {'h2o': 1.0, 'd2o': 0.25}
        fit = MultiFitter()
        for name, data in contrast.items():
            fit.add(name, copy.deepcopy(data), model='sphere')
            entry = fit[name]
            entry.set_param('scale', value=0.0, min=0.0, max=1.0, vary=False)
            entry.set_param('background', value=0.02, min=0.0, max=1.0, vary=True)
            fit.set_dataset_weight(name, weights[name])
        fit.share('background')
        result = fit.fit(method='lm')

        sums = {
            name: float(np.sum(1.0 / np.asarray(fit[name].data.dy, dtype=float) ** 2))
            for name in contrast
        }
        curvature = sum(weights[name] * sums[name] for name in sums)
        middle = sum(weights[name] ** 2 * sums[name] for name in sums)
        expected = middle / curvature**2

        assert result.cov.shape == (1, 1)
        # A few parts in a thousand: bumps differentiates numerically and the
        # kernel is compiled in single precision, so the agreement is not exact.
        assert float(result.cov[0, 0]) == pytest.approx(expected, rel=5e-3)
        # The inverse curvature answers a different question; here it is ~20%
        # larger, and reporting it would overstate the uncertainty.
        assert abs(float(result.cov[0, 0]) * curvature - 1.0) > 0.1


class TestDerivedUncertainty:
    def test_an_affine_constraint_propagates_its_coefficient(self, contrast):
        fit = contrast_fit(contrast, share=('radius',))
        fit['d2o'].set_param('scale', min=0.0, max=0.08)
        fit.constrain('d2o.scale', '2 * h2o.scale')
        result = fit.fit(method='lm')
        root = result.parameters['h2o.scale']
        derived = result.parameters['d2o.scale']
        assert derived.status == 'derived'
        assert derived.value == pytest.approx(2 * root.value)
        assert derived.stderr == pytest.approx(2 * root.stderr)
        assert 'propagated' in derived.uncertainty_source

    def test_a_difference_of_two_roots_uses_the_cross_covariance(self, contrast):
        fit = contrast_fit(contrast, share=('radius',))
        fit['d2o'].set_param('background', min=-0.2, max=0.2)
        fit.constrain('d2o.background', 'h2o.background + h2o.scale')
        result = fit.fit(method='lm')

        labels = result.cov_labels
        gradient = np.zeros(len(labels))
        gradient[labels.index('h2o.background')] = 1.0
        gradient[labels.index('h2o.scale')] = 1.0
        expected = math.sqrt(float(gradient @ np.asarray(result.cov) @ gradient))

        assert result.parameters['d2o.background'].stderr == pytest.approx(expected)
        # Quadrature would ignore the covariance term and give a different answer.
        quadrature = math.hypot(
            result.parameters['h2o.background'].stderr, result.parameters['h2o.scale'].stderr
        )
        assert result.parameters['d2o.background'].stderr != pytest.approx(quadrature, rel=1e-6)

    def test_constraints_hold_at_preview_and_after_the_fit(self, contrast):
        fit = contrast_fit(contrast, share=('radius',))
        fit['d2o'].set_param('scale', min=0.0, max=0.08)
        fit.constrain('d2o.scale', '2 * h2o.scale')
        before = fit.get_parameter_table()
        pairs = {row['parameter']: row['value'] for row in before}
        assert pairs['d2o.scale'] == pytest.approx(2 * pairs['h2o.scale'])
        result = fit.fit(method='lm')
        assert result.parameters['d2o.scale'].value == pytest.approx(
            2 * result.parameters['h2o.scale'].value
        )
        assert fit['d2o'].params['scale']['value'] == pytest.approx(
            2 * fit['h2o'].params['scale']['value']
        )


# =========================================================================
# Preflight and degenerate cases
# =========================================================================


class TestPreflight:
    def test_a_dataset_without_uncertainties_is_refused_by_name(self, contrast):
        noiseless = examples.simulate('sphere', radius=45, noise=0.0, seed=7, npoints=30)
        fit = contrast_fit(contrast)
        fit.add('bad', noiseless, model='sphere')
        fit['bad'].set_param('radius', value=45, min=10, max=100, vary=True)
        with pytest.raises(ValueError, match="'bad'.*uncertainties"):
            fit.fit(method='lm')

    def test_identical_datasets_are_warned_about(self, contrast):
        fit = MultiFitter()
        for name in ('one', 'two'):
            fit.add(name, copy.deepcopy(contrast['h2o']), model='sphere')
            fit[name].set_param('radius', value=40, min=10, max=100, vary=True)
        with pytest.warns(UserWarning, match='identical data'):
            fit.fit(method='lm')

    def test_an_all_fixed_analysis_evaluates_once(self, contrast):
        fit = MultiFitter()
        for name, data in contrast.items():
            fit.add(name, copy.deepcopy(data), model='sphere')
            fit[name].set_param('radius', value=45, vary=False)
        with pytest.warns(UserWarning, match='No parameters are free'):
            result = fit.fit(method='lm')
        assert result.n_free == 0
        assert 'no free parameters' in result.message
        assert math.isfinite(result.chisq)

    def test_an_unsupported_engine_fails_clearly(self, contrast):
        fit = contrast_fit(contrast)
        with pytest.raises(ValueError, match='not implemented'):
            fit.fit(engine='lmfit')

    def test_a_negative_degrees_of_freedom_yields_nan(self, contrast):
        tiny = examples.simulate('sphere', radius=45, noise=0.02, seed=8, npoints=5)
        fit = MultiFitter()
        for index in range(2):
            name = f'd{index}'
            fit.add(name, copy.deepcopy(tiny), model='sphere')
            for parameter, bounds in (
                ('radius', (10, 100)),
                ('scale', (0.001, 5)),
                ('background', (0.0, 5)),
                ('sld', (-10, 10)),
                ('sld_solvent', (-10, 10)),
            ):
                fit[name].set_param(parameter, min=bounds[0], max=bounds[1], vary=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = fit.fit(method='lm')
        assert result.dof <= 0
        assert math.isnan(result.reduced_chisq)


class TestStaleResults:
    def test_changing_the_configuration_warns_on_reuse(self, contrast):
        fit = contrast_fit(contrast)
        fit.fit(method='lm')
        fit['h2o'].set_q_range(qmin=0.03, qmax=0.2)
        with pytest.warns(UserWarning, match='configuration has changed'):
            fit.plot_results(show=False)

    def test_an_unchanged_configuration_does_not_warn(self, contrast):
        fit = contrast_fit(contrast)
        fit.fit(method='lm')
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            fit.plot_results(show=False)

    def test_a_failed_fit_leaves_the_configuration_intact(self, contrast):
        fit = contrast_fit(contrast)
        before = fit['h2o'].params['radius']['value']
        # bumps raises its own type for an unknown optimizer; what matters here
        # is that whatever it raises leaves the analysis untouched.
        with pytest.raises(BaseException):  # noqa: B017, PT011
            fit.fit(method='no_such_optimizer')
        assert fit['h2o'].params['radius']['value'] == before
        assert fit.result is None


# =========================================================================
# Reporting and export
# =========================================================================


class TestReportAndExport:
    @pytest.fixture(scope='class')
    def fitted(self, contrast):
        fit = contrast_fit(contrast)
        fit.set_dataset_weight('d2o', 0.5)
        fit.fit(method='lm')
        return fit

    def test_the_report_renders_three_ways(self, fitted):
        report = fitted.get_fit_report()
        text = str(report)
        assert 'Simultaneous fit report' in text
        assert 'Shared quantities' in text
        assert 'χ²/N is a mean squared normalized residual' in text
        markdown = report.to_markdown()
        assert markdown.startswith('**Simultaneous fit:**')
        assert '| Dataset | Model |' in markdown
        html = report._repr_html_()
        assert html.startswith('<div>') and '</div>' in html

    def test_the_report_never_presents_a_per_dataset_reduced_chi_squared(self, fitted):
        text = str(fitted.get_fit_report())
        for entry in fitted.result.datasets.values():
            assert f'{entry.chisq / max(entry.n_points - fitted.result.n_free, 1):.6g}' not in text

    def test_to_dict_is_json_serializable(self, fitted):
        import json

        payload = fitted.result.to_dict()
        json.dumps(payload, allow_nan=False)
        assert payload['n_datasets'] == 2
        assert set(payload['datasets']) == {'h2o', 'd2o'}
        assert payload['parameters']['h2o.radius']['status'] == 'shared'

    def test_save_results_writes_the_expected_files(self, fitted, tmp_path):
        target = tmp_path / 'joint'
        fitted.save_results(str(target))
        names = set(os.listdir(target))
        assert {
            'parameters.csv',
            'datasets.csv',
            'covariance.csv',
            'h2o_curve.csv',
            'd2o_curve.csv',
            'manifest.txt',
        } <= names

    def test_exported_curves_reconstruct_the_reported_totals(self, fitted, tmp_path):
        target = tmp_path / 'joint'
        fitted.save_results(str(target))
        total = 0.0
        objective = 0.0
        for name in ('h2o', 'd2o'):
            rows = read_curve(target / f'{name}_curve.csv')
            total += float(np.sum(rows['Residual'] ** 2))
            objective += float(np.sum(rows['Objective_residual'] ** 2))
        assert total == pytest.approx(fitted.result.chisq, rel=1e-6)
        assert objective == pytest.approx(fitted.result.objective, rel=1e-6)

    def test_the_manifest_names_the_weighting_convention(self, fitted, tmp_path):
        target = tmp_path / 'joint'
        fitted.save_results(str(target))
        text = (target / 'manifest.txt').read_text(encoding='utf-8')
        assert 'fitting priorities' in text
        assert 'weighted-sandwich-known-dI' in text

    def test_parameters_csv_carries_status_and_root(self, fitted, tmp_path):
        target = tmp_path / 'joint'
        fitted.save_results(str(target))
        lines = (target / 'parameters.csv').read_text(encoding='utf-8').splitlines()
        header = lines[0].split(',')
        assert header == [
            'parameter',
            'value',
            'stderr',
            'formatted',
            'status',
            'root',
            'expression',
        ]
        rows = {line.split(',')[0]: line.split(',') for line in lines[1:]}
        assert rows['d2o.radius'][4] == 'shared'
        assert rows['h2o.radius'][5] == rows['d2o.radius'][5]

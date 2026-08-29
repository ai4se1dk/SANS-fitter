"""Behavioural tests the suite was missing (issue #21).

The existing fitting tests assert only that a result dictionary has the
expected keys, which cannot distinguish a converged fit from an optimizer that
returned its starting point. These tests fit data generated from a known model
and assert the parameters come back, on both engines, and that the surrounding
machinery (masked rows, failed configuration) behaves.
"""

import contextlib
import io
import os
import tempfile
from unittest import mock

import numpy as np
import pytest

from sans_fitter import SANSFitter, examples

ENGINES = [('bumps', 'amoeba'), ('lmfit', 'leastsq'), ('lmfit', 'least_squares')]

# Parameters every case holds fixed, matching examples.simulate's defaults.
FIXED = {'sld': 4.0, 'sld_solvent': 1.0, 'scale': 1.0, 'background': 0.001}

CASES = [
    ('sphere', {'radius': 50.0}, {'radius': 30.0}),
    ('cylinder', {'radius': 20.0, 'length': 400.0}, {'radius': 12.0, 'length': 250.0}),
]


@contextlib.contextmanager
def quiet():
    """SANSFitter reports progress on stdout; tests do not need the wall of text."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def fit_truth_case(model, truth, start, engine, method, data=None, npoints=80):
    """Fit simulated *model* data from *start* and return the result dict."""
    if data is None:
        data = examples.simulate(model, npoints=npoints, noise=0.02, seed=3, **truth, **FIXED)
    with quiet():
        fitter = SANSFitter()
        fitter.set_data(data)
        fitter.set_model(model)
        for name, value in FIXED.items():
            fitter.set_param(name, value=value, vary=False)
        for name, value in start.items():
            fitter.set_param(name, value=value, min=value / 10, max=value * 10, vary=True)
        return fitter, fitter.fit(engine=engine, method=method)


class TestTruthRecovery:
    """Every engine must actually find the generating parameters."""

    @pytest.mark.parametrize('engine,method', ENGINES)
    @pytest.mark.parametrize('model,truth,start', CASES)
    def test_parameters_are_recovered(self, model, truth, start, engine, method):
        _fitter, result = fit_truth_case(model, truth, start, engine, method)
        for name, expected in truth.items():
            assert result['parameters'][name]['value'] == pytest.approx(expected, rel=0.05), name

    @pytest.mark.parametrize('engine,method', ENGINES)
    @pytest.mark.parametrize('model,truth,start', CASES)
    def test_the_optimizer_moved_off_its_starting_point(self, model, truth, start, engine, method):
        """A zero Jacobian makes an optimizer 'converge' where it began."""
        _fitter, result = fit_truth_case(model, truth, start, engine, method)
        for name, initial in start.items():
            assert result['parameters'][name]['value'] != initial, name

    @pytest.mark.parametrize('engine,method', ENGINES)
    def test_uncertainties_are_finite_positive_and_cover_the_truth(self, engine, method):
        model, truth, start = CASES[0]
        _fitter, result = fit_truth_case(model, truth, start, engine, method)
        info = result['parameters']['radius']
        assert np.isfinite(info['stderr'])
        assert info['stderr'] > 0
        # Noise is 2%, so a stderr that lands the truth 5+ sigma away is not
        # describing this fit.
        assert abs(info['value'] - truth['radius']) < 5 * info['stderr']


class TestCrossEngineAgreement:
    """The same problem must give the same answer whichever engine runs it."""

    @classmethod
    def setup_class(cls):
        cls.model, cls.truth, cls.start = CASES[0]
        cls.npoints = 80
        cls.results = {
            f'{engine}/{method}': fit_truth_case(
                cls.model, cls.truth, cls.start, engine, method, npoints=cls.npoints
            )[1]
            for engine, method in ENGINES
        }

    def test_all_engines_agree_on_the_fitted_value(self):
        values = [r['parameters']['radius']['value'] for r in self.results.values()]
        for value in values[1:]:
            assert value == pytest.approx(values[0], rel=1e-3)

    def test_all_engines_agree_on_the_uncertainty(self):
        errors = [r['parameters']['radius']['stderr'] for r in self.results.values()]
        for stderr in errors[1:]:
            assert stderr == pytest.approx(errors[0], rel=0.05)

    def test_chisq_differs_only_by_the_degrees_of_freedom(self):
        """Documents a known discrepancy rather than hiding it.

        bumps normalizes chi-squared by the degrees of freedom; the scipy
        engine stores the raw sum of squared residuals. Same fit, ~79x apart.
        """
        dof = self.npoints - 1  # one varying parameter
        bumps = self.results['bumps/amoeba']['chisq']
        scipy = self.results['lmfit/leastsq']['chisq']
        assert scipy / dof == pytest.approx(bumps, rel=0.05)


def data_with_nan_rows():
    """Simulated data carrying NaN in I and in dI, which the loader masks out."""
    data = examples.simulate('sphere', npoints=60, noise=0.02, seed=3, radius=50.0, **FIXED)
    data.y[5] = np.nan
    data.dy[11] = np.nan
    return data


class TestMaskedData:
    """A single NaN row must not break fitting, plotting or export."""

    N_MASKED = 2

    @pytest.mark.parametrize('engine,method', ENGINES)
    def test_truth_is_still_recovered(self, engine, method):
        _fitter, result = fit_truth_case(
            'sphere', {'radius': 50.0}, {'radius': 30.0}, engine, method, data=data_with_nan_rows()
        )
        assert result['parameters']['radius']['value'] == pytest.approx(50.0, rel=0.05)

    @pytest.mark.parametrize('engine,method', ENGINES)
    def test_the_fitted_curve_covers_the_unmasked_points(self, engine, method):
        fitter, _result = fit_truth_case(
            'sphere', {'radius': 50.0}, {'radius': 30.0}, engine, method, data=data_with_nan_rows()
        )
        curve = fitter._fit_contract.require_fitted_curve()
        assert len(curve) == len(fitter.data.x) - self.N_MASKED
        assert np.all(np.isfinite(curve))

    @pytest.mark.parametrize('engine,method', ENGINES)
    def test_plotting_and_export_survive_the_masked_rows(self, engine, method):
        fitter, _result = fit_truth_case(
            'sphere', {'radius': 50.0}, {'radius': 30.0}, engine, method, data=data_with_nan_rows()
        )
        with quiet():
            figure = fitter.plot_results(show=False)
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, 'results.csv')
                fitter.save_results(path)
                with open(path) as handle:
                    lines = handle.read().splitlines()

        assert figure is not None
        rows = [line for line in lines if line and not line.startswith(('#', 'Q,'))]
        assert len(rows) == len(fitter.data.x) - self.N_MASKED
        assert all('nan' not in row.lower() for row in rows)


class TestStateAfterFailedStructureFactor:
    """A rejected set_structure_factor must leave nothing half-applied."""

    def build_fitter(self):
        data = examples.simulate('sphere', npoints=60, noise=0.02, seed=3, radius=50.0, **FIXED)
        with quiet():
            fitter = SANSFitter()
            fitter.set_data(data)
            fitter.set_model('sphere')
            fitter.set_param('radius', value=30.0, min=3.0, max=300.0, vary=True)
        return fitter

    @pytest.mark.parametrize(
        'kwargs,message',
        [
            ({'radius_effective_mode': 'invalid_mode'}, 'Invalid radius_effective_mode'),
            ({'structure_factor_name': 'not_a_structure_factor'}, 'Unsupported structure factor'),
        ],
    )
    def test_rejected_call_changes_nothing(self, kwargs, message):
        fitter = self.build_fitter()
        before_params = {name: dict(info) for name, info in fitter.params.items()}
        kwargs = {'structure_factor_name': 'hardsphere'} | kwargs

        with pytest.raises(ValueError, match=message), quiet():
            fitter.set_structure_factor(**kwargs)

        assert fitter.model_name == 'sphere'
        assert fitter.get_structure_factor() is None
        assert fitter.params == before_params
        assert 'radius_effective' not in fitter.params

    def test_the_kernel_is_not_left_as_the_product_model(self):
        """Otherwise the next fit silently evaluates the product kernel."""
        fitter = self.build_fitter()
        with pytest.raises(ValueError), quiet():
            fitter.set_structure_factor('hardsphere', radius_effective_mode='invalid_mode')

        with quiet():
            after = fitter.fit(engine='bumps', method='amoeba')
        reference = self.build_fitter()
        with quiet():
            expected = reference.fit(engine='bumps', method='amoeba')

        assert after['chisq'] == pytest.approx(expected['chisq'], rel=1e-6)
        assert after['parameters']['radius']['value'] == pytest.approx(
            expected['parameters']['radius']['value'], rel=1e-6
        )

    def test_a_failed_removal_leaves_the_product_model_intact(self):
        """The mirror case: kernel and params must not disagree either way."""
        fitter = self.build_fitter()
        with quiet():
            fitter.set_structure_factor('hardsphere')
        product_kernel = fitter.kernel
        product_params = {name: dict(info) for name, info in fitter.params.items()}

        with mock.patch.object(
            fitter._param_manager, 'remove_structure_factor', side_effect=RuntimeError('boom')
        ):
            with pytest.raises(ValueError), quiet():
                fitter.remove_structure_factor()

        assert fitter.kernel is product_kernel
        assert fitter.params == product_params
        assert fitter.get_structure_factor() == 'hardsphere'

    def test_a_later_legitimate_cycle_still_works(self):
        fitter = self.build_fitter()
        before_params = {name: dict(info) for name, info in fitter.params.items()}
        with pytest.raises(ValueError), quiet():
            fitter.set_structure_factor('hardsphere', radius_effective_mode='invalid_mode')

        with quiet():
            fitter.set_structure_factor('hardsphere')
            assert 'radius_effective' in fitter.params
            fitter.remove_structure_factor()

        assert fitter.get_structure_factor() is None
        assert fitter.params == before_params

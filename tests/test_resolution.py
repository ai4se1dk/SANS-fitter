"""Explicit resolution / smearing control (issue #75).

The tests assert the *sasmodels resolution class* through the public seam —
``type(DirectModel(evaluation_data, kernel).resolution)`` — rather than
re-implementing ``_interpret_data``'s branch logic. sasmodels is the oracle
here; there is no need to compare against a live SasView.
"""

import contextlib
import io
import os
import tempfile
import warnings

import numpy as np
import pytest
from sasmodels.core import load_model
from sasmodels.direct_model import DirectModel
from sasmodels.resolution import Perfect1D, Pinhole1D, Slit1D

from sans_fitter import SANSFitter, examples
from sans_fitter.data.loader import load_sans_data, normalize_sans_data
from sans_fitter.data.resolution import (
    RESOLUTION_MODES,
    ResolutionSetting,
    apply_resolution,
    validate_resolution,
)

from .helpers import (
    create_loading_test_data_file,
    create_loading_test_data_file_with_resolution,
)

FIXED = {'sld': 4.0, 'sld_solvent': 1.0, 'scale': 1.0, 'background': 0.001}


@contextlib.contextmanager
def quiet():
    """SANSFitter reports progress on stdout; these tests do not need it."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


@pytest.fixture(scope='module')
def sphere_kernel():
    return load_model('sphere', dtype='single', platform='dll')


def resolution_class(data, kernel, mode, **kwargs):
    """Return the sasmodels resolution class *mode* selects for *data*."""
    setting = validate_resolution(mode, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        evaluation_data = apply_resolution(data, setting)
    return type(DirectModel(evaluation_data, kernel).resolution)


@pytest.fixture
def data_no_resolution():
    """A loaded file with no dQ column — sasdata zero-fills dx, it is not None."""
    path = create_loading_test_data_file(num_points=20)
    try:
        data = load_sans_data(path)
    finally:
        os.unlink(path)
    assert data.dx is not None, 'precondition: sasdata zero-fills the absent dQ column'
    return data


@pytest.fixture
def data_with_pinhole():
    """A loaded file carrying a real dQ column."""
    path = create_loading_test_data_file_with_resolution(num_points=20)
    try:
        data = load_sans_data(path)
    finally:
        os.unlink(path)
    return data


@pytest.fixture
def data_with_slit(data_no_resolution):
    """A USANS-shaped dataset: real slit columns, only a zero-filled dx."""
    data_no_resolution.dxl = np.full(data_no_resolution.x.size, 0.05)
    data_no_resolution.dxw = np.full(data_no_resolution.x.size, 0.005)
    return data_no_resolution


class TestDispatch:
    """Each mode must produce the sasmodels resolution class it claims."""

    def test_none_on_a_file_with_real_dq_is_perfect(self, data_with_pinhole, sphere_kernel):
        assert resolution_class(data_with_pinhole, sphere_kernel, 'none') is Perfect1D

    def test_slit_wins_over_a_present_dx_column(self, data_with_pinhole, sphere_kernel):
        """The priority trap: while dx is non-None the slit branch is unreachable."""
        cls = resolution_class(data_with_pinhole, sphere_kernel, 'slit', slit_length=0.05)
        assert cls is Slit1D

    def test_slit_wins_over_a_zero_filled_dx_column(self, data_no_resolution, sphere_kernel):
        """The common case: every sasdata-loaded file has a non-None dx."""
        cls = resolution_class(data_no_resolution, sphere_kernel, 'slit', slit_length=0.05)
        assert cls is Slit1D

    def test_a_slit_width_equal_to_the_lowest_q_is_refused_with_a_readable_error(
        self, data_no_resolution
    ):
        """sasmodels would raise OverflowError from inside its grid extension."""
        q_min = float(np.min(data_no_resolution.x))
        setting = validate_resolution('slit', slit_length=0.05, slit_width=q_min)
        with pytest.raises(ValueError, match='smallest fitted'):
            apply_resolution(data_no_resolution, setting)

    def test_a_file_carrying_only_a_slit_width_is_refused_with_a_readable_error(
        self, data_no_resolution
    ):
        """sasmodels raises NotImplementedError for a width with no length."""
        data_no_resolution.dxw = np.full(data_no_resolution.x.size, 0.001)
        with pytest.raises(ValueError, match='slit width with no slit length'):
            apply_resolution(data_no_resolution, ResolutionSetting())

    def test_pinhole_smears_a_file_that_has_no_dq_column(self, data_no_resolution, sphere_kernel):
        cls = resolution_class(data_no_resolution, sphere_kernel, 'pinhole', dq_over_q=0.1)
        assert cls is Pinhole1D

    def test_data_uses_a_real_dq_column(self, data_with_pinhole, sphere_kernel):
        assert resolution_class(data_with_pinhole, sphere_kernel, 'data') is Pinhole1D

    def test_data_uses_slit_columns_when_there_is_no_real_dq(self, data_with_slit, sphere_kernel):
        """The USANS case, which fits silently unsmeared without the dx clearing."""
        assert resolution_class(data_with_slit, sphere_kernel, 'data') is Slit1D

    def test_data_prefers_pinhole_and_warns_when_both_are_present(
        self, data_with_pinhole, sphere_kernel
    ):
        data_with_pinhole.dxl = np.full(data_with_pinhole.x.size, 0.05)
        with pytest.warns(UserWarning, match='slit columns'):
            evaluation_data = apply_resolution(data_with_pinhole, ResolutionSetting())
        assert type(DirectModel(evaluation_data, sphere_kernel).resolution) is Pinhole1D

    def test_data_without_any_resolution_warns_and_degrades(
        self, data_no_resolution, sphere_kernel
    ):
        with pytest.warns(UserWarning, match='no resolution columns'):
            evaluation_data = apply_resolution(data_no_resolution, ResolutionSetting())
        assert type(DirectModel(evaluation_data, sphere_kernel).resolution) is Perfect1D

    def test_pinhole_writes_sigma_q_on_the_copy_only(self, data_no_resolution):
        original_dx = np.array(data_no_resolution.dx, copy=True)
        setting = validate_resolution('pinhole', dq_over_q=0.1)
        evaluation_data = apply_resolution(data_no_resolution, setting)

        np.testing.assert_allclose(evaluation_data.dx, np.asarray(data_no_resolution.x) * 0.1)
        np.testing.assert_array_equal(data_no_resolution.dx, original_dx)

    def test_slit_writes_constant_columns_on_the_copy_only(self, data_no_resolution):
        setting = validate_resolution('slit', slit_length=0.05, slit_width=0.002)
        evaluation_data = apply_resolution(data_no_resolution, setting)

        n_points = data_no_resolution.x.size
        assert evaluation_data.dx is None
        np.testing.assert_allclose(evaluation_data.dxl, np.full(n_points, 0.05))
        np.testing.assert_allclose(evaluation_data.dxw, np.full(n_points, 0.002))
        assert data_no_resolution.dx is not None
        assert data_no_resolution.dxl is None

    def test_none_nulls_the_columns_rather_than_zeroing_them(self, data_with_pinhole):
        """Zeroing would leave the copy claiming a dQ column the file has."""
        evaluation_data = apply_resolution(data_with_pinhole, validate_resolution('none'))
        assert evaluation_data.dx is None
        assert evaluation_data.dxl is None
        assert evaluation_data.dxw is None

    def test_the_copy_preserves_the_fit_index(self, data_with_pinhole):
        """qmin/qmax/mask must survive: _interpret_data builds its index from them."""
        data_with_pinhole.qmin = float(data_with_pinhole.x[2])
        data_with_pinhole.qmax = float(data_with_pinhole.x[-3])
        data_with_pinhole.mask[1] = True

        evaluation_data = apply_resolution(
            data_with_pinhole, validate_resolution('pinhole', dq_over_q=0.1)
        )
        assert evaluation_data.qmin == data_with_pinhole.qmin
        assert evaluation_data.qmax == data_with_pinhole.qmax
        np.testing.assert_array_equal(evaluation_data.mask, data_with_pinhole.mask)

    def test_the_copy_does_not_share_arrays_with_the_original(self, data_with_pinhole):
        evaluation_data = apply_resolution(data_with_pinhole, ResolutionSetting())
        assert evaluation_data is not data_with_pinhole
        assert evaluation_data.dx is not data_with_pinhole.dx
        assert evaluation_data.y is not data_with_pinhole.y

    def test_sesans_data_is_left_alone(self, data_with_pinhole):
        """SESANS takes a different branch of _interpret_data entirely."""
        data_with_pinhole.isSesans = True
        evaluation_data = apply_resolution(data_with_pinhole, validate_resolution('none'))
        np.testing.assert_array_equal(evaluation_data.dx, data_with_pinhole.dx)


class TestValidation:
    """A rejected call must raise and leave the fitter exactly as it was."""

    def test_every_mode_is_accepted(self):
        assert RESOLUTION_MODES == ('data', 'none', 'pinhole', 'slit')
        assert validate_resolution('data').mode == 'data'
        assert validate_resolution('none').mode == 'none'
        assert validate_resolution('pinhole', dq_over_q=0.1).dq_over_q == 0.1
        assert validate_resolution('slit', slit_length=0.05).slit_length == 0.05

    def test_unknown_mode_lists_the_alternatives(self):
        with pytest.raises(ValueError, match="Unknown resolution mode 'gaussian'"):
            validate_resolution('gaussian')

    @pytest.mark.parametrize(
        'mode,kwargs',
        [
            ('data', {'dq_over_q': 0.1}),
            ('data', {'slit_length': 0.05}),
            ('none', {'dq_over_q': 0.1}),
            ('none', {'slit_width': 0.05}),
            ('pinhole', {'dq_over_q': 0.1, 'slit_length': 0.05}),
            ('pinhole', {'dq_over_q': 0.1, 'slit_width': 0.001}),
            ('slit', {'slit_length': 0.05, 'dq_over_q': 0.1}),
        ],
    )
    def test_arguments_that_do_not_belong_to_the_mode_are_rejected(self, mode, kwargs):
        with pytest.raises(ValueError, match='not accepted by resolution mode'):
            validate_resolution(mode, **kwargs)

    def test_pinhole_requires_a_width(self):
        with pytest.raises(ValueError, match='requires dq_over_q'):
            validate_resolution('pinhole')

    def test_slit_requires_a_length(self):
        """sasmodels cannot smear from a width alone; it raises
        NotImplementedError deep inside its weight-matrix construction."""
        with pytest.raises(ValueError, match='requires slit_length'):
            validate_resolution('slit')

    def test_a_width_only_slit_is_rejected_rather_than_reaching_sasmodels(self):
        with pytest.raises(ValueError, match='requires slit_length'):
            validate_resolution('slit', slit_width=0.001)

    def test_slit_width_is_optional(self):
        setting = validate_resolution('slit', slit_length=0.05)
        assert setting.slit_length == 0.05
        assert setting.slit_width is None

    @pytest.mark.parametrize('bad', [0.0, -0.1])
    def test_pinhole_width_must_be_positive(self, bad):
        with pytest.raises(ValueError, match='greater than zero'):
            validate_resolution('pinhole', dq_over_q=bad)

    @pytest.mark.parametrize('bad', [float('nan'), float('inf')])
    def test_pinhole_width_must_be_finite(self, bad):
        with pytest.raises(ValueError, match='must be finite'):
            validate_resolution('pinhole', dq_over_q=bad)

    @pytest.mark.parametrize('bad', [0.0, -0.05])
    def test_slit_length_must_be_positive(self, bad):
        with pytest.raises(ValueError, match='greater than zero'):
            validate_resolution('slit', slit_length=bad)

    def test_negative_slit_width_is_rejected(self):
        with pytest.raises(ValueError, match='must not be negative'):
            validate_resolution('slit', slit_length=0.05, slit_width=-0.001)

    def test_a_rejected_call_leaves_the_fitter_unchanged(self):
        fitter = SANSFitter()
        with quiet():
            fitter.set_resolution('pinhole', dq_over_q=0.1)
        before = fitter.get_resolution()

        with pytest.raises(ValueError):
            fitter.set_resolution('slit', dq_over_q=0.2)
        with pytest.raises(ValueError):
            fitter.set_resolution('nonsense')

        assert fitter.get_resolution() == before


class TestFitterApi:
    def test_the_default_is_data_mode(self):
        assert SANSFitter().get_resolution() == {
            'mode': 'data',
            'dq_over_q': None,
            'slit_length': None,
            'slit_width': None,
        }

    def test_get_resolution_works_before_any_data_is_loaded(self):
        """The mode is fitter state; the evaluation copy is built at fit time."""
        fitter = SANSFitter()
        with quiet():
            fitter.set_resolution('pinhole', dq_over_q=0.05)
        assert fitter.data is None
        assert fitter.get_resolution()['dq_over_q'] == 0.05

    def test_get_resolution_returns_a_copy(self):
        fitter = SANSFitter()
        returned = fitter.get_resolution()
        returned['mode'] = 'tampered'
        assert fitter.get_resolution()['mode'] == 'data'

    def test_the_mode_survives_load_data(self):
        """Resolution is an analysis choice, like the model and the parameters."""
        fitter = SANSFitter()
        path = create_loading_test_data_file(num_points=20)
        try:
            with quiet():
                fitter.set_resolution('pinhole', dq_over_q=0.07)
                fitter.load_data(path)
        finally:
            os.unlink(path)
        assert fitter.get_resolution() == {
            'mode': 'pinhole',
            'dq_over_q': 0.07,
            'slit_length': None,
            'slit_width': None,
        }

    def test_the_mode_survives_set_data(self):
        fitter = SANSFitter()
        with quiet():
            fitter.set_resolution('slit', slit_length=0.05)
            fitter.set_data(examples.simulate('sphere', npoints=20, seed=1))
        assert fitter.get_resolution()['mode'] == 'slit'

    def test_the_load_summary_reports_the_active_mode(self):
        """A stale custom width must not reach a new dataset unnoticed."""
        fitter = SANSFitter()
        path = create_loading_test_data_file(num_points=20)
        stream = io.StringIO()
        try:
            with quiet():
                fitter.set_resolution('pinhole', dq_over_q=0.07)
            with contextlib.redirect_stdout(stream):
                fitter.load_data(path)
        finally:
            os.unlink(path)
        output = stream.getvalue()
        assert 'Resolution mode:' in output
        assert 'pinhole' in output
        assert '0.07' in output

    def test_the_fit_reports_the_active_mode_before_the_engine_runs(self):
        fitter = _sphere_fitter(examples.simulate('sphere', radius=50.0, npoints=40, seed=3))
        stream = io.StringIO()
        with quiet():
            fitter.set_resolution('pinhole', dq_over_q=0.05)
        with contextlib.redirect_stdout(stream):
            fitter.fit(engine='bumps', method='amoeba')
        output = stream.getvalue()
        assert 'Resolution: pinhole' in output
        assert output.index('Resolution: pinhole') < output.index('Initial')


def _sphere_fitter(data, radius_start=30.0):
    """A one-parameter sphere fit, the smallest thing that actually converges."""
    fitter = SANSFitter()
    with quiet():
        fitter.set_data(data)
        fitter.set_model('sphere')
        for name, value in FIXED.items():
            fitter.set_param(name, value=value, vary=False)
        fitter.set_param('radius', value=radius_start, min=3.0, max=300.0, vary=True)
    return fitter


class TestBehaviour:
    """Smearing must actually change the curve and the fit, not just the class."""

    def test_pinhole_fills_in_the_form_factor_minima(self, sphere_kernel):
        """A shape assertion: smearing washes out the sharp sphere minima."""
        data = examples.simulate('sphere', radius=50.0, npoints=200, noise=0.0, seed=1, **FIXED)
        pars = dict(FIXED, radius=50.0)

        sharp = DirectModel(apply_resolution(data, validate_resolution('none')), sphere_kernel)(
            **pars
        )
        smeared = DirectModel(
            apply_resolution(data, validate_resolution('pinhole', dq_over_q=0.2)), sphere_kernel
        )(**pars)

        # The deepest minimum of the sharp curve is filled in by smearing.
        trough = int(np.argmin(sharp))
        assert smeared[trough] > sharp[trough] * 2

    def test_data_and_a_matching_pinhole_are_the_same_smearing(self, sphere_kernel):
        """dx = dq·q is exactly what simulate(dq=...) attaches."""
        data = examples.simulate('sphere', radius=50.0, npoints=60, noise=0.0, seed=1, dq=0.1)
        pars = dict(FIXED, radius=50.0)

        from_columns = DirectModel(
            apply_resolution(data, validate_resolution('data')), sphere_kernel
        )(**pars)
        from_setting = DirectModel(
            apply_resolution(data, validate_resolution('pinhole', dq_over_q=0.1)), sphere_kernel
        )(**pars)

        np.testing.assert_allclose(from_columns, from_setting)

    def test_truth_is_recovered_from_smeared_data_under_the_default_mode(self):
        data = examples.simulate(
            'sphere', radius=50.0, npoints=80, noise=0.02, seed=3, dq=0.1, **FIXED
        )
        fitter = _sphere_fitter(data)
        with quiet():
            result = fitter.fit(engine='bumps', method='amoeba')
        assert result['parameters']['radius']['value'] == pytest.approx(50.0, rel=0.05)

    def test_ignoring_real_smearing_fits_worse(self):
        """'none' on smeared data must cost χ², or smearing is doing nothing."""
        data = examples.simulate(
            'sphere', radius=50.0, npoints=80, noise=0.02, seed=3, dq=0.1, **FIXED
        )
        smeared_fitter = _sphere_fitter(data)
        sharp_fitter = _sphere_fitter(data)
        with quiet():
            smeared = smeared_fitter.fit(engine='bumps', method='amoeba')
            sharp_fitter.set_resolution('none')
            sharp = sharp_fitter.fit(engine='bumps', method='amoeba')

        assert sharp['chisq'] > smeared['chisq'] * 1.5

    def test_component_curves_are_smeared_like_the_total(self):
        """Otherwise plot_results(show_components=True) draws sharp parts
        under a smeared total."""
        truth = {
            'A_scale': 10.0,
            'A_cor_length': 50.0,
            'B_scale': 5.0,
            'B_peak_pos': 0.1,
            'B_peak_hwhm': 0.01,
            'scale': 1.0,
            'background': 0.01,
        }
        data = examples.simulate('dab+peak_lorentz', npoints=60, noise=0.02, seed=2, **truth)

        curves = {}
        for mode, kwargs in [('none', {}), ('pinhole', {'dq_over_q': 0.3})]:
            fitter = SANSFitter()
            with quiet():
                fitter.set_data(data)
                fitter.set_models('dab', 'peak_lorentz')
                fitter.set_resolution(mode, **kwargs)
                fitter.set_param('dab_cor_length', value=50.0, vary=True)
                fitter.set_param('peak_lorentz_peak_pos', value=0.1, vary=True)
                fitter.fit(engine='bumps', method='amoeba')
            curves[mode] = fitter._fit_contract.artifacts.component_curves

        assert set(curves['pinhole']) == {'dab', 'peak_lorentz'}
        # The Lorentzian peak is the component smearing visibly changes.
        sharp_peak = curves['none']['peak_lorentz']
        smeared_peak = curves['pinhole']['peak_lorentz']
        assert len(smeared_peak) == len(sharp_peak)
        assert np.max(smeared_peak) < np.max(sharp_peak)


class TestNeverMutatesTheUsersData:
    def test_a_full_round_trip_leaves_every_column_untouched(self):
        data = examples.simulate(
            'sphere', radius=50.0, npoints=60, noise=0.02, seed=3, dq=0.1, **FIXED
        )
        fitter = _sphere_fitter(data)
        dx_before = np.array(fitter.data.dx, copy=True)
        dx_identity = fitter.data.dx

        with quiet():
            fitter.set_resolution('slit', slit_length=0.05, slit_width=0.01)
            fitter.fit(engine='bumps', method='amoeba')
            fitter.plot_results(show=False)

        np.testing.assert_array_equal(fitter.data.dx, dx_before)
        assert fitter.data.dx is dx_identity
        assert fitter.data.dxl is None
        assert fitter.data.dxw is None

    def test_pinhole_does_not_write_dx_onto_a_dataset_that_had_none(self):
        data = examples.simulate('sphere', radius=50.0, npoints=60, noise=0.02, seed=3, **FIXED)
        fitter = _sphere_fitter(data)
        with quiet():
            fitter.set_resolution('pinhole', dq_over_q=0.1)
            fitter.fit(engine='lmfit', method='leastsq')
        assert not np.any(np.nan_to_num(np.asarray(fitter.data.dx, dtype=float)))


class TestReporting:
    def test_the_setting_is_stamped_on_the_fit_result_contract(self):
        data = examples.simulate('sphere', radius=50.0, npoints=40, noise=0.02, seed=3, **FIXED)
        fitter = _sphere_fitter(data)
        with quiet():
            fitter.set_resolution('pinhole', dq_over_q=0.08)
            fitter.fit(engine='bumps', method='amoeba')
        assert fitter._fit_contract.resolution == {
            'mode': 'pinhole',
            'dq_over_q': 0.08,
            'slit_length': None,
            'slit_width': None,
        }

    def test_the_saved_csv_records_the_resolution_that_was_used(self):
        data = examples.simulate(
            'sphere', radius=50.0, npoints=40, noise=0.02, seed=3, dq=0.1, **FIXED
        )
        fitter = _sphere_fitter(data)
        with quiet():
            fitter.set_resolution('pinhole', dq_over_q=0.08)
            fitter.fit(engine='bumps', method='amoeba')

        handle = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        handle.close()
        try:
            with quiet():
                fitter.save_results(handle.name)
            with open(handle.name) as f:
                content = f.read()
        finally:
            os.unlink(handle.name)

        assert '# Resolution mode: pinhole' in content
        # Written to a file, not a console: the description must not carry a
        # glyph chosen from what sys.stdout happened to be able to encode.
        header = next(line for line in content.splitlines() if 'Resolution mode' in line)
        assert header.isascii(), header

    @pytest.mark.parametrize(
        'mode,kwargs',
        [
            ('data', {}),
            ('none', {}),
            ('pinhole', {'dq_over_q': 0.1}),
            ('slit', {'slit_length': 0.05, 'slit_width': 0.002}),
        ],
    )
    def test_every_mode_has_an_ascii_description_for_file_output(self, mode, kwargs):
        setting = validate_resolution(mode, **kwargs)
        assert setting.describe(ascii_only=True).isascii()


class TestDataOpsAndSimulatedDatasets:
    def test_an_in_memory_dataset_without_columns_can_be_smeared(self):
        """set_data datasets have dx=None, not a zero-filled array."""
        data = normalize_sans_data(examples.simulate('sphere', npoints=30, seed=1))
        assert data.dx is None
        setting = validate_resolution('pinhole', dq_over_q=0.1)
        evaluation_data = apply_resolution(data, setting)
        np.testing.assert_allclose(evaluation_data.dx, np.asarray(data.x) * 0.1)
        assert data.dx is None

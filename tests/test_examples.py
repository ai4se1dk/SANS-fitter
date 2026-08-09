"""Tests for the bundled example datasets and the data simulator (issue #53).

The registry test is deliberately exhaustive: every entry is loaded and its
model and parameter names are checked against sasmodels. The example files live
inside the installed ``sasdata`` package rather than in this repository, so a
sasdata upgrade that renames or drops a file would otherwise break users
silently. These tests are the canary for that.
"""

import contextlib
import io

import numpy as np
import pytest
from sasmodels.core import load_model

from sans_fitter import examples
from sans_fitter.data_loader import _has_real_data

ALL_EXAMPLES = examples.list_examples()


def _valid_parameter_names(model_name: str) -> set:
    """Return every parameter name sasmodels accepts for *model_name*."""
    kernel = load_model(model_name, dtype='single', platform='dll')
    return {p.name for p in kernel.info.parameters.call_parameters}


# =============================================================================
# Registry integrity
# =============================================================================


class TestRegistry:
    def test_registry_is_not_empty(self):
        assert len(ALL_EXAMPLES) > 0

    @pytest.mark.parametrize('name', ALL_EXAMPLES)
    def test_name_matches_key(self, name):
        assert examples.get_example(name).name == name

    @pytest.mark.parametrize('name', ALL_EXAMPLES)
    def test_file_exists(self, name):
        """The file is present in the installed sasdata package."""
        assert examples.example_path(name).endswith(examples.get_example(name).filename)

    @pytest.mark.parametrize('name', ALL_EXAMPLES)
    def test_loads_as_fit_ready_data(self, name):
        data = examples.load(name)
        assert len(data.x) > 0
        assert len(data.x) == len(data.y)
        assert data.qmin is not None and data.qmax is not None
        assert data.qmin <= data.qmax
        assert data.mask is not None
        assert np.all(np.asarray(data.x)[~np.asarray(data.mask)] > 0)

    @pytest.mark.parametrize('name', ALL_EXAMPLES)
    def test_declared_model_loads(self, name):
        example = examples.get_example(name)
        assert load_model(example.model, dtype='single', platform='dll') is not None

    @pytest.mark.parametrize('name', ALL_EXAMPLES)
    def test_suggested_parameters_exist_on_model(self, name):
        """Every suggested parameter is real, so load_fitter cannot KeyError."""
        example = examples.get_example(name)
        valid = _valid_parameter_names(example.model)
        unknown = set(example.params) - valid
        assert not unknown, f"{name}: parameters not in model '{example.model}': {unknown}"

    @pytest.mark.parametrize('name', ALL_EXAMPLES)
    def test_parameter_bounds_bracket_the_start(self, name):
        example = examples.get_example(name)
        for param, settings in example.params.items():
            value = settings.get('value')
            if value is None:
                continue
            if 'min' in settings:
                assert settings['min'] <= value, f'{name}.{param}: value below min'
            if 'max' in settings:
                assert value <= settings['max'], f'{name}.{param}: value above max'

    @pytest.mark.parametrize('name', ALL_EXAMPLES)
    def test_description_is_meaningful(self, name):
        assert len(examples.get_example(name).description) > 40

    def test_unknown_example_raises_with_available_names(self):
        with pytest.raises(KeyError) as exc:
            examples.get_example('no_such_example')
        assert 'cylinder' in str(exc.value)

    def test_list_examples_filters_by_tag(self):
        measured = examples.list_examples(tag='measured')
        assert measured
        assert set(measured) < set(ALL_EXAMPLES)
        for name in measured:
            assert 'measured' in examples.get_example(name).tags

    def test_list_examples_unknown_tag_is_empty(self):
        assert examples.list_examples(tag='not-a-tag') == []


class TestCuratedClaims:
    """The descriptions make specific promises; check they hold."""

    def test_cylinder_truth_reproduces_the_data_exactly(self):
        """The headline claim for the verification dataset."""
        from sasmodels.direct_model import DirectModel

        data = examples.load('cylinder')
        truth = examples.get_example('cylinder').truth
        kernel = load_model('cylinder', dtype='single', platform='dll')
        computed = np.asarray(DirectModel(data, kernel)(**truth))
        np.testing.assert_allclose(computed, data.y, rtol=1e-4)

    def test_format_pair_holds_identical_measurements(self):
        """canSAS XML and NXcanSAS HDF5 must decode to the same measurement.

        Tolerance is 1e-4, not exact: the XML file stores Q to six significant
        figures as text while the HDF5 keeps full binary precision, so the two
        agree to about 3e-6 and no closer. That is a property of the files, not
        of the loader.
        """
        xml = examples.load('canSAS_xml')
        hdf5 = examples.load('nxcanSAS_h5')
        assert len(xml.x) == len(hdf5.x)
        np.testing.assert_allclose(xml.x, hdf5.x, rtol=1e-4)
        np.testing.assert_allclose(xml.y, hdf5.y, rtol=1e-4)

    def test_resolution_pair_differs_only_in_dq(self):
        """'sphere' has no dQ, 'sphere_smeared' does, on the same Q grid."""
        plain = examples.load('sphere')
        smeared = examples.load('sphere_smeared')
        np.testing.assert_allclose(plain.x, smeared.x, rtol=1e-6)
        assert not _has_real_data(plain.dx)
        assert _has_real_data(smeared.dx)

    @pytest.mark.parametrize('name', examples.list_examples(tag='measured'))
    def test_measured_data_claims_no_truth(self, name):
        """Ground truth is only claimed where it genuinely exists."""
        assert examples.get_example(name).truth is None


# =============================================================================
# load_fitter presets
# =============================================================================


class TestLoadFitter:
    @pytest.mark.parametrize('name', ALL_EXAMPLES)
    def test_preset_is_configured(self, name):
        example = examples.get_example(name)
        fitter = examples.load_fitter(name)

        assert fitter.data is not None
        assert fitter.kernel is not None
        assert example.model in (fitter.model_name or '')

        for param, settings in example.params.items():
            if 'value' in settings:
                assert fitter.params[param]['value'] == pytest.approx(settings['value'])
            if 'vary' in settings:
                assert fitter.params[param]['vary'] == settings['vary']

    @pytest.mark.parametrize('name', ALL_EXAMPLES)
    def test_preset_has_something_to_fit(self, name):
        fitter = examples.load_fitter(name)
        assert any(info['vary'] for info in fitter.params.values())

    def test_structure_factor_is_applied(self):
        fitter = examples.load_fitter('sds_micelles')
        assert fitter.get_structure_factor() == 'hayter_msa'

    def test_polydispersity_is_applied(self):
        fitter = examples.load_fitter('polydisperse_spheres')
        assert fitter.is_polydispersity_enabled()
        assert fitter.get_pd_param('radius')['pd'] == pytest.approx(0.15)

    def test_quiet_by_default(self, capsys):
        examples.load_fitter('cylinder')
        assert capsys.readouterr().out == ''

    def test_verbose_when_asked(self, capsys):
        examples.load_fitter('cylinder', quiet=False)
        assert 'cylinder' in capsys.readouterr().out


class TestPresetsConverge:
    """The presets must actually reach a fit with the default engine.

    'cylinder' is excluded by design — it has no dI column, so bumps refuses
    it. That restriction is recorded in the example's `notes` and asserted
    below rather than being quietly skipped.
    """

    FITTABLE = [n for n in ALL_EXAMPLES if n != 'cylinder']

    @pytest.mark.slow
    @pytest.mark.parametrize('name', FITTABLE)
    def test_fits_with_default_engine(self, name):
        fitter = examples.load_fitter(name)
        with contextlib.redirect_stdout(io.StringIO()):
            result = fitter.fit(engine='bumps', method='amoeba')
        assert np.isfinite(result['chisq'])
        assert result['parameters']

    def test_cylinder_documents_its_engine_restriction(self):
        example = examples.get_example('cylinder')
        assert 'bumps' in example.notes
        assert not _has_real_data(examples.load('cylinder').dy)


# =============================================================================
# describe()
# =============================================================================


class TestDescribe:
    def test_overview_lists_every_example(self, capsys):
        examples.describe()
        out = capsys.readouterr().out
        for name in ALL_EXAMPLES:
            assert name in out

    def test_detail_reports_live_file_facts(self, capsys):
        examples.describe('silica_spheres')
        out = capsys.readouterr().out
        assert 'Ludox_silica.xml' in out
        assert 'Q range' in out
        assert 'points' in out

    def test_detail_shows_notes(self, capsys):
        examples.describe('cylinder')
        assert 'NOTE:' in capsys.readouterr().out

    def test_summary_does_not_split_on_a_decimal_point(self, capsys):
        """'0.2 M NaCl' must not be truncated mid-number in the overview."""
        examples.describe()
        for line in capsys.readouterr().out.splitlines():
            assert not line.rstrip().endswith(' with 0')


# =============================================================================
# simulate()
# =============================================================================


class TestSimulate:
    def test_returns_fit_ready_data(self):
        data = examples.simulate('sphere', npoints=40)
        assert len(data.x) == 40
        assert data.qmin is not None and data.qmax is not None
        assert data.mask is not None
        assert np.all(data.x > 0)

    def test_truth_records_the_full_parameter_set(self):
        data = examples.simulate('sphere', radius=42.0)
        assert data.truth['radius'] == 42.0
        # Unspecified parameters are filled from the model defaults, so `truth`
        # is complete rather than only what the caller passed.
        assert 'sld_solvent' in data.truth
        assert 'background' in data.truth

    def test_is_reproducible_for_a_fixed_seed(self):
        a = examples.simulate('sphere', seed=7)
        b = examples.simulate('sphere', seed=7)
        np.testing.assert_array_equal(a.y, b.y)

    def test_different_seeds_differ(self):
        a = examples.simulate('sphere', seed=1)
        b = examples.simulate('sphere', seed=2)
        assert not np.array_equal(a.y, b.y)

    def test_noise_zero_is_exact(self):
        data = examples.simulate('sphere', noise=0)
        assert np.all(data.dy == 0)

    def test_uncertainties_do_not_collapse_in_the_minima(self):
        """The regression that made a simulated sphere fit to the wrong radius.

        A purely relative dI drives the uncertainty to ~0 where the form factor
        dips, giving those points runaway weight. Counting statistics keep the
        smallest dI within a few decades of the largest.
        """
        data = examples.simulate('sphere', radius=50, scale=0.1, npoints=80, noise=0.02)
        positive = data.dy[data.dy > 0]
        assert positive.min() > 0
        assert positive.max() / positive.min() < 1e3

    def test_resolution_is_attached_and_smears(self):
        sharp = examples.simulate('sphere', radius=50, noise=0, npoints=80)
        smeared = examples.simulate('sphere', radius=50, noise=0, npoints=80, dq=0.1)
        assert _has_real_data(smeared.dx)
        assert not _has_real_data(sharp.dx)
        # Smearing fills in the form-factor minima, so the deepest dip rises.
        assert smeared.y.min() > sharp.y.min()

    def test_explicit_q_grid_is_used(self):
        q = np.linspace(0.01, 0.2, 25)
        data = examples.simulate('sphere', q=q)
        np.testing.assert_allclose(data.x, q)

    def test_a_bare_pd_width_actually_applies(self):
        """sasmodels ignores a _pd width with no _pd_n; simulate must not."""
        mono = examples.simulate('sphere', radius=50, noise=0, npoints=60)
        poly = examples.simulate('sphere', radius=50, radius_pd=0.2, noise=0, npoints=60)
        assert not np.allclose(mono.y, poly.y)

    def test_pd_companion_defaults_are_recorded_in_truth(self):
        poly = examples.simulate('sphere', radius=50, radius_pd=0.2, noise=0, npoints=20)
        assert poly.truth['radius_pd_n'] == 35
        assert poly.truth['radius_pd_type'] == 'gaussian'

    def test_explicit_pd_settings_are_not_overridden(self):
        poly = examples.simulate(
            'sphere', radius=50, radius_pd=0.2, radius_pd_type='schulz', noise=0, npoints=20
        )
        assert poly.truth['radius_pd_type'] == 'schulz'

    def test_zero_pd_width_stays_monodisperse(self):
        mono = examples.simulate('sphere', radius=50, noise=0, npoints=40)
        zero_pd = examples.simulate('sphere', radius=50, radius_pd=0.0, noise=0, npoints=40)
        np.testing.assert_allclose(mono.y, zero_pd.y)

    def test_product_model_works(self):
        data = examples.simulate('sphere@hardsphere', radius=50, noise=0, npoints=30)
        assert len(data.x) == 30

    @pytest.mark.parametrize(
        'kwargs',
        [
            {'qmin': 0},
            {'qmin': -1},
            {'qmin': 0.5, 'qmax': 0.1},
            {'npoints': 1},
            {'noise': -0.1},
            {'q': np.array([])},
            {'q': np.array([0.1, -0.2])},
        ],
    )
    def test_invalid_arguments_raise_valueerror(self, kwargs):
        with pytest.raises(ValueError):
            examples.simulate('sphere', **kwargs)

    def test_unknown_model_raises_valueerror(self):
        with pytest.raises(ValueError, match='Failed to load model'):
            examples.simulate('definitely_not_a_model')

    def test_unknown_parameter_is_rejected_by_name(self):
        with pytest.raises(ValueError, match='not valid for model'):
            examples.simulate('sphere', not_a_parameter=1.0)


class TestSimulateRoundTrip:
    """Simulated data must be recoverable — this is what makes it teachable."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        'model,truth,start',
        [
            ('sphere', {'radius': 50.0}, {'radius': 30.0}),
            ('cylinder', {'radius': 20.0, 'length': 400.0}, {'radius': 15.0, 'length': 300.0}),
        ],
    )
    def test_fit_recovers_the_truth(self, model, truth, start):
        from sans_fitter import SANSFitter

        fixed = {'sld': 4.0, 'sld_solvent': 1.0, 'scale': 1.0, 'background': 0.001}
        data = examples.simulate(model, npoints=80, noise=0.02, seed=3, **truth, **fixed)

        with contextlib.redirect_stdout(io.StringIO()):
            fitter = SANSFitter()
            fitter.set_data(data)
            fitter.set_model(model)
            for param, value in fixed.items():
                fitter.set_param(param, value=value, vary=False)
            for param, value in start.items():
                fitter.set_param(param, value=value, min=value / 10, max=value * 10, vary=True)
            result = fitter.fit(engine='bumps', method='amoeba')

        for param, expected in truth.items():
            assert result['parameters'][param]['value'] == pytest.approx(expected, rel=0.05)
        # Error bars that honestly describe the scatter put reduced chi2 near 1.
        assert result['chisq'] < 5


class TestSimulatePair:
    def test_pair_shares_a_q_grid(self):
        sample, background = examples.simulate_pair('sphere', radius=50, npoints=50)
        np.testing.assert_allclose(sample.x, background.x)

    def test_background_carries_only_the_flat_level(self):
        _, background = examples.simulate_pair(
            'sphere', radius=50, background_level=0.5, noise=0, npoints=40
        )
        np.testing.assert_allclose(background.y, 0.5, rtol=1e-5)

    def test_sample_includes_the_background(self):
        sample, _ = examples.simulate_pair('sphere', radius=50, background_level=0.5)
        assert sample.truth['background'] == pytest.approx(0.5)

    def test_pair_is_subtractable_by_data_ops(self):
        from sans_fitter import data_ops

        sample, background = examples.simulate_pair('sphere', radius=50, npoints=50)
        result = data_ops.subtract(sample, background)
        assert len(result.x) == 50
        assert result.qmin is not None

    def test_pair_is_labelled_for_readable_provenance(self):
        from sans_fitter import data_ops

        sample, background = examples.simulate_pair('sphere', radius=50, npoints=30)
        assert sample.filename != background.filename
        title = data_ops.subtract(sample, background).title
        assert 'sample' in title and 'background' in title

    def test_the_two_datasets_have_independent_noise(self):
        sample, background = examples.simulate_pair('sphere', radius=50, seed=5, npoints=40)
        assert not np.array_equal(sample.y, background.y)

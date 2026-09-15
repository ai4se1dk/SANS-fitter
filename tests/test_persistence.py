"""Tests for save_analysis / load_analysis and the report document.

Almost nothing here runs an optimizer. A serialization test wants to know that
state and curves survive a round trip, and a fit adds minutes of runtime, a
dependence on which optimizer happens to converge, and nothing to the question
being asked. ``attach_synthetic_fit`` builds the same contract an engine would
hand to ``_finalize_fit``, so the result path is exercised end to end without
one.
"""

import json
import os
import warnings

import numpy as np
import pytest

from sans_fitter import SANSFitter
from sans_fitter.data.provenance import fingerprint_arrays
from sans_fitter.persistence import (
    SCHEMA_FORMAT,
    SCHEMA_VERSION,
    AnalysisFileError,
    build_fit_context,
    decode_numbers,
    encode_numbers,
)
from sans_fitter.results import FitArtifacts, FitResultContract, PosteriorDigest, PosteriorSummary

from .helpers import create_loading_test_data_file, create_multi_dataset_xml_file

EXAMPLE_DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'example_sans_data.dat'
)


# =========================================================================
# Fixtures and helpers
# =========================================================================


@pytest.fixture
def fitter():
    """A sphere fit on the bundled example data."""
    f = SANSFitter()
    f.load_data(EXAMPLE_DATA)
    f.set_model('sphere')
    f.set_param('radius', value=40.0, min=1.0, max=200.0, vary=True)
    return f


def attach_synthetic_fit(f, *, engine='bumps', method='lm', posterior=None):
    """Give *f* a fit result for its current configuration, without fitting.

    Mirrors what an engine produces and what ``_finalize_fit`` records: the
    theory at the current values, the residuals that theory implies, and the
    fit context describing the configuration it belongs to.
    """
    evaluation_data = f._evaluation_data(warn=False)
    curve, fit_index = f._evaluate(evaluation_data)
    y = np.asarray(evaluation_data.y, dtype=float)[fit_index]
    dy = np.asarray(evaluation_data.dy, dtype=float)[fit_index]
    sigma = np.where(np.nan_to_num(dy) == 0, 1.0, dy)
    # bumps reports (theory - I)/dI, the scipy engine (I - theory)/dI.
    residuals = (curve - y) / sigma if engine == 'bumps' else (y - curve) / sigma
    chisq = float(np.sum(residuals**2))

    varying = [name for name, info in f.params.items() if info['vary']]
    n_points = int(fit_index.sum())
    n_free = len(varying)
    contract = FitResultContract(
        engine=engine,
        method=method,
        chisq=chisq,
        reduced_chisq=chisq / (n_points - n_free) if n_points > n_free else float('nan'),
        n_points=n_points,
        n_free=n_free,
        dof=n_points - n_free,
        weighting_note='dI',
        parameters={
            name: {
                'value': info['value'],
                'stderr': 0.5,
                'formatted': f'{info["value"]:.4g} +/- 0.5',
                'fixed': not info['vary'],
                'linked_to': f.get_links().get(name),
            }
            for name, info in f.params.items()
        },
        resolution=f.get_resolution(),
        converged=True,
        message='synthetic',
        cov=np.eye(n_free) * 0.25 if n_free else None,
        cov_labels=varying,
        cov_source='synthetic',
        on_bounds=[],
        artifacts=FitArtifacts(
            fitted_curve=curve,
            fit_index=fit_index,
            residuals=residuals,
            posterior=posterior,
        ),
    )
    contract.fit_context = build_fit_context(
        f._param_manager.export_config(),
        f.get_resolution(),
        f.get_q_range(),
        n_points,
        fingerprint_arrays(f.data),
    )
    f._fit_contract = contract
    f.fit_result = contract.to_legacy_dict()
    return contract


def round_trip(f, tmp_path, name='analysis.json', **kwargs):
    path = os.path.join(str(tmp_path), name)
    f.save_analysis(path, **kwargs)
    return SANSFitter.load_analysis(path), path


def assert_same_configuration(a, b):
    assert a._param_manager.export_config() == b._param_manager.export_config()
    assert a.get_resolution() == b.get_resolution()
    assert a.get_q_range() == pytest.approx(b.get_q_range())


# =========================================================================
# Configuration round-trip, one per model shape
# =========================================================================


def test_atomic_model_round_trip(fitter, tmp_path):
    loaded, _ = round_trip(fitter, tmp_path)
    assert_same_configuration(fitter, loaded)
    assert loaded.model_name == 'sphere'


def test_raw_composite_expression_round_trip(tmp_path):
    """A raw set_model() composite has components but no alias layer."""
    f = SANSFitter()
    f.load_data(EXAMPLE_DATA)
    f.set_model('dab+peak_lorentz')
    f.set_param('A_cor_length', value=55.0, vary=True)
    loaded, _ = round_trip(f, tmp_path)
    assert_same_configuration(f, loaded)
    assert loaded.params['A_cor_length']['value'] == 55.0


def test_product_mixture_round_trip(tmp_path):
    f = SANSFitter()
    f.load_data(EXAMPLE_DATA)
    f.set_models('dab', 'peak_lorentz', operation='*')
    loaded, _ = round_trip(f, tmp_path)
    assert_same_configuration(f, loaded)


def test_shared_parameter_round_trip(tmp_path):
    """sphere and cylinder both have sld, so it can genuinely be shared."""
    f = SANSFitter()
    f.load_data(EXAMPLE_DATA)
    f.set_models('sphere', 'cylinder', shared=['sld'])
    f.set_param('sld', value=2.5, vary=True)
    loaded, _ = round_trip(f, tmp_path)
    assert_same_configuration(f, loaded)
    assert loaded.params['sld']['value'] == 2.5
    assert 'sphere_radius' in loaded.params


def test_auto_suffixed_monikers_round_trip(tmp_path):
    """Positional duplicates get sphere1/sphere2; keyword monikers do not."""
    f = SANSFitter()
    f.load_data(EXAMPLE_DATA)
    f.set_models('sphere', 'sphere')
    assert [moniker for _p, moniker, _n in f.get_components()] == ['sphere1', 'sphere2']
    f.set_param('sphere2_radius', value=77.0, vary=True)
    loaded, _ = round_trip(f, tmp_path)
    assert_same_configuration(f, loaded)
    assert loaded.params['sphere2_radius']['value'] == 77.0


def test_explicit_monikers_round_trip(tmp_path):
    f = SANSFitter()
    f.load_data(EXAMPLE_DATA)
    f.set_models(small='sphere', large='sphere')
    assert [moniker for _p, moniker, _n in f.get_components()] == ['small', 'large']
    f.set_param('large_radius', value=120.0, vary=True)
    loaded, _ = round_trip(f, tmp_path)
    assert_same_configuration(f, loaded)
    assert loaded.params['large_radius']['value'] == 120.0


def test_structure_factor_part_inside_mixture_round_trip(tmp_path):
    """The '@' rides inside the expression; _structure_factor_name stays None."""
    f = SANSFitter()
    f.load_data(EXAMPLE_DATA)
    f.set_models('sphere@hardsphere', 'peak_lorentz')
    assert f.get_structure_factor() is None
    loaded, _ = round_trip(f, tmp_path)
    assert_same_configuration(f, loaded)
    assert loaded.get_structure_factor() is None


def test_structure_factor_link_radius_round_trip(tmp_path):
    """The mode creates its link before values are restored (probe 5)."""
    f = SANSFitter()
    f.load_data(EXAMPLE_DATA)
    f.set_model('sphere')
    f.set_structure_factor('hardsphere', radius_effective_mode='link_radius')
    f.set_param('radius', value=63.0, min=5.0, max=150.0, vary=True)
    assert f.get_links() == {'radius_effective': 'radius'}

    loaded, _ = round_trip(f, tmp_path)
    assert_same_configuration(f, loaded)
    assert loaded.get_structure_factor() == 'hardsphere'
    assert loaded.get_links() == {'radius_effective': 'radius'}
    assert loaded.params['radius']['value'] == 63.0
    assert loaded.params['radius_effective']['value'] == 63.0
    assert loaded.params['radius']['min'] == 5.0


def test_explicit_link_round_trip(fitter, tmp_path):
    fitter.set_param('sld_solvent', value=3.0)
    fitter.link_params('sld', 'sld_solvent')
    loaded, _ = round_trip(fitter, tmp_path)
    assert loaded.get_links() == {'sld': 'sld_solvent'}
    assert loaded.params['sld']['value'] == 3.0
    assert loaded.params['sld']['vary'] is False


def test_polydispersity_round_trip(fitter, tmp_path):
    fitter.set_pd_param('radius', pd_width=0.18, pd_type='lognormal', pd_n=41, vary=True)
    fitter.enable_polydispersity(True)
    loaded, _ = round_trip(fitter, tmp_path)
    pd = loaded.get_pd_param('radius')
    assert pd['pd'] == 0.18
    assert pd['pd_type'] == 'lognormal'
    assert pd['pd_n'] == 41
    assert pd['vary'] is True
    assert loaded.is_polydispersity_enabled() is True


def test_zero_width_polydispersity_with_vary_round_trip(fitter, tmp_path):
    fitter.set_pd_param('radius', pd_width=0.0, vary=True)
    fitter.enable_polydispersity(True)
    loaded, _ = round_trip(fitter, tmp_path)
    assert loaded.get_pd_param('radius')['pd'] == 0.0
    assert loaded.get_pd_param('radius')['vary'] is True


def test_configured_but_disabled_polydispersity_stays_disabled(fitter, tmp_path):
    """Unconditionally enabling on load would change the analysis."""
    fitter.set_pd_param('radius', pd_width=0.22, pd_type='schulz')
    fitter.enable_polydispersity(False)
    loaded, _ = round_trip(fitter, tmp_path)
    assert loaded.is_polydispersity_enabled() is False
    assert loaded.get_pd_param('radius')['pd'] == 0.22
    assert loaded.get_pd_param('radius')['pd_type'] == 'schulz'


@pytest.mark.parametrize(
    'mode,kwargs',
    [
        ('data', {}),
        ('none', {}),
        ('pinhole', {'dq_over_q': 0.07}),
        ('slit', {'slit_length': 0.05}),
    ],
)
def test_resolution_modes_round_trip(fitter, tmp_path, mode, kwargs):
    fitter.set_resolution(mode, **kwargs)
    loaded, _ = round_trip(fitter, tmp_path)
    assert loaded.get_resolution() == fitter.get_resolution()
    assert np.allclose(loaded.calculate(), fitter.calculate(), equal_nan=True)


def test_resolution_none_over_a_file_with_dq(tmp_path):
    """'none' must survive: the file's dQ column would otherwise smear it."""
    from .helpers import create_loading_test_data_file_with_resolution

    path = create_loading_test_data_file_with_resolution(20)
    f = SANSFitter()
    f.load_data(path)
    f.set_model('sphere')
    f.set_resolution('none')
    unsmeared = f.calculate()

    loaded, _ = round_trip(f, tmp_path)
    assert loaded.get_resolution()['mode'] == 'none'
    assert np.allclose(loaded.calculate(), unsmeared, equal_nan=True)


def test_q_range_round_trip(fitter, tmp_path):
    fitter.set_q_range(0.01, 0.2)
    loaded, _ = round_trip(fitter, tmp_path)
    assert loaded.get_q_range() == pytest.approx((0.01, 0.2))
    assert loaded._full_q_range == pytest.approx(fitter._full_q_range)


def test_full_q_range_is_not_persisted_as_a_restriction(fitter, tmp_path):
    _, path = round_trip(fitter, tmp_path)
    assert json.load(open(path, encoding='utf-8'))['fit_range'] is None


def test_shuffled_key_order_loads_identically(fitter, tmp_path):
    """The reader must not depend on the order keys happen to appear in."""
    import random

    _, path = round_trip(fitter, tmp_path)
    document = json.load(open(path, encoding='utf-8'))

    def shuffle(node):
        if isinstance(node, dict):
            items = list(node.items())
            random.shuffle(items)
            return {key: shuffle(value) for key, value in items}
        return node

    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(shuffle(document), handle)
    loaded = SANSFitter.load_analysis(path)
    assert_same_configuration(fitter, loaded)


def test_saving_does_not_mutate_the_fitter(fitter, tmp_path):
    before = fitter._param_manager.export_config()
    fitter.save_analysis(os.path.join(str(tmp_path), 'a.json'))
    assert fitter._param_manager.export_config() == before


def test_save_without_a_model_is_refused(tmp_path):
    f = SANSFitter()
    f.load_data(EXAMPLE_DATA)
    with pytest.raises(AnalysisFileError, match='No model'):
        f.save_analysis(os.path.join(str(tmp_path), 'a.json'))


# =========================================================================
# Numeric codec
# =========================================================================


def test_infinite_bounds_survive_and_are_not_bare_tokens(fitter, tmp_path):
    """scale/background default to an infinite upper bound."""
    assert fitter.params['scale']['max'] == np.inf
    loaded, path = round_trip(fitter, tmp_path)
    text = open(path, encoding='utf-8').read()
    assert '"Infinity"' in text
    assert ': Infinity' not in text
    assert 'NaN' not in text
    assert loaded.params['scale']['max'] == np.inf


def test_written_file_is_strict_json(fitter, tmp_path):
    _, path = round_trip(fitter, tmp_path)
    # parse_constant fires only for Infinity/-Infinity/NaN, so this raises if
    # any bare constant reached the file.
    json.loads(
        open(path, encoding='utf-8').read(),
        parse_constant=lambda token: pytest.fail(f'bare {token} in the file'),
    )


def test_encoder_handles_what_a_default_hook_cannot():
    """json.dumps never calls a default hook for a built-in float (probe 1)."""

    class Hook(json.JSONEncoder):
        def default(self, o):  # pragma: no cover - never reached for floats
            return 'Infinity'

    assert json.dumps({'b': float('inf')}, cls=Hook) == '{"b": Infinity}'
    assert json.dumps(encode_numbers({'b': float('inf')}), allow_nan=False) == '{"b": "Infinity"}'


def test_encode_decode_round_trip_of_infinities():
    original = {'min': -np.inf, 'max': np.inf, 'value': 1.5, 'name': 'sphere'}
    assert decode_numbers(encode_numbers(original)) == original


def test_nan_in_configuration_is_rejected():
    with pytest.raises(AnalysisFileError, match='NaN'):
        encode_numbers({'value': float('nan')})


def test_numpy_scalars_normalize():
    encoded = encode_numbers({'a': np.float64(1.5), 'b': np.int64(3), 'c': np.bool_(True)})
    assert encoded == {'a': 1.5, 'b': 3, 'c': True}
    json.dumps(encoded, allow_nan=False)


def test_non_finite_parameter_value_is_rejected(fitter, tmp_path):
    _, path = round_trip(fitter, tmp_path)
    document = json.load(open(path, encoding='utf-8'))
    document['configuration']['params']['radius']['value'] = 'Infinity'
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(document, handle)
    with pytest.raises(AnalysisFileError, match='must be finite'):
        SANSFitter.load_analysis(path)


# =========================================================================
# Validation
# =========================================================================


def _corrupt(path, mutate):
    document = json.load(open(path, encoding='utf-8'))
    mutate(document)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(document, handle)
    return path


def test_missing_file(tmp_path):
    with pytest.raises(AnalysisFileError, match='not found'):
        SANSFitter.load_analysis(os.path.join(str(tmp_path), 'nope.json'))


def test_malformed_json(tmp_path):
    path = os.path.join(str(tmp_path), 'bad.json')
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('{not json')
    with pytest.raises(AnalysisFileError, match='not valid JSON'):
        SANSFitter.load_analysis(path)


def test_wrong_format(fitter, tmp_path):
    _, path = round_trip(fitter, tmp_path)
    _corrupt(path, lambda d: d['schema'].update(format='something-else'))
    with pytest.raises(AnalysisFileError, match='schema.format'):
        SANSFitter.load_analysis(path)


def test_newer_schema_version_is_refused(fitter, tmp_path):
    _, path = round_trip(fitter, tmp_path)
    _corrupt(path, lambda d: d['schema'].update(version=SCHEMA_VERSION + 1))
    with pytest.raises(AnalysisFileError, match='newer sans-fitter'):
        SANSFitter.load_analysis(path)


def test_missing_configuration_section(fitter, tmp_path):
    _, path = round_trip(fitter, tmp_path)
    _corrupt(path, lambda d: d.pop('configuration'))
    with pytest.raises(AnalysisFileError, match='configuration'):
        SANSFitter.load_analysis(path)


def test_unknown_model_is_an_error_not_a_warning(fitter, tmp_path):
    _, path = round_trip(fitter, tmp_path)
    _corrupt(path, lambda d: d['configuration'].update(model_name='not_a_real_model'))
    with pytest.raises(ValueError, match='not_a_real_model'):
        SANSFitter.load_analysis(path)


def test_unknown_parameter_is_an_error_not_a_warning(fitter, tmp_path):
    _, path = round_trip(fitter, tmp_path)
    _corrupt(
        path,
        lambda d: d['configuration']['params'].update(
            not_a_param={'value': 1.0, 'min': 0.0, 'max': 2.0, 'vary': False}
        ),
    )
    with pytest.raises(AnalysisFileError, match='Unknown: not_a_param'):
        SANSFitter.load_analysis(path)


def test_missing_parameter_is_an_error(fitter, tmp_path):
    _, path = round_trip(fitter, tmp_path)
    _corrupt(path, lambda d: d['configuration']['params'].pop('radius'))
    with pytest.raises(AnalysisFileError, match='Missing: radius'):
        SANSFitter.load_analysis(path)


def test_value_outside_bounds_is_rejected(fitter, tmp_path):
    _, path = round_trip(fitter, tmp_path)
    _corrupt(path, lambda d: d['configuration']['params']['radius'].update(value=5000.0))
    with pytest.raises(AnalysisFileError, match='outside its bounds'):
        SANSFitter.load_analysis(path)


def test_inverted_bounds_are_rejected(fitter, tmp_path):
    _, path = round_trip(fitter, tmp_path)
    _corrupt(
        path,
        lambda d: d['configuration']['params']['radius'].update(min=100.0, max=1.0, value=50.0),
    )
    with pytest.raises(AnalysisFileError, match='is above max'):
        SANSFitter.load_analysis(path)


def test_non_boolean_vary_is_rejected(fitter, tmp_path):
    _, path = round_trip(fitter, tmp_path)
    _corrupt(path, lambda d: d['configuration']['params']['radius'].update(vary='yes'))
    with pytest.raises(AnalysisFileError, match='must be true or false'):
        SANSFitter.load_analysis(path)


def test_bad_link_target_is_rejected(fitter, tmp_path):
    _, path = round_trip(fitter, tmp_path)
    _corrupt(path, lambda d: d['configuration'].update(links={'radius': 'nonexistent'}))
    with pytest.raises(AnalysisFileError, match='link target'):
        SANSFitter.load_analysis(path)


def test_link_chain_is_rejected(fitter, tmp_path):
    _, path = round_trip(fitter, tmp_path)
    _corrupt(
        path,
        lambda d: d['configuration'].update(links={'sld': 'sld_solvent', 'sld_solvent': 'radius'}),
    )
    with pytest.raises(AnalysisFileError, match='chain'):
        SANSFitter.load_analysis(path)


def test_link_mode_conflict_is_rejected(tmp_path):
    """A hand-edited file must not let setter order decide the meaning."""
    f = SANSFitter()
    f.load_data(EXAMPLE_DATA)
    f.set_model('sphere')
    f.set_structure_factor('hardsphere', radius_effective_mode='link_radius')
    _, path = round_trip(f, tmp_path)
    _corrupt(path, lambda d: d['configuration'].update(links={}))
    with pytest.raises(AnalysisFileError, match='link_radius'):
        SANSFitter.load_analysis(path)


def test_invalid_pd_type_is_rejected(fitter, tmp_path):
    fitter.set_pd_param('radius', pd_width=0.1)
    _, path = round_trip(fitter, tmp_path)
    _corrupt(
        path,
        lambda d: d['configuration']['polydispersity']['params']['radius'].update(
            pd_type='triangular'
        ),
    )
    with pytest.raises(AnalysisFileError, match='invalid pd_type'):
        SANSFitter.load_analysis(path)


def test_component_mismatch_is_rejected(tmp_path):
    """register_aliases overlays by position and never checks the model there."""
    f = SANSFitter()
    f.load_data(EXAMPLE_DATA)
    f.set_models('sphere', 'cylinder')
    _, path = round_trip(f, tmp_path)

    def swap(document):
        components = document['configuration']['components']
        components[0][2], components[1][2] = components[1][2], components[0][2]

    _corrupt(path, swap)
    with pytest.raises(AnalysisFileError, match='components'):
        SANSFitter.load_analysis(path)


def test_bare_json_constants_are_rejected(fitter, tmp_path):
    _, path = round_trip(fitter, tmp_path)
    text = open(path, encoding='utf-8').read().replace('"Infinity"', 'Infinity', 1)
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write(text)
    with pytest.raises(AnalysisFileError, match='non-standard JSON constant'):
        SANSFitter.load_analysis(path)


# =========================================================================
# The model trust boundary
# =========================================================================


def test_custom_model_is_refused_before_its_loader_runs(fitter, tmp_path):
    """No plugin is executed to prove this: the check precedes set_model."""
    _, path = round_trip(fitter, tmp_path)
    _corrupt(path, lambda d: d['configuration'].update(model_name='custom.evil_plugin'))
    with pytest.raises(AnalysisFileError, match='not built into'):
        SANSFitter.load_analysis(path)


def test_custom_model_opt_in_reaches_the_loader(fitter, tmp_path):
    """With the opt-in the expression is passed on, and fails as a model."""
    _, path = round_trip(fitter, tmp_path)
    _corrupt(path, lambda d: d['configuration'].update(model_name='custom.evil_plugin'))
    with pytest.raises(ValueError) as excinfo:
        SANSFitter.load_analysis(path, allow_custom_models=True)
    assert 'not built into' not in str(excinfo.value)


# =========================================================================
# Data identity
# =========================================================================


def test_moved_directory_still_loads(fitter, tmp_path):
    """The relative path is recorded against the analysis file, not the CWD."""
    import shutil

    home = os.path.join(str(tmp_path), 'home')
    os.makedirs(home)
    data_copy = os.path.join(home, 'data.dat')
    shutil.copy(EXAMPLE_DATA, data_copy)

    f = SANSFitter()
    f.load_data(data_copy)
    f.set_model('sphere')
    analysis = os.path.join(home, 'a.json')
    f.save_analysis(analysis)

    moved = os.path.join(str(tmp_path), 'moved')
    shutil.move(home, moved)
    loaded = SANSFitter.load_analysis(os.path.join(moved, 'a.json'))
    assert loaded.data is not None
    assert loaded.model_name == 'sphere'


def test_changed_working_directory_still_loads(fitter, tmp_path, monkeypatch):
    _, path = round_trip(fitter, tmp_path)
    elsewhere = os.path.join(str(tmp_path), 'elsewhere')
    os.makedirs(elsewhere)
    monkeypatch.chdir(elsewhere)
    assert SANSFitter.load_analysis(path).data is not None


def test_missing_data_file_names_both_paths(fitter, tmp_path):
    import shutil

    data_copy = os.path.join(str(tmp_path), 'gone.dat')
    shutil.copy(EXAMPLE_DATA, data_copy)
    f = SANSFitter()
    f.load_data(data_copy)
    f.set_model('sphere')
    path = os.path.join(str(tmp_path), 'a.json')
    f.save_analysis(path)
    os.remove(data_copy)

    with pytest.raises(AnalysisFileError) as excinfo:
        SANSFitter.load_analysis(path)
    assert 'Tried' in str(excinfo.value)
    assert 'data=' in str(excinfo.value)


def test_explicit_data_override_fails_as_requested(fitter, tmp_path):
    """An explicit data= must not fall back silently to the recorded sample."""
    _, path = round_trip(fitter, tmp_path)
    with pytest.raises(AnalysisFileError, match='given as data= not found'):
        SANSFitter.load_analysis(path, data=os.path.join(str(tmp_path), 'absent.dat'))


def test_dataset_selector_by_index_round_trips(tmp_path):
    path = create_multi_dataset_xml_file()
    f = SANSFitter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f.load_data(path, dataset=1)
    f.set_model('sphere')
    analysis = os.path.join(str(tmp_path), 'a.json')
    f.save_analysis(analysis)
    assert json.load(open(analysis, encoding='utf-8'))['data']['dataset'] == 1

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        loaded = SANSFitter.load_analysis(analysis)
    assert loaded._data_source.dataset_index == 1


def test_dataset_selector_by_name_round_trips(tmp_path):
    path = create_multi_dataset_xml_file()
    f = SANSFitter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f.load_data(path, dataset='beta sample')
    f.set_model('sphere')
    analysis = os.path.join(str(tmp_path), 'a.json')
    f.save_analysis(analysis)
    assert json.load(open(analysis, encoding='utf-8'))['data']['dataset'] == 'beta sample'

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        loaded = SANSFitter.load_analysis(analysis)
    assert loaded._data_source.dataset_index == 1


def test_in_memory_data_records_provenance_and_needs_an_override(tmp_path):
    from sans_fitter import data_ops

    sample = data_ops.load(EXAMPLE_DATA)
    scaled = data_ops.multiply(sample, 2.0)

    f = SANSFitter()
    f.set_data(scaled)
    f.set_model('sphere')
    path = os.path.join(str(tmp_path), 'a.json')
    f.save_analysis(path)

    section = json.load(open(path, encoding='utf-8'))['data']
    assert section['source'] == 'memory'
    assert 'path_absolute' not in section

    with pytest.raises(AnalysisFileError, match='in-memory dataset'):
        SANSFitter.load_analysis(path)

    loaded = SANSFitter.load_analysis(path, data=scaled)
    assert loaded.model_name == 'sphere'


def test_in_memory_provenance_is_quoted_in_the_error(tmp_path):
    from sans_fitter import data_ops

    sample = data_ops.load(EXAMPLE_DATA)
    difference = data_ops.subtract(sample, 0.5)
    f = SANSFitter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f.set_data(difference)
    f.set_model('sphere')
    path = os.path.join(str(tmp_path), 'a.json')
    f.save_analysis(path)
    with pytest.raises(AnalysisFileError) as excinfo:
        SANSFitter.load_analysis(path)
    assert 'sans_fitter.data_ops' in str(excinfo.value)


def test_source_file_change_is_detected(tmp_path):
    import shutil

    data_copy = os.path.join(str(tmp_path), 'moving.csv')
    shutil.copy(create_loading_test_data_file(20), data_copy)
    f = SANSFitter()
    f.load_data(data_copy)
    f.set_model('sphere')
    path = os.path.join(str(tmp_path), 'a.json')
    f.save_analysis(path)

    shutil.copy(create_loading_test_data_file(19), data_copy)
    with pytest.warns(UserWarning, match='changed since this analysis was saved'):
        SANSFitter.load_analysis(path)


def test_in_memory_edits_are_detected(fitter, tmp_path):
    """A file hash cannot see this; the array fingerprint can."""
    before = fingerprint_arrays(fitter.data)
    fitter.data.y = np.asarray(fitter.data.y, dtype=float) * 1.01
    assert fingerprint_arrays(fitter.data) != before

    _, path = round_trip(fitter, tmp_path)
    section = json.load(open(path, encoding='utf-8'))['data']
    assert section['array_fingerprint_now'] != section['array_fingerprint_at_load']


# =========================================================================
# Result, artifacts and staleness
# =========================================================================


def test_result_round_trip_restores_every_contract_field(fitter, tmp_path):
    original = attach_synthetic_fit(fitter)
    loaded, _ = round_trip(fitter, tmp_path)

    assert loaded.fit_result is not None
    restored = loaded._fit_contract
    assert restored.engine == original.engine
    assert restored.method == original.method
    assert restored.chisq == pytest.approx(original.chisq)
    assert restored.reduced_chisq == pytest.approx(original.reduced_chisq)
    assert (restored.n_points, restored.n_free, restored.dof) == (
        original.n_points,
        original.n_free,
        original.dof,
    )
    assert restored.weighting_note == original.weighting_note
    assert restored.converged is True
    assert restored.message == 'synthetic'
    assert restored.cov_source == 'synthetic'
    assert restored.cov_labels == original.cov_labels
    assert np.allclose(restored.cov, original.cov)
    assert restored.resolution == original.resolution
    assert restored.parameters['radius']['formatted'] == original.parameters['radius']['formatted']
    assert restored.parameters['radius']['stderr'] == pytest.approx(0.5)


def test_loaded_result_rebuilds_curve_residuals_and_index(fitter, tmp_path):
    original = attach_synthetic_fit(fitter)
    loaded, _ = round_trip(fitter, tmp_path)
    rebuilt = loaded._fit_contract.artifacts
    assert np.allclose(rebuilt.fitted_curve, original.artifacts.fitted_curve)
    assert np.allclose(rebuilt.residuals, original.artifacts.residuals)
    assert np.array_equal(rebuilt.fit_index, original.artifacts.fit_index)
    assert float(np.sum(rebuilt.residuals**2)) == pytest.approx(original.chisq)


def test_loaded_result_supports_plot_save_and_report(fitter, tmp_path):
    attach_synthetic_fit(fitter)
    loaded, _ = round_trip(fitter, tmp_path)
    assert loaded.plot_results(show=False) is not None
    assert 'radius' in loaded.get_fit_report().to_markdown()
    out = os.path.join(str(tmp_path), 'out.csv')
    loaded.save_results(out)
    assert 'Q,' in open(out, encoding='utf-8').read()


def test_exported_csv_matches_the_original(fitter, tmp_path):
    attach_synthetic_fit(fitter)
    loaded, _ = round_trip(fitter, tmp_path)
    first = os.path.join(str(tmp_path), 'a.csv')
    second = os.path.join(str(tmp_path), 'b.csv')
    fitter.save_results(first)
    loaded.save_results(second)
    assert open(first, encoding='utf-8').read() == open(second, encoding='utf-8').read()


def test_result_with_no_context_is_not_attached(fitter, tmp_path):
    """A result from before this feature cannot be vouched for."""
    contract = attach_synthetic_fit(fitter)
    contract.fit_context = None
    _, path = round_trip(fitter, tmp_path)
    document = json.load(open(path, encoding='utf-8'))
    assert document['result'] is None
    assert 'predates' in document['result_omitted']


@pytest.mark.parametrize(
    'mutate,expected',
    [
        (lambda f: f.set_param('radius', value=99.0), 'parameters changed'),
        (lambda f: f.set_q_range(0.02, 0.2), 'Q range changed'),
        (lambda f: f.set_resolution('none'), 'resolution setting changed'),
        (lambda f: f.set_pd_param('radius', pd_width=0.3), 'parameters changed'),
        (lambda f: f.enable_polydispersity(True), 'parameters changed'),
    ],
)
def test_post_fit_mutation_omits_the_result(fitter, tmp_path, mutate, expected):
    attach_synthetic_fit(fitter)
    mutate(fitter)
    _, path = round_trip(fitter, tmp_path)
    document = json.load(open(path, encoding='utf-8'))
    assert document['result'] is None
    assert expected in document['result_omitted']


def test_data_replacement_omits_the_result(fitter, tmp_path):
    attach_synthetic_fit(fitter)
    fitter.data.y = np.asarray(fitter.data.y, dtype=float) * 1.05
    _, path = round_trip(fitter, tmp_path)
    document = json.load(open(path, encoding='utf-8'))
    assert document['result'] is None
    assert 'data changed' in document['result_omitted']


def test_include_result_false_writes_a_template(fitter, tmp_path):
    attach_synthetic_fit(fitter)
    path = os.path.join(str(tmp_path), 'template.json')
    fitter.save_analysis(path, include_result=False)
    document = json.load(open(path, encoding='utf-8'))
    assert document['result'] is None
    assert document['result_omitted'] is None

    loaded = SANSFitter.load_analysis(path)
    assert loaded.fit_result is None
    assert_same_configuration(fitter, loaded)


def test_inconsistent_saved_chisq_attaches_no_artifacts(fitter, tmp_path):
    attach_synthetic_fit(fitter)
    _, path = round_trip(fitter, tmp_path)
    _corrupt(path, lambda d: d['result'].update(chisq=d['result']['chisq'] * 3.0))
    loaded = SANSFitter.load_analysis(path)
    assert loaded.fit_result is None


def test_inconsistent_saved_point_count_attaches_no_artifacts(fitter, tmp_path):
    attach_synthetic_fit(fitter)
    _, path = round_trip(fitter, tmp_path)
    _corrupt(path, lambda d: d['result'].update(n_points=d['result']['n_points'] - 3))
    loaded = SANSFitter.load_analysis(path)
    assert loaded.fit_result is None


def test_component_curves_are_rebuilt_for_a_sum_mixture(tmp_path):
    f = SANSFitter()
    f.load_data(EXAMPLE_DATA)
    f.set_models('sphere', 'cylinder')
    f.set_param('sphere_radius', value=45.0, vary=True)
    attach_synthetic_fit(f)
    loaded, _ = round_trip(f, tmp_path)
    assert loaded._fit_contract.artifacts.component_curves


# =========================================================================
# Posterior
# =========================================================================


def _posterior():
    samples = np.random.default_rng(0).normal(40.0, 1.0, size=(200, 1))
    return PosteriorSummary(
        labels=['radius'],
        samples=samples,
        best={'radius': 40.0},
        mean={'radius': 40.1},
        median={'radius': 40.0},
        std={'radius': 1.0},
        ci_68={'radius': (39.0, 41.0)},
        ci_95={'radius': (38.0, 42.0)},
        diagnostics={'radius': {'r_hat': 1.01, 'ess': 150.0}},
    )


def test_posterior_statistics_survive_without_the_chain(fitter, tmp_path):
    attach_synthetic_fit(fitter, posterior=_posterior())
    loaded, _ = round_trip(fitter, tmp_path)
    restored = loaded._fit_contract.artifacts.posterior
    assert isinstance(restored, PosteriorDigest)
    assert restored.n_samples == 200
    assert restored.median['radius'] == 40.0
    assert restored.diagnostics['radius']['r_hat'] == 1.01
    assert 'Posterior summary' in restored.format_summary()


def test_loaded_posterior_reaches_the_report(fitter, tmp_path):
    attach_synthetic_fit(fitter, posterior=_posterior())
    loaded, _ = round_trip(fitter, tmp_path)
    assert 'Posterior summary' in loaded.get_fit_report().to_markdown()


def test_sample_dependent_apis_fail_clearly(fitter, tmp_path):
    attach_synthetic_fit(fitter, posterior=_posterior())
    loaded, _ = round_trip(fitter, tmp_path)
    with pytest.raises(ValueError, match='stores posterior statistics but not the sample'):
        loaded.get_posterior()


def test_posterior_samples_are_not_written_to_the_file(fitter, tmp_path):
    attach_synthetic_fit(fitter, posterior=_posterior())
    _, path = round_trip(fitter, tmp_path)
    payload = json.load(open(path, encoding='utf-8'))['result']['posterior']
    assert 'samples' not in payload
    assert os.path.getsize(path) < 100_000


# =========================================================================
# Reports
# =========================================================================


def test_html_report_needs_no_image_backend(fitter, tmp_path):
    path = os.path.join(str(tmp_path), 'r.html')
    report = fitter.report(path)
    text = open(path, encoding='utf-8').read()
    assert text.startswith('<!DOCTYPE html>')
    assert 'plotly' in text.lower()
    assert report.to_html() == text


def test_html_offline_embeds_the_library(fitter, tmp_path):
    cdn = fitter.report(fmt='html').to_html()
    offline = fitter.report(fmt='html', offline=True).to_html()
    assert len(offline) > len(cdn) * 5


def test_report_without_a_fit_is_a_configuration_report(fitter, tmp_path):
    report = fitter.report(fmt='markdown')
    assert 'no fit has been run' in report.to_markdown()
    assert '| radius |' in report.to_markdown()


def test_report_with_a_fit_uses_fitreport(fitter, tmp_path):
    attach_synthetic_fit(fitter)
    markdown = fitter.report(fmt='markdown').to_markdown()
    html_text = fitter.report(fmt='html').to_html()
    assert 'no fit has been run' not in markdown
    for text in (markdown, html_text):
        assert 'radius' in text
        assert 'synthetic' in text  # the optimizer message, from FitReport


def test_report_reuses_fitreport_rendering(fitter, monkeypatch):
    """Both formats must go through FitReport, not just one."""
    from sans_fitter.report import FitReport

    attach_synthetic_fit(fitter)
    monkeypatch.setattr(FitReport, 'to_markdown', lambda self: 'MARKDOWN-SENTINEL')
    monkeypatch.setattr(FitReport, '_repr_html_', lambda self: 'HTML-SENTINEL')
    assert 'MARKDOWN-SENTINEL' in fitter.report(fmt='markdown').to_markdown()
    assert 'HTML-SENTINEL' in fitter.report(fmt='html').to_html()


def test_unknown_extension_lists_the_supported_ones(fitter, tmp_path):
    with pytest.raises(ValueError, match=r'\.html'):
        fitter.report(os.path.join(str(tmp_path), 'r.pdf'))


def test_markdown_asset_is_named_from_the_report_stem(fitter, tmp_path, monkeypatch):
    """A constant name would let two reports overwrite each other's figure."""
    monkeypatch.setattr('sans_fitter.reporting._render_image', lambda figure: b'PNG')
    first = os.path.join(str(tmp_path), 'sample_a.md')
    second = os.path.join(str(tmp_path), 'sample_b.md')
    fitter.report(first)
    fitter.report(second)
    assert os.path.exists(os.path.join(str(tmp_path), 'sample_a_fit.png'))
    assert os.path.exists(os.path.join(str(tmp_path), 'sample_b_fit.png'))
    assert 'sample_a_fit.png' in open(first, encoding='utf-8').read()


def test_markdown_without_a_backend_warns_and_omits_the_figure(fitter, tmp_path, monkeypatch):
    monkeypatch.setattr('sans_fitter.reporting._render_image', lambda figure: None)
    with pytest.warns(UserWarning, match='static image backend'):
        report = fitter.report(os.path.join(str(tmp_path), 'r.md'))
    assert report.assets == {}
    assert 'figure was omitted' in report.to_markdown()


def test_unusable_backend_is_treated_as_missing(fitter, tmp_path, monkeypatch):
    """Importing kaleido is not proof that it can render."""

    def explode(*args, **kwargs):
        raise RuntimeError('Chrome not found')

    monkeypatch.setattr('plotly.graph_objects.Figure.to_image', explode)
    with pytest.warns(UserWarning, match='static image backend'):
        fitter.report(os.path.join(str(tmp_path), 'r.md'))


def test_markdown_without_a_path_omits_the_figure(fitter, monkeypatch):
    """There is nowhere to put a sidecar, so it must not be referenced."""
    monkeypatch.setattr('sans_fitter.reporting._render_image', lambda figure: b'PNG')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        report = fitter.report(fmt='markdown')
    assert report.assets == {}
    assert '.png' not in report.to_markdown()


def test_report_escapes_interpolated_text(fitter):
    """Escaping must cover free text from engines, not only table cells.

    An optimizer message is the realistic carrier: monikers and model names are
    identifiers, but a message is whatever the engine produced.
    """
    contract = attach_synthetic_fit(fitter)
    contract.message = '<img src=x onerror="alert(1)">'
    fitter.fit_result = contract.to_legacy_dict()

    html_text = fitter.report(fmt='html').to_html()
    assert '<img src=x' not in html_text
    assert '&lt;img src=x' in html_text


def test_report_does_not_open_a_browser(fitter, monkeypatch):
    def fail(*args, **kwargs):  # pragma: no cover - only runs on regression
        raise AssertionError('report() must render with show=False')

    monkeypatch.setattr('plotly.graph_objects.Figure.show', fail)
    fitter.report(fmt='html')


def test_failed_report_write_leaves_the_previous_file(fitter, tmp_path, monkeypatch):
    path = os.path.join(str(tmp_path), 'r.html')
    fitter.report(path)
    good = open(path, encoding='utf-8').read()

    monkeypatch.setattr(
        'sans_fitter.reporting._atomic_write',
        lambda target, text: (_ for _ in ()).throw(OSError('disk full')),
    )
    with pytest.raises(OSError):
        fitter.report(path)
    assert open(path, encoding='utf-8').read() == good


def test_report_directory_must_exist(fitter, tmp_path):
    with pytest.raises(ValueError, match='Directory does not exist'):
        fitter.report(os.path.join(str(tmp_path), 'nope', 'r.html'))


# =========================================================================
# File integrity
# =========================================================================


def test_failed_save_leaves_the_previous_analysis(fitter, tmp_path, monkeypatch):
    path = os.path.join(str(tmp_path), 'a.json')
    fitter.save_analysis(path)
    good = open(path, encoding='utf-8').read()

    monkeypatch.setattr(
        'sans_fitter.persistence.json.dumps',
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError('boom')),
    )
    with pytest.raises(ValueError):
        fitter.save_analysis(path)
    assert open(path, encoding='utf-8').read() == good


def test_no_temporary_files_are_left_behind(fitter, tmp_path):
    fitter.save_analysis(os.path.join(str(tmp_path), 'a.json'))
    assert not [name for name in os.listdir(str(tmp_path)) if '.tmp-' in name]


def test_save_directory_must_exist(fitter, tmp_path):
    with pytest.raises(AnalysisFileError, match='Directory does not exist'):
        fitter.save_analysis(os.path.join(str(tmp_path), 'nope', 'a.json'))


def test_unicode_paths_and_text(tmp_path):
    import shutil

    directory = os.path.join(str(tmp_path), 'mesures_échantillon')
    os.makedirs(directory)
    data_copy = os.path.join(directory, 'données.dat')
    shutil.copy(EXAMPLE_DATA, data_copy)

    f = SANSFitter()
    f.load_data(data_copy)
    f.set_model('sphere')
    path = os.path.join(directory, 'analyse.json')
    f.save_analysis(path)
    loaded = SANSFitter.load_analysis(path)
    assert loaded.model_name == 'sphere'


def test_schema_header_identifies_the_writer(fitter, tmp_path):
    _, path = round_trip(fitter, tmp_path)
    schema = json.load(open(path, encoding='utf-8'))['schema']
    assert schema['format'] == SCHEMA_FORMAT
    assert schema['version'] == SCHEMA_VERSION
    assert schema['written_by']


def test_weighting_key_is_reserved(fitter, tmp_path):
    """Item 3 of #72 fills this in; it is present and null until then."""
    _, path = round_trip(fitter, tmp_path)
    document = json.load(open(path, encoding='utf-8'))
    assert 'weighting' in document
    assert document['weighting'] is None


# =========================================================================
# Real fits
#
# Everything above avoids the optimizer deliberately. These few do run one,
# because two things cannot be checked any other way: that each engine's own
# residual convention is reproduced, and that a converged, identifiable fit
# survives a save and a refit. The fixtures are small and deterministic.
# =========================================================================


@pytest.mark.parametrize('engine,method', [('bumps', 'lm'), ('lmfit', 'leastsq')])
def test_real_fit_round_trip_reproduces_artifacts(tmp_path, engine, method):
    """The engines disagree on residual sign, and chi-squared cannot tell.

    bumps reports (theory - I)/dI and the scipy engine (I - theory)/dI. A sum
    of squares is identical either way, so only comparing the vector itself
    catches a reconstruction that follows the wrong convention.
    """
    f = SANSFitter()
    f.load_data(EXAMPLE_DATA)
    f.set_model('sphere')
    f.set_param('radius', value=40.0, min=1.0, max=200.0, vary=True)
    original = f.fit(engine=engine, method=method)

    loaded, _ = round_trip(f, tmp_path)
    assert loaded.fit_result is not None
    assert loaded.fit_result['chisq'] == pytest.approx(original['chisq'])

    before, after = f._fit_contract.artifacts, loaded._fit_contract.artifacts
    assert np.allclose(before.fitted_curve, after.fitted_curve)
    assert np.allclose(before.residuals, after.residuals)
    assert np.array_equal(before.fit_index, after.fit_index)


@pytest.mark.parametrize('engine,method', [('bumps', 'lm'), ('lmfit', 'leastsq')])
def test_real_fit_csv_export_is_identical(tmp_path, engine, method):
    f = SANSFitter()
    f.load_data(EXAMPLE_DATA)
    f.set_model('sphere')
    f.set_param('radius', value=40.0, min=1.0, max=200.0, vary=True)
    f.fit(engine=engine, method=method)

    loaded, _ = round_trip(f, tmp_path)
    first = os.path.join(str(tmp_path), 'original.csv')
    second = os.path.join(str(tmp_path), 'loaded.csv')
    f.save_results(first)
    loaded.save_results(second)
    assert open(first, encoding='utf-8').read() == open(second, encoding='utf-8').read()


def test_refit_after_load_matches_a_refit_of_the_original(tmp_path):
    """The restored fitter is equivalent to the one that was saved.

    Stated as equivalence rather than as "a refit reproduces the saved
    numbers", which is not something the file can promise: it records the
    engine and method but no iteration budget, seed or starting point, and an
    optimizer does not necessarily stay at the optimum it reported. Refitting
    both fitters from the same restored state is the claim that holds.
    """
    f = SANSFitter()
    f.load_data(EXAMPLE_DATA)
    f.set_model('sphere')
    f.set_param('radius', value=40.0, min=1.0, max=200.0, vary=True)
    f.fit(engine='bumps', method='lm')

    loaded, _ = round_trip(f, tmp_path)
    assert loaded.params['radius']['value'] == f.params['radius']['value']

    from_loaded = loaded.fit(engine='bumps', method='lm')
    from_original = f.fit(engine='bumps', method='lm')
    assert from_loaded['parameters']['radius']['value'] == pytest.approx(
        from_original['parameters']['radius']['value'], rel=1e-9
    )
    assert from_loaded['chisq'] == pytest.approx(from_original['chisq'], rel=1e-9)


def test_real_fit_with_resolution_round_trips(tmp_path):
    """A custom resolution must reach the rebuild, not the file's own columns."""
    f = SANSFitter()
    f.load_data(EXAMPLE_DATA)
    f.set_model('sphere')
    f.set_param('radius', value=40.0, min=1.0, max=200.0, vary=True)
    f.set_resolution('pinhole', dq_over_q=0.05)
    f.fit(engine='bumps', method='lm')

    loaded, _ = round_trip(f, tmp_path)
    assert loaded.get_resolution()['dq_over_q'] == 0.05
    assert loaded.fit_result is not None
    assert np.allclose(
        loaded._fit_contract.artifacts.fitted_curve, f._fit_contract.artifacts.fitted_curve
    )

"""Explicit data weighting for the least-squares objective (#72, item 3).

The engines are the oracle here, as sasmodels is in ``test_resolution.py``:
a fit under a weighting mode must match a ``'dI'`` fit of a copy whose ``dy``
was set to the mode's formula by hand. Nothing below re-implements an engine.
"""

import copy
import json
import os
import warnings

import numpy as np
import pytest
from sasdata.dataloader.data_info import Data1D

from sans_fitter import SANSFitter, examples
from sans_fitter.data.loader import get_fit_index, normalize_sans_data
from sans_fitter.data.weighting import (
    DEFAULT_WEIGHTING_MODE,
    WEIGHTING_MODES,
    WeightingSetting,
    apply_weighting,
    validate_weighting,
)
from sans_fitter.fitting.bumps_engine import _build_bumps_problem
from sans_fitter.persistence import SCHEMA_VERSION
from sans_fitter.plotting import plot_fit
from sans_fitter.results import FitArtifacts, FitResultContract

FORMULAS = {
    'sqrt': lambda y: np.sqrt(np.abs(y)),
    'abs': np.abs,
    'none': np.ones_like,
}


def make_fitter(data, *, mode='dI', radius=45.0):
    """A sphere fitter on *data* with only the radius free, unsmeared."""
    fitter = SANSFitter()
    fitter.set_data(data)
    fitter.set_model('sphere')
    fitter.set_resolution('none')
    fitter.set_param('radius', value=radius, min=10, max=100, vary=True)
    fitter.set_weighting(mode)
    return fitter


@pytest.fixture
def noisy():
    return examples.simulate('sphere', radius=50, noise=0.02, seed=0)


@pytest.fixture
def noise_free():
    return examples.simulate('sphere', radius=50, noise=0, seed=0)


@pytest.fixture
def awkward():
    """Negative, zero and NaN intensities, with a real dI column."""
    y = np.array([4.0, -2.0, 0.0, np.nan, 9.0])
    return Data1D(
        x=np.linspace(0.01, 0.05, 5),
        y=y,
        dy=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
    )


# =========================================================================
# 1. Validation
# =========================================================================


@pytest.mark.parametrize('mode', WEIGHTING_MODES)
def test_every_mode_is_accepted(mode):
    fitter = SANSFitter()
    fitter.set_weighting(mode)
    assert fitter.get_weighting() == {'mode': mode}


def test_default_and_reset():
    fitter = SANSFitter()
    assert DEFAULT_WEIGHTING_MODE == 'dI'
    assert fitter.get_weighting() == {'mode': 'dI'}
    fitter.set_weighting('none')
    fitter.set_weighting()
    assert fitter.get_weighting() == {'mode': 'dI'}


@pytest.mark.parametrize('bad', ['DI', 'di', 'sqrtI', '', None, 'SQRT'])
def test_unknown_modes_are_rejected_and_listed(bad):
    fitter = SANSFitter()
    fitter.set_weighting('abs')
    with pytest.raises(ValueError) as error:
        fitter.set_weighting(bad)
    for mode in WEIGHTING_MODES:
        assert repr(mode) in str(error.value)
    # A rejected call changes nothing.
    assert fitter.get_weighting() == {'mode': 'abs'}


def test_get_weighting_returns_a_fresh_dict():
    fitter = SANSFitter()
    fitter.get_weighting()['mode'] = 'none'
    assert fitter.get_weighting() == {'mode': 'dI'}


def test_descriptions_have_ascii_forms():
    for mode in WEIGHTING_MODES:
        setting = validate_weighting(mode)
        assert setting.describe(ascii_only=True).isascii()
        assert setting.note(ascii_only=True).isascii()
    assert validate_weighting('none').note() == 'unweighted'
    assert validate_weighting('sqrt').note(ascii_only=True) == 'sqrt(|I|)'


# =========================================================================
# 2. apply_weighting formulas
# =========================================================================


@pytest.mark.parametrize('mode', ['sqrt', 'abs', 'none'])
def test_formulas_match_sasview(awkward, mode):
    original_dy = awkward.dy.copy()
    weighted = apply_weighting(awkward, WeightingSetting(mode))

    expected = FORMULAS[mode](np.asarray(awkward.y, dtype=float))
    np.testing.assert_array_equal(weighted.dy, expected)
    assert len(weighted.dy) == len(awkward.y)
    # The input is untouched with the default copy=True.
    assert weighted is not awkward
    np.testing.assert_array_equal(awkward.dy, original_dy)


def test_di_mode_writes_nothing(awkward):
    weighted = apply_weighting(awkward, WeightingSetting('dI'))
    np.testing.assert_array_equal(weighted.dy, awkward.dy)
    assert weighted is not awkward


def test_copy_false_writes_in_place(awkward):
    weighted = apply_weighting(awkward, WeightingSetting('none'), copy=False)
    assert weighted is awkward
    np.testing.assert_array_equal(awkward.dy, np.ones(5))


@pytest.mark.parametrize('mode', ['sqrt', 'abs', 'none'])
def test_missing_dy_is_created(mode):
    data = Data1D(x=np.array([0.01, 0.02]), y=np.array([4.0, 9.0]))
    assert data.dy is None
    weighted = apply_weighting(data, WeightingSetting(mode))
    np.testing.assert_array_equal(weighted.dy, FORMULAS[mode](np.array([4.0, 9.0])))


@pytest.mark.parametrize('mode', WEIGHTING_MODES)
def test_sesans_is_left_untouched(awkward, mode):
    awkward.isSesans = True
    weighted = apply_weighting(awkward, WeightingSetting(mode))
    np.testing.assert_array_equal(weighted.dy, awkward.dy)


def test_fitter_data_is_never_modified(noisy):
    original_dy = np.array(noisy.dy, copy=True)
    fitter = make_fitter(noisy, mode='none')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fitter.fit()
    np.testing.assert_array_equal(fitter.data.dy, original_dy)


# =========================================================================
# 3. Equivalence oracle
# =========================================================================


@pytest.mark.parametrize('engine', ['bumps', 'lmfit'])
@pytest.mark.parametrize('mode', ['sqrt', 'abs'])
def test_mode_equals_a_hand_weighted_di_fit(noisy, engine, mode):
    by_mode = make_fitter(noisy, mode=mode).fit(engine=engine)

    manual = copy.deepcopy(noisy)
    manual.dy = FORMULAS[mode](np.asarray(manual.y, dtype=float))
    by_hand_fitter = make_fitter(manual, mode='dI')
    by_hand = by_hand_fitter.fit(engine=engine)

    assert by_mode['parameters']['radius']['value'] == pytest.approx(
        by_hand['parameters']['radius']['value'], rel=1e-9
    )
    assert by_mode['chisq'] == pytest.approx(by_hand['chisq'], rel=1e-9)


# =========================================================================
# 4. Noise-free data, the guard on both entry points, SESANS
# =========================================================================


def test_noise_free_data_still_refused_under_di_with_the_new_exit(noise_free):
    fitter = make_fitter(noise_free)
    with pytest.raises(ValueError, match=r"set_weighting\('none'\)") as error:
        fitter.fit()
    assert "engine='lmfit'" in str(error.value)


@pytest.mark.parametrize('engine', ['bumps', 'lmfit'])
def test_unweighted_fit_recovers_noise_free_truth(noise_free, engine):
    fitter = make_fitter(noise_free, mode='none')
    # Fixed parameters at the truth, so only the radius is identifiable.
    for name in ('scale', 'background', 'sld', 'sld_solvent'):
        fitter.set_param(name, value=noise_free.truth[name], vary=False)
    result = fitter.fit(engine=engine)
    contract = fitter._fit_contract
    index = get_fit_index(noise_free)

    residuals = contract.artifacts.residuals
    y = np.asarray(noise_free.y)[index]
    assert result['chisq'] == pytest.approx(float(np.sum(residuals**2)))
    assert result['chisq'] == pytest.approx(
        float(np.sum((y - contract.artifacts.fitted_curve) ** 2)), rel=1e-6
    )
    # Noise-free, so the identifiable parameter and the curve are recovered.
    assert result['parameters']['radius']['value'] == pytest.approx(
        noise_free.truth['radius'], rel=1e-3
    )
    np.testing.assert_allclose(contract.artifacts.fitted_curve, y, rtol=1e-3)


def test_dream_runs_under_none_and_warns_once(noise_free):
    fitter = make_fitter(noise_free, mode='none', radius=49)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = fitter.fit_bayesian(samples=300, burn=20)
    messages = [str(w.message) for w in caught if "Weighting mode 'none'" in str(w.message)]
    assert len(messages) == 1
    assert result['weighting_note'] == 'unweighted'
    assert result['parameters']['radius']['value'] == pytest.approx(50, rel=1e-2)


def test_dream_guard_raises_before_sampling(noise_free, monkeypatch):
    def must_not_run(*args, **kwargs):
        raise AssertionError('sampling started')

    monkeypatch.setattr('sans_fitter.fitter.fit_bumps_dream', must_not_run)
    fitter = make_fitter(noise_free)
    with pytest.raises(ValueError, match='DREAM sampler') as error:
        fitter.fit_bayesian(samples=100)
    message = str(error.value)
    assert "set_weighting('none')" in message
    assert "engine='lmfit'" not in message


def test_sesans_refuses_non_di_weighting_at_fit_time(noisy):
    fitter = make_fitter(noisy, mode='sqrt')
    sesans = copy.deepcopy(fitter._evaluation_data(warn=False))
    sesans.isSesans = True
    with pytest.raises(ValueError, match='SESANS data is always weighted'):
        fitter._check_fit_uncertainties('bumps', sesans)
    fitter.set_weighting('dI')
    fitter._check_fit_uncertainties('bumps', sesans)  # unaffected


# =========================================================================
# 5. Preview consistency
# =========================================================================


@pytest.mark.parametrize('mode', WEIGHTING_MODES)
def test_preview_chisq_matches_the_bumps_initial_chisq(noisy, mode):
    fitter = make_fitter(noisy, mode=mode)
    figure = fitter.plot_model(show=False)

    problem, _ = _build_bumps_problem(
        fitter._evaluation_data(warn=False),
        fitter.kernel,
        fitter._param_manager.snapshot_fit_state(),
    )
    assert f'= {problem.chisq():.4f}' in figure.layout.title.text


# =========================================================================
# 6. Zero sigma under a derived weighting
# =========================================================================


@pytest.fixture
def one_zero_intensity(noisy):
    data = copy.deepcopy(noisy)
    y = np.array(data.y, dtype=float)
    y[10] = 0.0
    data.y = y
    return normalize_sans_data(data)


def test_zero_sigma_under_abs_raises_on_bumps(one_zero_intensity):
    fitter = make_fitter(one_zero_intensity, mode='abs')
    with pytest.raises(ValueError, match="Weighting mode 'abs'") as error:
        fitter.fit()
    assert '1 of' in str(error.value)


def test_zero_sigma_under_abs_unit_weights_on_lmfit(one_zero_intensity):
    fitter = make_fitter(one_zero_intensity, mode='abs')
    with pytest.warns(UserWarning, match="Weighting mode 'abs'"):
        result = fitter.fit(engine='lmfit')
    n = int(get_fit_index(one_zero_intensity).sum())
    assert result['weighting_note'] == (f'|I| (1 of {n} points unit-weighted because sigma = 0)')


# =========================================================================
# 7. Contract
# =========================================================================


@pytest.mark.parametrize(
    'mode, note', [('dI', 'dI'), ('sqrt', 'sqrt(|I|)'), ('abs', '|I|'), ('none', 'unweighted')]
)
def test_contract_records_setting_and_note(noisy, mode, note):
    fitter = make_fitter(noisy, mode=mode)
    result = fitter.fit()
    assert result['weighting_note'] == note
    assert fitter._fit_contract.weighting == fitter.get_weighting()
    assert fitter._fit_contract.fit_context['weighting'] == {'mode': mode}


# =========================================================================
# 8. Persistence
# =========================================================================


def save(fitter, tmp_path, name='analysis.json'):
    path = os.path.join(str(tmp_path), name)
    fitter.save_analysis(path)
    return path


def test_round_trip_restores_the_mode_and_result(noisy, tmp_path):
    fitter = make_fitter(noisy, mode='abs')
    fitter.fit()
    path = save(fitter, tmp_path)

    document = json.load(open(path, encoding='utf-8'))
    assert document['schema']['version'] == SCHEMA_VERSION == 2
    assert document['weighting'] == {'mode': 'abs'}

    restored = SANSFitter.load_analysis(path, data=noisy)
    assert restored.get_weighting() == {'mode': 'abs'}
    contract = restored._fit_contract
    assert contract is not None
    assert contract.weighting == {'mode': 'abs'}
    # Rebuilt from the restored evaluation copy, so under the mode's σ.
    assert float(np.sum(contract.artifacts.residuals**2)) == pytest.approx(contract.chisq)


def test_result_is_omitted_when_the_weighting_changed(noisy, tmp_path):
    fitter = make_fitter(noisy, mode='sqrt')
    fitter.fit()
    fitter.set_weighting('dI')
    document = json.load(open(save(fitter, tmp_path), encoding='utf-8'))
    assert document['result'] is None
    assert document['result_omitted'] == 'the weighting setting changed after the fit'


def test_schema_1_document_loads_as_di_and_keeps_its_result(noisy, tmp_path):
    fitter = make_fitter(noisy)
    fitter.fit()
    path = save(fitter, tmp_path)

    # Rewrite it as 0.4 wrote it: no weighting anywhere, the key reserved.
    document = json.load(open(path, encoding='utf-8'))
    document['schema']['version'] = 1
    document['weighting'] = None
    del document['result']['weighting']
    del document['result']['fit_context']['weighting']
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(document, handle)

    restored = SANSFitter.load_analysis(path, data=noisy)
    assert restored.get_weighting() == {'mode': 'dI'}
    assert restored._fit_contract is not None
    assert restored._fit_contract.chisq == pytest.approx(fitter._fit_contract.chisq)


# =========================================================================
# 9. Outputs
# =========================================================================


def test_csv_header_has_mode_and_note(noisy, tmp_path):
    fitter = make_fitter(noisy, mode='sqrt')
    fitter.fit()
    path = os.path.join(str(tmp_path), 'results.csv')
    fitter.save_results(path)
    header = [line.strip() for line in open(path, encoding='utf-8') if line.startswith('#')]
    assert '# Weighting mode: sqrt (sigma = sqrt(|I|))' in header
    assert '# Weighting: sqrt(|I|)' in header


def test_csv_exports_the_files_own_di(noisy, tmp_path):
    fitter = make_fitter(noisy, mode='none')
    fitter.fit()
    path = os.path.join(str(tmp_path), 'results.csv')
    fitter.save_results(path)
    lines = [line.strip() for line in open(path, encoding='utf-8') if not line.startswith('#')]
    column = lines[0].split(',').index('dI_exp')
    exported = np.array([float(line.split(',')[column]) for line in lines[1:]])
    index = get_fit_index(noisy)
    np.testing.assert_allclose(exported, np.asarray(noisy.dy)[index], rtol=1e-5)


def test_html_report_has_a_weighting_row(noisy, tmp_path):
    fitter = make_fitter(noisy, mode='abs')
    fitter.fit()
    path = os.path.join(str(tmp_path), 'report.html')
    fitter.report(path)
    html = open(path, encoding='utf-8').read()
    assert 'Weighting' in html
    assert 'abs (sigma = |I|)' in html


def test_mode_survives_set_data_and_is_reported(noisy, capsys):
    fitter = SANSFitter()
    fitter.set_weighting('sqrt')
    capsys.readouterr()
    fitter.set_data(noisy)
    out = capsys.readouterr().out
    assert 'Weighting mode: sqrt' in out
    assert fitter.get_weighting() == {'mode': 'sqrt'}


# =========================================================================
# 11. Plot
# =========================================================================


def residual_axis_title(figure):
    return figure.layout.yaxis2.title.text


def test_residual_axis_reads_intensity_units_under_none(noisy):
    # The file has dI, yet the title follows the weighting.
    fitter = make_fitter(noisy, mode='none')
    fitter.fit()
    assert residual_axis_title(fitter.plot_results(show=False)) == 'Residuals (I − fit)'
    assert residual_axis_title(fitter.plot_model(show=False)) == 'Residuals (I − fit)'


@pytest.mark.parametrize('mode', ['dI', 'sqrt', 'abs'])
def test_residual_axis_reads_sigma_otherwise(noisy, mode):
    fitter = make_fitter(noisy, mode=mode)
    fitter.fit()
    assert residual_axis_title(fitter.plot_results(show=False)) == 'Residuals (σ)'


@pytest.mark.parametrize('mode', WEIGHTING_MODES)
def test_error_bars_are_the_files_di(noisy, mode):
    fitter = make_fitter(noisy, mode=mode)
    fitter.fit()
    figure = fitter.plot_results(show=False)
    index = get_fit_index(noisy)
    np.testing.assert_allclose(figure.data[0].error_y.array, np.asarray(noisy.dy)[index])


def test_residual_fallback_divides_by_the_weighting_sigma(noisy):
    fitter = make_fitter(noisy, mode='abs')
    evaluation = fitter._evaluation_data(warn=False)
    index = get_fit_index(noisy)
    y = np.asarray(noisy.y)[index]
    curve = y * 1.1
    contract = FitResultContract(
        engine='bumps',
        method='amoeba',
        chisq=1.0,
        reduced_chisq=1.0,
        n_points=int(index.sum()),
        n_free=1,
        dof=int(index.sum()) - 1,
        weighting_note='|I|',
        parameters={},
        artifacts=FitArtifacts(fitted_curve=curve, fit_index=index),  # no residuals
    )
    figure = plot_fit(
        noisy,
        contract,
        'sphere',
        show=False,
        evaluation_data=evaluation,
        weighting_mode='abs',
    )
    np.testing.assert_allclose(figure.data[-1].y, (y - curve) / np.abs(y))

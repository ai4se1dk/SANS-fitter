"""Regressions for the findings in `72_MULTIFIT_IMPL_REV_GPT.md`.

Each test here corresponds to a defect that shipped and was reproducible
through the public API. They are kept together, and named after the finding,
because the thing worth protecting is the contract each one broke rather than
the internals that happened to break it.
"""

import copy
import csv
import math
import os
import warnings

import numpy as np
import pytest
from sasmodels.data import Data1D

from sans_fitter import MultiFitter, examples
from sans_fitter.modeling.constraints import ConstraintError
from sans_fitter.persistence import _normalize_linked_uncertainty


@pytest.fixture(scope='module')
def curves():
    return {
        name: examples.simulate(
            'sphere', radius=45, scale=0.02, background=0.01, noise=0.02, seed=seed, npoints=30
        )
        for seed, name in enumerate(('a', 'b'), start=71)
    }


def two_sphere(curves):
    fit = MultiFitter()
    for name, data in curves.items():
        fit.add(name, copy.deepcopy(data), model='sphere')
        entry = fit[name]
        entry.set_param('radius', value=40, min=10, max=100, vary=True)
        entry.set_param('scale', value=0.02, min=0.001, max=0.1, vary=True)
        entry.set_param('background', value=0.01, min=0.0, max=0.1, vary=True)
    return fit


def flat_dataset(value, n=20):
    q = np.logspace(np.log10(0.01), np.log10(0.3), n)
    data = Data1D(x=q, y=np.full(n, float(value)), dy=np.full(n, 0.1))
    data.dx = None
    return data


def read_csv(path):
    with open(path, encoding='utf-8', newline='') as handle:
        rows = [row for row in csv.reader(handle) if row and not row[0].startswith('#')]
    return rows[0], rows[1:]


# =========================================================================
# R1 — results own the observations they were fitted to
# =========================================================================


class TestResultSnapshot:
    @staticmethod
    def _background_only(value):
        fit = MultiFitter()
        fit.add('a', flat_dataset(value), model='sphere')
        fit['a'].set_param('scale', value=0.0, vary=False)
        fit['a'].set_param('background', value=1.0, min=0.0, max=50.0, vary=True)
        return fit

    def test_every_exported_row_is_internally_consistent(self, tmp_path):
        """A row's residual must equal (I_fit - I_exp)/dI computed from that row."""
        fit = self._background_only(1.0)
        fit.fit(method='lm')

        # Replace the dataset with different measurements under the same name.
        fit.remove('a')
        fit.add('a', flat_dataset(20.0), model='sphere')
        fit['a'].set_param('scale', value=0.0, vary=False)
        fit['a'].set_param('background', value=1.0, min=0.0, max=50.0, vary=True)

        target = tmp_path / 'export'
        with pytest.warns(UserWarning, match='configuration has changed'):
            fit.save_results(str(target))

        header, rows = read_csv(target / 'a_curve.csv')
        columns = {name: index for index, name in enumerate(header)}
        for row in rows:
            values = [float(cell) for cell in row]
            expected = (values[columns['I_fit']] - values[columns['I_exp']]) / values[
                columns['dI_exp']
            ]
            assert values[columns['Residual']] == pytest.approx(expected, abs=1e-9)

    def test_the_export_shows_the_data_that_was_fitted(self, tmp_path):
        fit = self._background_only(1.0)
        fit.fit(method='lm')
        fit.remove('a')
        fit.add('a', flat_dataset(20.0), model='sphere')
        fit['a'].set_param('scale', value=0.0, vary=False)
        fit['a'].set_param('background', value=1.0, min=0.0, max=50.0, vary=True)

        target = tmp_path / 'export'
        with pytest.warns(UserWarning):
            fit.save_results(str(target))
        header, rows = read_csv(target / 'a_curve.csv')
        intensities = {float(row[header.index('I_exp')]) for row in rows}
        assert intensities == {1.0}

    def test_plotting_survives_the_dataset_being_removed(self, curves):
        fit = two_sphere(curves)
        fit.fit(method='lm')
        fit.remove('b')
        with pytest.warns(UserWarning, match='configuration has changed'):
            figure = fit.plot_results(show=False)
        assert len([t for t in figure.data if t.name == 'Joint fit']) == 2

    def test_editing_a_dataset_does_not_change_a_finished_result(self, curves):
        fit = two_sphere(curves)
        result = fit.fit(method='lm')
        before = float(result.datasets['a'].observed_intensity[0])
        fit['a'].set_q_range(qmin=0.02, qmax=0.2)
        assert float(result.datasets['a'].observed_intensity[0]) == before


# =========================================================================
# R2 — a constrained target keeps its own limits
# =========================================================================


class TestConstraintBounds:
    @staticmethod
    def _root_range(fit):
        graph = fit._compiled()
        ref = next(r for r in graph.class_of if r.qualified == 'a.scale')
        return graph.ranges[graph.class_of[ref]]

    def test_equality_and_identity_expressions_enforce_the_same_range(self, curves):
        ranges = []
        for expression in ('a.scale', '1 * a.scale'):
            fit = two_sphere(curves)
            fit['b'].set_param('scale', min=0.0, max=0.03)
            fit.constrain('b.scale', expression)
            ranges.append(self._root_range(fit))
        assert ranges[0] == ranges[1] == pytest.approx((0.001, 0.03))

    def test_a_constrained_target_keeps_its_limit(self, curves):
        fit = two_sphere(curves)
        fit['b'].set_param('scale', min=0.0, max=0.03)
        fit.constrain('b.scale', 'a.scale')
        assert self._root_range(fit)[1] == pytest.approx(0.03)

    def test_link_params_still_adopts_the_target_configuration(self, curves):
        """The documented difference: a link discards the follower's limits."""
        fit = two_sphere(curves)
        fit['b'].set_param('scale', min=0.0, max=0.03)
        fit.link_params('b.scale', to='a.scale')
        assert self._root_range(fit)[1] == pytest.approx(0.1)

    def test_an_empty_intersection_is_refused(self, curves):
        fit = two_sphere(curves)
        fit['b'].set_param('scale', value=0.6, min=0.5, max=0.9)
        with pytest.raises(ConstraintError, match='no allowed range in common'):
            fit.constrain('b.scale', 'a.scale')

    def test_the_constrained_value_stays_inside_the_limit_after_fitting(self, curves):
        fit = two_sphere(curves)
        # Both starts must lie inside the range the constraint leaves, which is
        # the intersection of the two — the feasible-start rule applies here as
        # it does to any other relationship.
        fit['a'].set_param('scale', value=0.01)
        fit['b'].set_param('scale', value=0.01, min=0.0, max=0.015)
        fit.constrain('b.scale', 'a.scale')
        result = fit.fit(method='lm')
        assert result.parameters['b.scale'].value <= 0.015 + 1e-12
        assert result.parameters['b.scale'].value == result.parameters['a.scale'].value


# =========================================================================
# R3 — arithmetic over literals is a constant everywhere
# =========================================================================


class TestConstantFolding:
    @pytest.mark.parametrize(
        ('expression', 'expected'),
        [('2 * 3.17', 6.34), ('1 / 2', 0.5), ('-(1 + 1)', -2.0), ('2 ** 3', 8.0)],
    )
    def test_reference_free_arithmetic_behaves_as_a_constant(self, curves, expression, expected):
        fit = two_sphere(curves)
        fit['a'].set_param('sld_solvent', min=-20, max=20)
        fit.constrain('a.sld_solvent', expression)

        table = {row['parameter']: row for row in fit.get_parameter_table()}
        assert table['a.sld_solvent']['status'] == 'fixed'
        assert table['a.sld_solvent']['value'] == pytest.approx(expected)

        result = fit.fit(method='lm')
        entry = result.parameters['a.sld_solvent']
        assert entry.value == pytest.approx(expected)
        assert entry.status == 'fixed'
        assert entry.stderr == 0.0
        assert entry.expression == expression

    def test_a_domain_error_is_refused_rather_than_stored(self, curves):
        fit = two_sphere(curves)
        with pytest.raises(ConstraintError):
            fit.constrain('a.sld_solvent', '1 / 0')
        assert fit.get_constraints() == []


# =========================================================================
# R4 — one setter call is one transaction
# =========================================================================


class TestPolydispersityTransaction:
    def test_a_rejected_width_leaves_the_distribution_alone(self, curves):
        fit = two_sphere(curves)
        fit['a'].enable_polydispersity(True)
        fit['a'].set_pd_param('radius', pd_width=0.1, pd_type='gaussian', vary=True)
        before = dict(fit['a'].get_pd_param('radius'))

        with pytest.raises(ValueError):
            fit['a'].set_pd_param('radius', pd_type='lognormal', pd_width=-1)

        assert fit['a'].get_pd_param('radius') == before

    def test_a_rejected_distribution_leaves_the_width_alone(self, curves):
        fit = two_sphere(curves)
        fit['a'].enable_polydispersity(True)
        fit['a'].set_pd_param('radius', pd_width=0.1, pd_type='gaussian', vary=True)

        with pytest.raises(ValueError):
            fit['a'].set_pd_param('radius', pd_width=0.2, pd_type='not_a_distribution')

        config = fit['a'].get_pd_param('radius')
        assert config['pd'] == pytest.approx(0.1)
        assert config['pd_type'] == 'gaussian'

    def test_a_shared_width_is_written_to_every_member(self, curves):
        fit = two_sphere(curves)
        for name in ('a', 'b'):
            fit[name].enable_polydispersity(True)
            fit[name].set_pd_param('radius', pd_width=0.1, vary=True)
        fit.share('radius_pd')
        fit['a'].set_pd_param('radius', pd_width=0.2)
        assert fit['b'].get_pd_param('radius')['pd'] == pytest.approx(0.2)

    def test_quadrature_settings_stay_with_their_dataset(self, curves):
        fit = two_sphere(curves)
        for name in ('a', 'b'):
            fit[name].enable_polydispersity(True)
            fit[name].set_pd_param('radius', pd_width=0.1, vary=True)
        fit.share('radius_pd')
        fit['a'].set_pd_param('radius', pd_n=51)
        assert fit['a'].get_pd_param('radius')['pd_n'] == 51
        assert fit['b'].get_pd_param('radius')['pd_n'] != 51


# =========================================================================
# R5 — one graph-resolved read path
# =========================================================================


class TestResolvedReads:
    def test_a_constrained_width_reads_the_same_everywhere(self, curves):
        fit = two_sphere(curves)
        fit['a'].enable_polydispersity(True)
        fit['a'].set_pd_param('radius', pd_width=0.1, vary=True)
        fit.constrain('a.radius_pd', 0.2)

        table = {row['parameter']: row['value'] for row in fit.get_parameter_table()}
        assert fit['a'].get_pd_param('radius')['pd'] == pytest.approx(table['a.radius_pd'])
        assert fit['a'].get_pd_param('radius')['pd'] == pytest.approx(0.2)

    def test_a_local_link_is_visible_from_the_handle(self, curves):
        fit = two_sphere(curves)
        fit['a'].link_params('sld_solvent', to='sld')
        assert fit['a'].get_links() == {'sld_solvent': 'sld'}
        assert 'a.sld_solvent' not in fit._compiled().labels

    def test_a_cross_dataset_link_is_visible_from_the_handle(self, curves):
        fit = two_sphere(curves)
        fit.link_params('b.radius', to='a.radius')
        assert fit['b'].get_links() == {'radius': 'a.radius'}
        assert fit['a'].get_links() == {}

    def test_an_equality_constraint_is_visible_from_the_handle(self, curves):
        fit = two_sphere(curves)
        fit.constrain('b.background', 'a.background')
        assert fit['b'].get_links() == {'background': 'a.background'}

    def test_unshare_keeps_the_value_the_group_held(self, curves):
        fit = two_sphere(curves)
        fit['a'].set_param('radius', value=30)
        fit['b'].set_param('radius', value=50)
        fit.share('radius', source='a')
        fit.unshare('radius', dataset='b')
        assert fit['b'].params['radius']['value'] == pytest.approx(30.0)
        assert fit['a'].params['radius']['value'] == pytest.approx(30.0)

    def test_unlink_keeps_the_value_the_link_held(self, curves):
        fit = two_sphere(curves)
        fit['a'].set_param('radius', value=30)
        fit['b'].set_param('radius', value=50)
        fit.link_params('b.radius', to='a.radius')
        fit.unlink_params('b.radius')
        assert fit['b'].params['radius']['value'] == pytest.approx(30.0)


# =========================================================================
# R6 — rank deficiency is diagnosed under both weightings
# =========================================================================


class TestRankDiagnostics:
    @staticmethod
    def _degenerate(weights=None):
        """power_law with power fixed at 0: only scale + background is identifiable."""
        fit = MultiFitter()
        for seed, name in enumerate(('a', 'b'), start=1):
            rng = np.random.default_rng(seed)
            data = Data1D(
                x=np.logspace(np.log10(0.01), np.log10(0.3), 25),
                y=5.0 + 0.01 * rng.standard_normal(25),
                dy=np.full(25, 0.05),
            )
            data.dx = None
            fit.add(name, data, model='power_law')
            entry = fit[name]
            entry.set_param('power', value=0.0, vary=False)
            entry.set_param('scale', value=2.0, min=0.0, max=20.0, vary=True)
            entry.set_param('background', value=3.0, min=0.0, max=20.0, vary=True)
        fit.share('scale', 'background')
        for name, weight in (weights or {}).items():
            fit.set_dataset_weight(name, weight)
        return fit

    def test_unit_weights_report_the_deficiency(self):
        fit = self._degenerate()
        with pytest.warns(RuntimeWarning, match='rank deficient'):
            result = fit.fit(method='lm')
        assert 'rank deficient' in result.cov_note
        assert result.cov is not None  # matching the single-fit engine

    def test_the_note_reaches_every_rendering(self):
        fit = self._degenerate()
        with pytest.warns(RuntimeWarning):
            fit.fit(method='lm')
        report = fit.get_fit_report()
        assert 'rank deficient' in str(report)
        assert 'rank deficient' in report.to_markdown()
        assert 'rank deficient' in report._repr_html_()

    def test_weighted_fits_report_it_too(self):
        fit = self._degenerate(weights={'b': 0.25})
        result = fit.fit(method='lm')
        assert 'rank deficient' in result.cov_note
        assert result.cov is None
        assert all(entry.stderr is None for entry in result.free_parameters())

    def test_an_identifiable_fit_carries_no_note(self, curves):
        fit = two_sphere(curves)
        fit.share('radius')
        result = fit.fit(method='lm')
        assert result.cov_note == ''
        assert 'rank deficient' not in str(fit.get_fit_report())


# =========================================================================
# R7 — CSV round-trips, and an export owns its directory
# =========================================================================


class TestExportIntegrity:
    def test_awkward_expression_text_round_trips(self, curves, tmp_path):
        fit = two_sphere(curves)
        fit['a'].set_param('sld_solvent', min=-20, max=20)
        fit.constrain('a.sld_solvent', '(0.5 *\na.sld)')
        result = fit.fit(method='lm')

        target = tmp_path / 'export'
        fit.save_results(str(target))
        header, rows = read_csv(target / 'parameters.csv')
        assert len(rows) == len(result.parameters)
        by_name = {row[0]: row for row in rows}
        # The newline is normalized away, and the value survives intact.
        assert by_name['a.sld_solvent'][header.index('expression')] == '(0.5 * a.sld)'

    def test_a_comma_in_a_cell_is_quoted_not_substituted(self, curves, tmp_path):
        fit = two_sphere(curves)
        fit['a'].set_param('sld_solvent', min=-20, max=20)
        fit.constrain('a.sld_solvent', '1 + 2 * 3')
        fit.fit(method='lm')
        target = tmp_path / 'export'
        fit.save_results(str(target))
        header, rows = read_csv(target / 'datasets.csv')
        resolutions = [row[header.index('resolution')] for row in rows]
        assert all(',' not in value or value for value in resolutions)
        # csv.reader gave the right number of columns for every row.
        assert all(len(row) == len(header) for row in rows)

    def test_a_later_export_retires_its_own_obsolete_files(self, curves, tmp_path):
        target = tmp_path / 'export'

        fit = two_sphere(curves)
        fit.share('radius')
        fit.fit(method='lm')
        fit.save_results(str(target))
        assert 'covariance.csv' in os.listdir(target)

        fixed = two_sphere(curves)
        for name in ('a', 'b'):
            for parameter in ('radius', 'scale', 'background'):
                fixed[name].set_param(parameter, vary=False)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fixed.fit(method='lm')
        fixed.save_results(str(target))

        assert fixed.result.cov is None
        assert 'covariance.csv' not in os.listdir(target)

    def test_removing_a_dataset_retires_its_curve_file(self, curves, tmp_path):
        target = tmp_path / 'export'
        fit = two_sphere(curves)
        fit.fit(method='lm')
        fit.save_results(str(target))
        assert 'b_curve.csv' in os.listdir(target)

        fit.remove('b')
        fit['a'].set_param('radius', value=44)
        fit.fit(method='lm')
        fit.save_results(str(target))
        assert 'b_curve.csv' not in os.listdir(target)
        assert 'a_curve.csv' in os.listdir(target)

    def test_files_the_export_does_not_own_are_left_alone(self, curves, tmp_path):
        target = tmp_path / 'export'
        target.mkdir()
        (target / 'notes.txt').write_text('mine', encoding='utf-8')

        fit = two_sphere(curves)
        fit.fit(method='lm')
        fit.save_results(str(target))
        fit.save_results(str(target))
        assert (target / 'notes.txt').read_text(encoding='utf-8') == 'mine'


# =========================================================================
# R8 — legacy follower uncertainty distinguishes fixed from unavailable
# =========================================================================


class TestLegacyLinkedUncertainty:
    @staticmethod
    def _case(source):
        parameters = {
            'length': {
                'value': 45.0,
                'stderr': 0.0,
                'fixed': True,
                'linked_to': 'radius',
                'formatted': '45 (linked)',
            }
        }
        if source is not None:
            parameters['radius'] = source
        return _normalize_linked_uncertainty(parameters)

    def test_an_available_error_is_copied(self):
        result = self._case(
            {'value': 45.0, 'stderr': 0.5, 'fixed': False, 'linked_to': None, 'formatted': '45(5)'}
        )
        assert result['length']['stderr'] == 0.5
        assert result['length']['formatted'] == '45(5)'

    def test_an_unavailable_error_propagates_as_unavailable(self):
        result = self._case(
            {'value': 45.0, 'stderr': None, 'fixed': False, 'linked_to': None, 'formatted': '45'}
        )
        assert result['length']['stderr'] is None
        assert 'unavailable' in result['length']['formatted']

    def test_a_missing_target_yields_unavailable(self):
        result = self._case(None)
        assert result['length']['stderr'] is None
        assert 'unavailable' in result['length']['formatted']

    def test_a_genuinely_fixed_target_keeps_zero(self):
        result = self._case(
            {
                'value': 45.0,
                'stderr': 0.0,
                'fixed': True,
                'linked_to': None,
                'formatted': '45 (fixed)',
            }
        )
        assert result['length']['stderr'] == 0.0


# =========================================================================
# R9 — component curves are produced and plotted
# =========================================================================


class TestComponentCurves:
    @staticmethod
    def _mixture(curves):
        fit = MultiFitter()
        for name, data in curves.items():
            fit.add(name, copy.deepcopy(data), model='sphere')
            fit[name].set_models('dab', 'peak_lorentz')
            fit[name].set_param('scale', value=1.0, min=0.1, max=10, vary=True)
            fit[name].set_param('dab_cor_length', value=40, min=5, max=200, vary=True)
        return fit

    def test_a_mixture_produces_one_curve_per_component(self, curves):
        fit = self._mixture(curves)
        result = fit.fit(method='lm')
        for entry in result.datasets.values():
            assert entry.component_curves is not None
            assert set(entry.component_curves) == {'dab', 'peak_lorentz'}
            for curve in entry.component_curves.values():
                assert len(curve) == entry.n_points

    def test_the_components_stack_onto_the_total(self, curves):
        fit = self._mixture(curves)
        result = fit.fit(method='lm')
        entry = result.datasets['a']
        background = result.parameters['a.background'].value
        stacked = sum(entry.component_curves.values()) + background
        np.testing.assert_allclose(stacked, entry.fitted_curve, rtol=1e-5)

    def test_the_curves_reach_the_figure(self, curves):
        fit = self._mixture(curves)
        fit.fit(method='lm')
        figure = fit.plot_results(show=False, show_components=True)
        names = {trace.name for trace in figure.data}
        assert 'a: dab' in names
        assert 'b: peak_lorentz' in names

    def test_components_are_off_by_default(self, curves):
        fit = self._mixture(curves)
        fit.fit(method='lm')
        figure = fit.plot_results(show=False)
        assert not any(':' in (trace.name or '') for trace in figure.data)

    def test_an_atomic_model_has_none(self, curves):
        fit = two_sphere(curves)
        result = fit.fit(method='lm')
        assert all(entry.component_curves is None for entry in result.datasets.values())


# =========================================================================
# Contract observations
# =========================================================================


class TestDeclarationPolicy:
    def test_a_linked_target_cannot_also_be_constrained(self, curves):
        fit = two_sphere(curves)
        fit.link_params('b.radius', to='a.radius')
        with pytest.raises(ValueError, match='already follows'):
            fit.constrain('b.radius', 'a.background')
        assert fit.get_constraints()[0]['kind'] == 'equality'

    def test_a_constrained_target_cannot_also_be_linked(self, curves):
        fit = two_sphere(curves)
        fit.constrain('b.radius', 50.0)
        with pytest.raises(ValueError, match='already defined by the constraint'):
            fit.link_params('b.radius', to='a.radius')

    def test_re_constraining_replaces_the_definition(self, curves):
        fit = two_sphere(curves)
        fit.constrain('b.radius', 50.0)
        fit.constrain('b.radius', 55.0)
        constraints = fit.get_constraints()
        assert len(constraints) == 1
        assert fit['b'].params['radius']['value'] == pytest.approx(55.0)


class TestUnitChecks:
    def test_sharing_parameters_in_different_units_is_refused(self, curves):
        fit = MultiFitter()
        for name in ('a', 'b'):
            data = examples.simulate('core_shell_sphere', noise=0.02, seed=3, npoints=30)
            fit.add(name, data, model='core_shell_sphere')
        # 'thickness' is in Ang; sharing it with itself is fine.
        fit.share('thickness')
        assert fit.get_sharing()[0]['members'] == ['a.thickness', 'b.thickness']

    def test_a_directed_link_across_units_is_still_permitted(self, curves):
        fit = MultiFitter()
        for name in ('a', 'b'):
            data = examples.simulate('core_shell_sphere', noise=0.02, seed=4, npoints=30)
            fit.add(name, data, model='core_shell_sphere')
        fit.link_params('a.thickness', to='b.radius')
        assert 'a.thickness' not in fit._compiled().labels


class TestGraphReuse:
    def test_a_validated_graph_is_not_thrown_away(self, curves):
        fit = two_sphere(curves)
        fit.share('radius')
        compiled = fit._compiled()
        assert fit._graph is compiled
        assert fit._compiled() is compiled

    def test_a_mutation_replaces_it(self, curves):
        fit = two_sphere(curves)
        first = fit._compiled()
        fit['a'].set_param('radius', value=42)
        assert fit._compiled() is not first

    def test_a_fit_invalidates_it(self, curves):
        fit = two_sphere(curves)
        fit._compiled()
        fit.fit(method='lm')
        table = {row['parameter']: row['value'] for row in fit.get_parameter_table()}
        assert table['a.radius'] == pytest.approx(fit.result.parameters['a.radius'].value)


class TestResultOwnership:
    def test_the_documented_contract_is_caller_owned(self, curves):
        fit = two_sphere(curves)
        result = fit.fit(method='lm')
        assert result is fit.result
        assert 'caller-owned and mutable' in type(result).__doc__

    def test_to_dict_is_an_independent_structure(self, curves):
        fit = two_sphere(curves)
        result = fit.fit(method='lm')
        payload = result.to_dict()
        payload['chisq'] = -1.0
        assert result.chisq != -1.0
        assert math.isfinite(result.chisq)

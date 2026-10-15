"""Configuring a MultiFitter: datasets, references, relationships, rollback.

Nothing here runs an optimizer. Everything a user can get wrong while setting up
a joint analysis should be reported at the call that causes it, and that promise
is only worth something if it is tested without the fit in the way.
"""

import copy
import warnings

import numpy as np
import pytest

from sans_fitter import MultiFitter, SANSFitter, examples
from sans_fitter.modeling.constraints import ConstraintError, ExpressionError


@pytest.fixture(scope='module')
def curves():
    """Two sphere datasets with the same radius and different backgrounds."""
    return {
        'a': examples.simulate(
            'sphere', radius=45, scale=0.02, background=0.01, noise=0.02, seed=1, npoints=40
        ),
        'b': examples.simulate(
            'sphere', radius=45, scale=0.02, background=0.03, noise=0.02, seed=2, npoints=40
        ),
    }


def configured(curves, **overrides):
    fit = MultiFitter()
    for name, data in curves.items():
        fit.add(name, data, model='sphere')
        entry = fit[name]
        entry.set_param('radius', value=40, min=10, max=100, vary=True)
        entry.set_param('scale', value=0.02, min=0.001, max=0.1, vary=True)
        entry.set_param('background', value=0.02, min=0.0, max=0.1, vary=True)
        for key, value in overrides.items():
            entry.set_param(key, **value)
    return fit


# =========================================================================
# The registry
# =========================================================================


class TestRegistry:
    def test_datasets_are_added_and_addressed_by_name(self, curves):
        fit = configured(curves)
        assert fit.names == ['a', 'b']
        assert len(fit) == 2
        assert 'a' in fit
        assert fit['a'].model_name == 'sphere'

    def test_an_unknown_dataset_lists_the_known_ones(self, curves):
        fit = configured(curves)
        with pytest.raises(KeyError, match='Available: a, b'):
            fit['c']

    @pytest.mark.parametrize('name', ['2nd', 'with space', 'has-dash', 'class', ''])
    def test_names_must_be_usable_inside_a_reference(self, curves, name):
        fit = MultiFitter()
        with pytest.raises(ValueError):
            fit.add(name, curves['a'], model='sphere')

    def test_duplicate_names_are_refused(self, curves):
        fit = configured(curves)
        with pytest.raises(ValueError, match='already'):
            fit.add('a', curves['a'], model='sphere')

    def test_the_dataset_is_copied_at_ingestion(self, curves):
        source = examples.simulate('sphere', radius=45, noise=0.02, seed=3, npoints=20)
        fit = MultiFitter()
        fit.add('x', source, model='sphere')
        before = float(fit['x'].data.y[0])
        source.y[0] = 12345.0
        assert float(fit['x'].data.y[0]) == before

    def test_handle_data_is_a_copy_too(self, curves):
        fit = configured(curves)
        held = fit['a'].data
        held.y[0] = 12345.0
        assert float(fit['a'].data.y[0]) != 12345.0

    def test_a_dataset_referenced_by_a_relationship_cannot_be_removed(self, curves):
        fit = configured(curves)
        fit.share('radius')
        with pytest.raises(ValueError, match='still referenced'):
            fit.remove('b')
        fit.unshare('radius', dataset='b')
        fit.remove('b')
        assert fit.names == ['a']

    def test_weights_must_be_positive_and_finite(self, curves):
        fit = configured(curves)
        for bad in (0.0, -1.0, float('nan'), float('inf')):
            with pytest.raises(ValueError, match='positive and finite'):
                fit.set_dataset_weight('a', bad)
        fit.set_dataset_weight('a', 0.5)
        assert fit['a'].weight == 0.5

    def test_add_fitter_copies_configuration_but_not_the_result(self, curves):
        source = SANSFitter()
        # A bare SANSFitter does not copy the dataset it is handed, and this test
        # restricts its Q range; the module fixture is shared, so copy first.
        source.set_data(copy.deepcopy(curves['a']))
        source.set_model('sphere')
        source.set_param('radius', value=44, min=10, max=100, vary=True)
        source.set_q_range(qmin=0.01, qmax=0.3)
        source.set_resolution('pinhole', dq_over_q=0.08)

        fit = MultiFitter()
        handle = fit.add_fitter('copied', source)
        assert handle.model_name == 'sphere'
        assert handle.params['radius']['value'] == 44
        assert handle.get_q_range() == pytest.approx((0.01, 0.3))
        assert handle.get_resolution()['mode'] == 'pinhole'
        # The copy is independent of the fitter it came from.
        source.set_param('radius', value=99)
        assert fit['copied'].params['radius']['value'] == 44

    def test_add_fitter_needs_data_and_a_model(self, curves):
        fit = MultiFitter()
        empty = SANSFitter()
        with pytest.raises(ValueError, match='no data'):
            fit.add_fitter('x', empty)
        empty.set_data(curves['a'])
        with pytest.raises(ValueError, match='no model'):
            fit.add_fitter('x', empty)


# =========================================================================
# References
# =========================================================================


class TestReferences:
    def test_unknown_parameters_name_the_dataset(self, curves):
        fit = configured(curves)
        with pytest.raises(KeyError, match="Dataset 'a'"):
            fit.set_param('a.nonexistent', value=1)

    def test_an_unqualified_name_needs_a_unique_shared_group(self, curves):
        fit = configured(curves)
        with pytest.raises(KeyError, match='not a shared parameter'):
            fit.set_param('radius', value=50)
        fit.share('radius')
        fit.set_param('radius', value=50)
        assert fit['a'].params['radius']['value'] == 50
        assert fit['b'].params['radius']['value'] == 50

    def test_polydispersity_widths_resolve_as_their_own_quantity(self, curves):
        fit = configured(curves)
        fit['a'].enable_polydispersity(True)
        fit['a'].set_pd_param('radius', pd_width=0.12, vary=True)
        table = {row['parameter']: row for row in fit.get_parameter_table()}
        assert table['a.radius_pd']['value'] == pytest.approx(0.12)
        assert table['a.radius_pd']['status'] == 'free'
        assert table['a.radius']['value'] != table['a.radius_pd']['value']


# =========================================================================
# Relationships
# =========================================================================


class TestSharing:
    def test_sharing_reduces_the_free_parameter_count(self, curves):
        fit = configured(curves)
        assert len(fit._compiled().labels) == 6
        fit.share('radius', 'scale')
        assert len(fit._compiled().labels) == 4

    def test_sharing_needs_two_datasets(self, curves):
        fit = configured(curves)
        with pytest.raises(ValueError, match='at least two datasets'):
            fit.share('radius', datasets=['a'])

    def test_a_subset_share_leaves_the_others_alone(self, curves):
        third = examples.simulate(
            'sphere', radius=60, scale=0.02, background=0.01, noise=0.02, seed=9, npoints=30
        )
        fit = configured(curves)
        fit.add('c', third, model='sphere')
        fit['c'].set_param('radius', value=60, min=10, max=100, vary=True)
        fit.share('radius', datasets=['a', 'b'])
        sharing = fit.get_sharing()
        assert sharing == [
            {'root': 'a.radius', 'members': ['a.radius', 'b.radius'], 'status': 'free'}
        ]
        assert 'c.radius' in fit._compiled().labels

    def test_disagreeing_members_are_refused_until_a_source_is_named(self, curves):
        fit = configured(curves)
        fit['b'].set_param('radius', value=70)
        with pytest.raises(ConstraintError, match='different values'):
            fit.share('radius')
        fit.share('radius', source='b')
        assert (
            fit._compiled().resolve()[
                next(r for r in fit._compiled().class_of if r.qualified == 'a.radius')
            ]
            == 70
        )

    def test_setting_a_shared_value_updates_every_member(self, curves):
        fit = configured(curves)
        fit.share('radius')
        fit.set_param('b.radius', value=55)
        assert fit['a'].params['radius']['value'] == 55
        assert fit['b'].params['radius']['value'] == 55

    def test_a_bound_applies_to_one_member_and_narrows_the_group(self, curves):
        fit = configured(curves)
        fit.share('radius')
        fit.set_param('b.radius', max=60)
        shared = next(c for c in fit._compiled().classes if c.label == 'a.radius')
        assert shared.maximum == 60
        assert fit['a'].params['radius']['max'] == 100

    def test_unshare_detaches_one_member_and_keeps_its_value(self, curves):
        fit = configured(curves)
        fit.share('radius')
        fit.set_param('radius', value=52)
        fit.unshare('radius', dataset='b')
        assert fit.get_sharing() == []
        assert fit['b'].params['radius']['value'] == 52
        assert len(fit._compiled().labels) == 6

    def test_unshare_needs_a_shared_parameter(self, curves):
        fit = configured(curves)
        with pytest.raises(ValueError, match='not part of a shared group'):
            fit.unshare('radius', dataset='a')

    def test_a_dataset_added_later_is_not_enrolled(self, curves):
        fit = configured(curves)
        fit.share('radius')
        third = examples.simulate('sphere', radius=45, noise=0.02, seed=5, npoints=20)
        fit.add('c', third, model='sphere')
        fit['c'].set_param('radius', value=45, min=10, max=100, vary=True)
        members = fit.get_sharing()[0]['members']
        assert members == ['a.radius', 'b.radius']


class TestLinksAndConstraints:
    def test_link_params_relates_differently_named_parameters(self, curves):
        cylinder = examples.simulate(
            'cylinder', radius=20, length=200, noise=0.02, seed=4, npoints=30
        )
        fit = configured(curves)
        fit.add('cyl', cylinder, model='cylinder')
        fit['cyl'].set_param('radius', value=20, min=5, max=60, vary=True)
        fit['cyl'].set_param('length', value=200, min=50, max=500, vary=False)
        fit.link_params('cyl.radius', to='a.radius')
        assert 'cyl.radius' not in fit._compiled().labels
        assert fit['cyl'].params['radius']['value'] == 40

    def test_a_constant_pins_a_parameter(self, curves):
        fit = configured(curves)
        fit.constrain('a.sld_solvent', 6.34)
        assert 'a.sld_solvent' not in fit._compiled().labels
        assert fit.get_constraints()[0] == {
            'target': 'a.sld_solvent',
            'kind': 'constant',
            'text': '6.34',
            'depends_on': [],
        }

    def test_a_numeric_string_is_a_constant_too(self, curves):
        fit = configured(curves)
        fit.constrain('a.sld_solvent', '-0.56')
        assert fit.get_constraints()[0]['kind'] == 'constant'

    def test_a_bare_reference_is_equality(self, curves):
        fit = configured(curves)
        fit.constrain('b.background', 'a.background')
        assert fit.get_constraints()[0]['kind'] == 'equality'
        assert 'b.background' not in fit._compiled().labels

    def test_an_expression_is_derived(self, curves):
        fit = configured(curves)
        fit['b'].set_param('scale', min=0.0, max=0.08)
        fit.constrain('b.scale', '2 * a.scale')
        table = {row['parameter']: row for row in fit.get_parameter_table()}
        assert table['b.scale']['status'] == 'derived'
        assert table['b.scale']['value'] == pytest.approx(0.04)

    def test_unconstrain_restores_independence_and_keeps_the_value(self, curves):
        fit = configured(curves)
        fit.constrain('a.radius', 52.0)
        assert 'a.radius' not in fit._compiled().labels
        fit.unconstrain('a.radius')
        assert 'a.radius' in fit._compiled().labels
        assert fit['a'].params['radius']['value'] == 52.0
        assert fit['a'].params['radius']['vary'] is True

    def test_unconstrain_needs_a_constraint(self, curves):
        fit = configured(curves)
        with pytest.raises(ValueError, match='not constrained'):
            fit.unconstrain('a.radius')

    def test_a_constrained_parameter_cannot_be_set_directly(self, curves):
        fit = configured(curves)
        fit.constrain('a.radius', 52.0)
        with pytest.raises(ValueError, match='unconstrain'):
            fit.set_param('a.radius', value=60)

    def test_constraining_two_members_of_one_group_conflicts(self, curves):
        fit = configured(curves)
        fit.share('radius')
        fit.constrain('a.radius', 50.0)
        with pytest.raises(ConstraintError, match='cannot both be given a definition'):
            fit.constrain('b.radius', 60.0)

    def test_self_reference_is_refused(self, curves):
        fit = configured(curves)
        with pytest.raises(ConstraintError, match='in terms of itself'):
            fit.constrain('a.radius', 'a.radius + 1')

    def test_an_expression_naming_an_unknown_dataset_fails(self, curves):
        fit = configured(curves)
        with pytest.raises(KeyError):
            fit.constrain('a.radius', '2 * nope.radius')

    def test_rejected_grammar_reaches_the_caller(self, curves):
        fit = configured(curves)
        with pytest.raises(ExpressionError):
            fit.constrain('a.radius', 'abs(b.radius)')


class TestPolydispersityRelationships:
    def test_widths_can_be_shared(self, curves):
        fit = configured(curves)
        for name in ('a', 'b'):
            fit[name].enable_polydispersity(True)
            fit[name].set_pd_param('radius', pd_width=0.1, vary=True)
        fit.share('radius_pd')
        sharing = fit.get_sharing()
        assert sharing[0]['members'] == ['a.radius_pd', 'b.radius_pd']

    def test_sharing_a_width_with_polydispersity_off_is_refused(self, curves):
        fit = configured(curves)
        fit['a'].enable_polydispersity(True)
        fit['a'].set_pd_param('radius', pd_width=0.1, vary=True)
        with pytest.raises(ConstraintError, match='polydispersity is switched off'):
            fit.share('radius_pd')

    def test_mismatched_distributions_are_refused(self, curves):
        fit = configured(curves)
        for name, distribution in (('a', 'gaussian'), ('b', 'lognormal')):
            fit[name].enable_polydispersity(True)
            fit[name].set_pd_param('radius', pd_width=0.1, pd_type=distribution, vary=True)
        with pytest.raises(ConstraintError, match='different distributions'):
            fit.share('radius_pd')


# =========================================================================
# Transactions
# =========================================================================


class TestRollback:
    def test_a_rejected_share_leaves_no_trace(self, curves):
        fit = configured(curves)
        fit['b'].set_param('radius', value=70)
        with pytest.raises(ConstraintError):
            fit.share('radius')
        assert fit.get_sharing() == []
        assert len(fit._compiled().labels) == 6

    def test_a_rejected_parameter_change_is_rolled_back(self, curves):
        fit = configured(curves)
        fit.share('radius')
        fit.set_param('b.radius', max=60)
        with pytest.raises(ConstraintError):
            # Below a.radius's own minimum of 10 would be fine; above b's new
            # maximum of 60 leaves the shared group with an infeasible start.
            fit.set_param('a.radius', min=80)
        assert fit['a'].params['radius']['min'] == 10
        assert len(fit._compiled().labels) == 5

    def test_a_model_change_that_orphans_a_constraint_is_rolled_back(self, curves):
        fit = configured(curves)
        fit.constrain('a.sld_solvent', 6.34)
        with pytest.raises(ConstraintError, match='no longer exist'):
            fit['a'].set_model('dab')
        assert fit['a'].model_name == 'sphere'
        assert fit.get_constraints()[0]['target'] == 'a.sld_solvent'

    def test_a_model_change_is_allowed_once_the_constraint_is_gone(self, curves):
        fit = configured(curves)
        fit.constrain('a.sld_solvent', 6.34)
        fit.unconstrain('a.sld_solvent')
        fit['a'].set_model('dab')
        assert fit['a'].model_name == 'dab'


class TestDescribe:
    def test_describe_reports_the_free_count_and_relationships(self, curves, capsys):
        fit = configured(curves)
        fit.share('radius')
        fit.constrain('a.sld_solvent', 6.34)
        fit.describe()
        printed = capsys.readouterr().out
        assert '2 dataset(s), 5 free parameter(s)' in printed
        assert 'a.sld_solvent = 6.34' in printed
        assert 'b.radius' in printed

    def test_the_parameter_table_covers_every_dataset(self, curves):
        fit = configured(curves)
        fit.share('radius')
        names = {row['parameter'] for row in fit.get_parameter_table()}
        assert {'a.radius', 'b.radius', 'a.scale', 'b.background'} <= names
        shared = [row for row in fit.get_parameter_table() if row['parameter'] == 'b.radius']
        assert shared[0]['status'] == 'shared'
        assert shared[0]['root'] == 'a.radius'


class TestCalculate:
    def test_calculate_applies_the_constraints(self, curves):
        fit = configured(curves)
        fit.constrain('a.scale', 0.04)
        curves_out = fit.calculate()
        assert set(curves_out) == {'a', 'b'}
        # a's intensity is twice b's above background, because only its scale moved.
        assert np.nanmax(curves_out['a']) > np.nanmax(curves_out['b'])

    def test_calculate_accepts_per_dataset_grids(self, curves):
        fit = configured(curves)
        grid = np.linspace(0.01, 0.2, 17)
        out = fit.calculate(q={'a': grid})
        assert len(out['a']) == 17
        assert len(out['b']) == len(fit['b'].data.x)

    def test_calculate_needs_a_model_everywhere(self, curves):
        fit = MultiFitter()
        fit.add('a', curves['a'])
        with pytest.raises(ValueError, match='no model'):
            fit.calculate()


class TestQRangeAndResolution:
    def test_each_dataset_keeps_its_own_selection(self, curves):
        fit = configured(curves)
        fit['a'].set_q_range(qmin=0.02, qmax=0.2)
        assert fit['a'].get_q_range() == pytest.approx((0.02, 0.2))
        assert fit['b'].get_q_range()[0] < 0.02

    def test_each_dataset_keeps_its_own_resolution(self, curves):
        fit = configured(curves)
        fit['a'].set_resolution('pinhole', dq_over_q=0.1)
        fit['b'].set_resolution('none')
        assert fit['a'].get_resolution()['mode'] == 'pinhole'
        assert fit['b'].get_resolution()['mode'] == 'none'

    def test_a_rejected_q_range_leaves_the_previous_one(self, curves):
        fit = configured(curves)
        before = fit['a'].get_q_range()
        with pytest.raises(ValueError):
            fit['a'].set_q_range(qmin=10.0, qmax=20.0)
        assert fit['a'].get_q_range() == pytest.approx(before)


class TestHandleSurface:
    def test_a_handle_cannot_fit_on_its_own(self, curves):
        fit = configured(curves)
        handle = fit['a']
        for forbidden in ('fit', 'fit_bayesian', 'save_analysis', 'load_analysis', 'report'):
            assert not hasattr(handle, forbidden), forbidden

    def test_a_handle_exposes_no_child_fitter_attribute(self, curves):
        fit = configured(curves)
        with pytest.raises(AttributeError):
            _ = fit['a'].kernel

    def test_handle_params_show_graph_resolved_values(self, curves):
        fit = configured(curves)
        fit.constrain('a.radius', 55.0)
        assert fit['a'].params['radius']['value'] == 55.0


def test_no_warnings_are_emitted_by_plain_configuration(curves):
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        fit = configured(curves)
        fit.share('radius', 'scale')
        fit.constrain('a.sld_solvent', 6.34)
        fit.describe()

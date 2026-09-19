"""The parameter graph on its own: parsing, equality classes, bounds, cycles.

No optimizer and no sasmodels kernel here — the graph is deliberately
backend-independent, and that is worth testing directly, because every failure
it is supposed to catch is one that would otherwise surface as a puzzling fit
rather than as an error.
"""

import math

import pytest

from sans_fitter.modeling.constraints import (
    ConstraintError,
    ConstraintSpec,
    ExpressionError,
    ParameterDescriptor,
    ParameterRef,
    ShareGroup,
    compile_graph,
    parse_expression,
    parse_reference,
)


def ref(text: str) -> ParameterRef:
    dataset, name = parse_reference(text)
    if name.endswith('_pd'):
        return ParameterRef(dataset, name[:-3], pd=True)
    return ParameterRef(dataset, name)


def descriptor(text, value=1.0, vary=True, minimum=0.0, maximum=100.0, **kwargs):
    return ParameterDescriptor(
        ref=ref(text), value=value, vary=vary, minimum=minimum, maximum=maximum, **kwargs
    )


def build(descriptors, *, local_links=None, directed=None, shares=(), constraints=None):
    return compile_graph(
        {item.ref: item for item in descriptors},
        local_links=local_links or {},
        directed_links=directed or {},
        share_groups=list(shares),
        constraints=constraints or {},
    )


def constant(target, value, minimum=0.0, maximum=10.0):
    return ConstraintSpec(target=ref(target), kind='constant', text=str(value), value=value)


def expression(target, text, resolver=ref):
    parsed = parse_expression(text, lambda dataset, local: resolver(f'{dataset}.{local}'))
    return ConstraintSpec(target=ref(target), kind='expression', text=text, expression=parsed)


# =========================================================================
# References and parsing
# =========================================================================


class TestReferences:
    def test_qualified_names_round_trip(self):
        assert ref('h2o.radius').qualified == 'h2o.radius'
        assert ref('h2o.radius_pd').qualified == 'h2o.radius_pd'

    @pytest.mark.parametrize('text', ['radius', 'a.b.c', '.radius', 'h2o.', ''])
    def test_malformed_references_are_rejected(self, text):
        with pytest.raises(ConstraintError):
            parse_reference(text)

    def test_composite_parameter_names_are_not_split_on_underscores(self):
        dataset, name = parse_reference('h2o.dab_cor_length')
        assert (dataset, name) == ('h2o', 'dab_cor_length')


class TestExpressionGrammar:
    def test_arithmetic_is_parsed_and_evaluated(self):
        parsed = parse_expression('2 * a.radius + 10', lambda d, n: ref(f'{d}.{n}'))
        assert parsed.refs == {ref('a.radius')}
        assert parsed.evaluate({ref('a.radius'): 5.0}) == 20.0

    def test_gradient_is_exact_for_the_supported_grammar(self):
        parsed = parse_expression('a.x * b.y - 3 / b.y', lambda d, n: ref(f'{d}.{n}'))
        values = {ref('a.x'): 2.0, ref('b.y'): 4.0}
        gradient = parsed.gradient(values)
        assert gradient[ref('a.x')] == pytest.approx(4.0)
        assert gradient[ref('b.y')] == pytest.approx(2.0 + 3.0 / 16.0)

    @pytest.mark.parametrize(
        'text',
        [
            "__import__('os')",
            'abs(a.radius)',
            'a.radius.real',
            'radius',
            '[a.radius]',
            'a.radius if 1 else 2',
            'a.radius > 2',
            'a.radius ** 9',
            'a.radius ** 0.5',
            'a.radius ** b.radius',
        ],
    )
    def test_everything_outside_the_grammar_is_rejected(self, text):
        with pytest.raises(ExpressionError):
            parse_expression(text, lambda d, n: ref(f'{d}.{n}'))

    def test_length_and_depth_are_capped(self):
        with pytest.raises(ExpressionError, match='characters long'):
            parse_expression('a.x + ' * 200 + 'a.x', lambda d, n: ref(f'{d}.{n}'))
        # Redundant parentheses leave no trace in the AST, so depth has to come
        # from real operators.
        with pytest.raises(ExpressionError, match='nests deeper'):
            parse_expression('1+' * 40 + 'a.x', lambda d, n: ref(f'{d}.{n}'))

    def test_repeated_references_cancel_in_the_affine_form(self):
        parsed = parse_expression('a.x - a.x + 3', lambda d, n: ref(f'{d}.{n}'))
        affine = parsed.affine(lambda r: r)
        assert affine is not None
        assert affine.coefficients == {}
        assert affine.constant == 3.0

    def test_division_by_a_varying_quantity_is_not_affine(self):
        parsed = parse_expression('a.x / b.y', lambda d, n: ref(f'{d}.{n}'))
        assert parsed.affine(lambda r: r) is None


# =========================================================================
# Equality classes
# =========================================================================


class TestEqualityClasses:
    def test_sharing_collapses_members_into_one_free_root(self):
        graph = build(
            [descriptor('a.radius', 45.0), descriptor('b.radius', 45.0)],
            shares=[ShareGroup((ref('a.radius'), ref('b.radius')), label='radius')],
        )
        assert graph.n_free == 1
        assert graph.labels == ['a.radius']
        assert graph.classes[graph.class_of[ref('b.radius')]].shared

    def test_unrelated_datasets_stay_independent(self):
        graph = build(
            [descriptor('a.radius'), descriptor('b.radius'), descriptor('c.radius')],
            shares=[ShareGroup((ref('a.radius'), ref('b.radius')), label='radius')],
        )
        assert graph.n_free == 2
        assert graph.labels == ['a.radius', 'c.radius']

    def test_labels_do_not_depend_on_insertion_order_or_link_direction(self):
        forward = build(
            [descriptor('b.radius'), descriptor('a.radius')],
            shares=[ShareGroup((ref('b.radius'), ref('a.radius')), label='radius')],
        )
        backward = build(
            [descriptor('a.radius'), descriptor('b.radius')],
            shares=[ShareGroup((ref('a.radius'), ref('b.radius')), label='radius')],
        )
        assert forward.labels == backward.labels == ['a.radius']

    def test_directed_links_adopt_the_targets_configuration(self):
        graph = build(
            [
                descriptor('a.radius', 45.0, minimum=10.0, maximum=100.0),
                descriptor('b.radius', 12.0, vary=False, minimum=0.0, maximum=20.0),
            ],
            directed={ref('b.radius'): ref('a.radius')},
        )
        entry = graph.classes[graph.class_of[ref('b.radius')]]
        assert entry.kind == 'free'
        assert (entry.value, entry.minimum, entry.maximum) == (45.0, 10.0, 100.0)

    def test_sharing_intersects_bounds(self):
        graph = build(
            [
                descriptor('a.radius', 45.0, minimum=10.0, maximum=100.0),
                descriptor('b.radius', 45.0, minimum=20.0, maximum=60.0),
            ],
            shares=[ShareGroup((ref('a.radius'), ref('b.radius')), label='radius')],
        )
        entry = graph.classes[graph.class_of[ref('a.radius')]]
        assert (entry.minimum, entry.maximum) == (20.0, 60.0)

    def test_sharing_rejects_bounds_that_do_not_overlap(self):
        with pytest.raises(ConstraintError, match='no allowed range'):
            build(
                [
                    descriptor('a.radius', 45.0, minimum=10.0, maximum=30.0),
                    descriptor('b.radius', 45.0, minimum=50.0, maximum=90.0),
                ],
                shares=[ShareGroup((ref('a.radius'), ref('b.radius')), label='radius')],
            )

    def test_sharing_rejects_disagreeing_values(self):
        with pytest.raises(ConstraintError, match='different values'):
            build(
                [descriptor('a.radius', 45.0), descriptor('b.radius', 30.0)],
                shares=[ShareGroup((ref('a.radius'), ref('b.radius')), label='radius')],
            )

    def test_sharing_rejects_disagreeing_vary_flags(self):
        with pytest.raises(ConstraintError, match='whether to vary'):
            build(
                [descriptor('a.radius', 5.0), descriptor('b.radius', 5.0, vary=False)],
                shares=[ShareGroup((ref('a.radius'), ref('b.radius')), label='radius')],
            )

    def test_an_explicit_source_settles_a_disagreement(self):
        graph = build(
            [descriptor('a.radius', 45.0), descriptor('b.radius', 30.0)],
            shares=[
                ShareGroup(
                    (ref('a.radius'), ref('b.radius')), source=ref('b.radius'), label='radius'
                )
            ],
        )
        entry = graph.classes[graph.class_of[ref('a.radius')]]
        assert entry.value == 30.0
        assert entry.label == 'b.radius'

    def test_an_equality_chain_collapses_to_one_root(self):
        graph = build(
            [descriptor('a.x'), descriptor('b.x'), descriptor('c.x')],
            directed={ref('c.x'): ref('b.x'), ref('b.x'): ref('a.x')},
        )
        assert graph.n_free == 1
        assert len(graph.classes[graph.class_of[ref('c.x')]].members) == 3


class TestPolydispersityRules:
    def test_widths_with_different_distributions_cannot_be_related(self):
        with pytest.raises(ConstraintError, match='different distributions'):
            build(
                [
                    descriptor('a.radius_pd', 0.1, maximum=1.0, pd_type='gaussian'),
                    descriptor('b.radius_pd', 0.1, maximum=1.0, pd_type='lognormal'),
                ],
                shares=[ShareGroup((ref('a.radius_pd'), ref('b.radius_pd')), label='radius_pd')],
            )

    def test_a_width_cannot_be_related_to_an_ordinary_parameter(self):
        with pytest.raises(ConstraintError, match='different quantities'):
            build(
                [
                    descriptor('a.radius_pd', 0.1, maximum=1.0, pd_type='gaussian'),
                    descriptor('b.radius', 0.1),
                ],
                directed={ref('b.radius'): ref('a.radius_pd')},
            )

    def test_a_width_on_an_entry_with_polydispersity_off_is_rejected(self):
        with pytest.raises(ConstraintError, match='disabled'):
            build(
                [
                    descriptor('a.radius_pd', 0.1, maximum=1.0, pd_type='gaussian'),
                    descriptor(
                        'b.radius_pd', 0.1, maximum=1.0, pd_type='gaussian', pd_enabled=False
                    ),
                ],
                shares=[ShareGroup((ref('a.radius_pd'), ref('b.radius_pd')), label='radius_pd')],
            )

    def test_matching_widths_share_normally(self):
        graph = build(
            [
                descriptor('a.radius_pd', 0.12, maximum=1.0, pd_type='gaussian'),
                descriptor('b.radius_pd', 0.12, maximum=1.0, pd_type='gaussian'),
            ],
            shares=[ShareGroup((ref('a.radius_pd'), ref('b.radius_pd')), label='radius_pd')],
        )
        assert graph.n_free == 1
        assert graph.classes[graph.free_roots[0]].pd


# =========================================================================
# Constraints
# =========================================================================


class TestConstraints:
    def test_a_constant_removes_a_coordinate(self):
        graph = build(
            [descriptor('a.sld_solvent', 1.0, minimum=-10.0, maximum=10.0)],
            constraints={ref('a.sld_solvent'): constant('a.sld_solvent', 6.34)},
        )
        assert graph.n_free == 0
        assert graph.resolve()[ref('a.sld_solvent')] == 6.34

    def test_a_constant_outside_the_bounds_is_refused(self):
        with pytest.raises(ConstraintError, match='outside the allowed range'):
            build(
                [descriptor('a.sld_solvent', 1.0, minimum=-1.0, maximum=1.0)],
                constraints={ref('a.sld_solvent'): constant('a.sld_solvent', 6.34)},
            )

    def test_an_expression_is_derived_and_resolves(self):
        graph = build(
            [descriptor('a.radius', 5.0, maximum=20.0), descriptor('b.length', 20.0, maximum=60.0)],
            constraints={ref('b.length'): expression('b.length', '2 * a.radius + 10')},
        )
        assert graph.n_free == 1
        assert graph.resolve()[ref('b.length')] == 20.0
        assert graph.resolve([7.0])[ref('b.length')] == 24.0

    def test_derived_gradients_chain_through_the_graph(self):
        graph = build(
            [
                descriptor('a.x', 2.0, maximum=4.0),
                descriptor('b.y', 4.0, maximum=8.0),
                descriptor('c.z', 8.0, maximum=20.0),
            ],
            constraints={
                ref('b.y'): expression('b.y', '2 * a.x'),
                ref('c.z'): expression('c.z', '2 * b.y'),
            },
        )
        gradients = graph.gradients()
        assert gradients[graph.class_of[ref('b.y')]][0] == pytest.approx(2.0)
        assert gradients[graph.class_of[ref('c.z')]][0] == pytest.approx(4.0)

    def test_two_definitions_of_one_quantity_conflict(self):
        with pytest.raises(ConstraintError, match='cannot both be given a definition'):
            build(
                [descriptor('a.x', 1.0), descriptor('b.x', 1.0)],
                shares=[ShareGroup((ref('a.x'), ref('b.x')), label='x')],
                constraints={
                    ref('a.x'): constant('a.x', 2.0),
                    ref('b.x'): constant('b.x', 3.0),
                },
            )

    def test_a_dependency_cycle_names_the_path(self):
        with pytest.raises(ConstraintError, match='cycle'):
            build(
                [descriptor('a.x', 1.0, maximum=100.0), descriptor('b.y', 1.0, maximum=100.0)],
                constraints={
                    ref('a.x'): expression('a.x', '2 * b.y'),
                    ref('b.y'): expression('b.y', '2 * a.x'),
                },
            )

    def test_self_reference_is_rejected(self):
        with pytest.raises(ConstraintError):
            build(
                [descriptor('a.x', 1.0, maximum=100.0)],
                constraints={ref('a.x'): expression('a.x', 'a.x + 1')},
            )


class TestBoundEnforcement:
    def test_a_one_root_affine_constraint_narrows_its_root(self):
        graph = build(
            [
                descriptor('a.scale', 0.01, minimum=0.001, maximum=0.1),
                descriptor('b.scale', 0.02, minimum=0.0, maximum=0.03),
            ],
            constraints={ref('b.scale'): expression('b.scale', '2 * a.scale')},
        )
        root = graph.class_of[ref('a.scale')]
        assert graph.ranges[root] == pytest.approx((0.001, 0.015))

    def test_a_start_outside_the_narrowed_range_is_refused(self):
        with pytest.raises(ConstraintError, match='outside the range'):
            build(
                [
                    descriptor('a.scale', 0.05, minimum=0.001, maximum=0.1),
                    descriptor('b.scale', 0.1, minimum=0.0, maximum=0.03),
                ],
                constraints={ref('b.scale'): expression('b.scale', '2 * a.scale')},
            )

    def test_an_impossible_constraint_names_both_ranges(self):
        with pytest.raises(ConstraintError, match='does not overlap'):
            build(
                [
                    descriptor('a.x', 5.0, minimum=5.0, maximum=10.0),
                    descriptor('b.y', 50.0, minimum=100.0, maximum=200.0),
                ],
                constraints={ref('b.y'): expression('b.y', '2 * a.x')},
            )

    def test_a_divisor_reaching_zero_is_refused_with_advice(self):
        with pytest.raises(ConstraintError, match='divisor can reach zero'):
            build(
                [
                    descriptor('a.x', 1.0, minimum=-1.0, maximum=1.0),
                    descriptor('b.y', 1.0, minimum=0.0, maximum=100.0),
                ],
                constraints={ref('b.y'): expression('b.y', '1 / a.x')},
            )

    def test_shared_aliases_cancel_exactly_rather_than_spanning_an_interval(self):
        # Interval arithmetic on 'a.x - b.x' would give [-9, 9] and be refused
        # against a target bounded to [0, 1]. The affine form sees one class.
        graph = build(
            [
                descriptor('a.x', 5.0, minimum=1.0, maximum=10.0),
                descriptor('b.x', 5.0, minimum=1.0, maximum=10.0),
                descriptor('c.y', 0.5, minimum=0.0, maximum=1.0),
            ],
            shares=[ShareGroup((ref('a.x'), ref('b.x')), label='x')],
            constraints={ref('c.y'): expression('c.y', 'a.x - b.x + 0.5')},
        )
        assert graph.resolve()[ref('c.y')] == pytest.approx(0.5)

    def test_a_free_root_needs_finite_non_degenerate_bounds(self):
        with pytest.raises(ConstraintError, match='non-finite bound'):
            build([descriptor('a.x', 1.0, minimum=0.0, maximum=math.inf)])
        with pytest.raises(ConstraintError, match='collapsed to the single value'):
            build([descriptor('a.x', 1.0, minimum=1.0, maximum=1.0)])

    def test_a_start_outside_its_own_bounds_is_refused(self):
        with pytest.raises(ConstraintError, match='outside its range'):
            build([descriptor('a.x', 50.0, minimum=0.0, maximum=10.0)])


class TestResolution:
    def test_local_links_are_mandatory_edges(self):
        graph = build(
            [descriptor('a.radius_effective', 1.0), descriptor('a.radius', 45.0, maximum=100.0)],
            local_links={ref('a.radius_effective'): ref('a.radius')},
        )
        assert graph.n_free == 1
        assert graph.resolve()[ref('a.radius_effective')] == 45.0

    def test_resolve_applies_root_values_to_every_member(self):
        graph = build(
            [descriptor('a.r', 5.0, maximum=10.0), descriptor('b.r', 5.0, maximum=10.0)],
            shares=[ShareGroup((ref('a.r'), ref('b.r')), label='r')],
        )
        resolved = graph.resolve([7.5])
        assert resolved[ref('a.r')] == resolved[ref('b.r')] == 7.5

    def test_start_vector_follows_the_label_order(self):
        graph = build([descriptor('b.x', 2.0, maximum=9.0), descriptor('a.x', 1.0, maximum=9.0)])
        assert graph.labels == ['a.x', 'b.x']
        assert list(graph.start_vector()) == [1.0, 2.0]

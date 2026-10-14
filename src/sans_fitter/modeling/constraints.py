"""The parameter graph behind a simultaneous fit, with no optimizer in sight.

A multi-dataset fit is two problems wearing one coat. The first is numerical and
belongs to bumps: evaluate N models, concatenate residuals, minimise. The second
is bookkeeping, and is the one that goes wrong quietly — which parameters are the
same quantity under different names, which are computed from others, how many
independent coordinates that leaves, and which bounds actually constrain the fit.
This module owns the second problem and knows nothing about the first, so the
same graph can drive bumps today and scipy later without being redesigned.

Three relationships reach it:

- **Equality.** ``share()`` (symmetric), ``link_params()`` (directed), and the
  per-dataset links a :class:`~sans_fitter.SANSFitter` already carries
  (``link_params``, ``shared=`` on ``set_models``, ``radius_effective_mode``).
  These collapse into equivalence classes: one class is one number.
- **Constants.** A class pinned to a value contributes no coordinate.
- **Arithmetic.** A class computed from other classes through a small expression
  language, likewise contributing no coordinate but carrying a derivative, so its
  uncertainty can be propagated from the roots it depends on.

The vocabulary, used consistently below: a **ref** is one parameter of one
dataset; a **class** is a set of refs that are the same number; a **root** is a
class whose value is not computed from anything else; and a *free* root is a root
the optimizer moves. ``n_free`` is a count of free roots, never of refs — that is
the whole point of counting them here rather than asking the optimizer.

The three ways of saying "these are equal" differ deliberately, and the
difference is in what the target gives up:

- ``link_params(a, to=b)`` says "a follows b", so b's value, vary flag **and
  bounds** win. The follower's own configuration is discarded wholesale, which
  is what the single-dataset API has always meant.
- ``share('radius')`` says "these are one measurement", so the members must
  already agree (or name a ``source``) and their bounds **intersect**. Picking
  one member's bounds would discard a limit set on another.
- ``constrain(target, 'other.parameter')`` defines the target's *value* and
  nothing else, so the target keeps its own limits and they intersect as in a
  share. Anything else would make ``constrain(x, 'y')`` and
  ``constrain(x, '1 * y')`` — the same relationship, spelled two ways — permit
  different values.
"""

import ast
import math
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Longest constraint expression accepted, in characters. A constraint is a
#: physical relationship between a handful of parameters; anything approaching
#: this length is a program, and this is not a language for programs.
MAX_EXPRESSION_LENGTH = 500
#: Deepest expression tree accepted. Guards the recursive walkers below against
#: a pathological input long before Python's own recursion limit.
MAX_EXPRESSION_DEPTH = 24
#: Largest |exponent| in a constant power. Beyond this the interval arithmetic
#: that certifies bounds stops being informative.
MAX_ABS_POWER = 4
#: How close two share-group members' values must be to count as agreeing.
VALUE_AGREEMENT_RTOL = 1e-9
#: Iteration cap for propagating target bounds back onto root ranges.
MAX_BOUND_TIGHTENING_PASSES = 16

__all__ = [
    'ConstraintError',
    'ExpressionError',
    'ParameterRef',
    'ParameterDescriptor',
    'ShareGroup',
    'ConstraintSpec',
    'ParameterClass',
    'CompiledGraph',
    'parse_expression',
    'parse_reference',
    'compile_graph',
    'MAX_ABS_POWER',
    'MAX_EXPRESSION_DEPTH',
    'MAX_EXPRESSION_LENGTH',
]


class ConstraintError(ValueError):
    """A parameter graph that cannot be compiled into a well-posed fit."""


class ExpressionError(ConstraintError):
    """A constraint expression outside the supported grammar or domain."""


# =========================================================================
# References
# =========================================================================


@dataclass(frozen=True, slots=True, order=True)
class ParameterRef:
    """One parameter of one dataset: the atom the graph is built from.

    *name* is the entry-local parameter name **after alias resolution**, so
    ``h2o.sld`` and ``h2o.A_sld`` under one composite model are the same ref
    rather than two that happen to move together. *pd* marks a polydispersity
    width, which is a distinct quantity from the parameter it belongs to
    (``radius`` and ``radius_pd`` are separate refs with the same *name*).
    """

    dataset: str
    name: str
    pd: bool = False

    @property
    def qualified(self) -> str:
        """``dataset.parameter`` — the public spelling, and the covariance label."""
        return f'{self.dataset}.{self.name}_pd' if self.pd else f'{self.dataset}.{self.name}'

    def __str__(self) -> str:
        return self.qualified


def parse_reference(text: str) -> tuple[str, str]:
    """Split ``'h2o.radius'`` into ``('h2o', 'radius')``.

    Exactly one dot separates the two halves. Composite parameter names contain
    underscores (``dab_cor_length``) and nothing else that could be mistaken for
    a separator, so no name is ever split on anything but that dot.

    Raises:
        ConstraintError: If the text is not a single ``dataset.parameter`` pair.
    """
    if not isinstance(text, str):
        raise ConstraintError(f'A parameter reference must be a string, got {type(text).__name__}.')
    parts = text.strip().split('.')
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ConstraintError(
            f"Invalid parameter reference '{text}'. Use exactly 'dataset.parameter', "
            "for example 'h2o.radius' or 'h2o.radius_pd'."
        )
    return parts[0], parts[1]


# =========================================================================
# Expression language (release B)
# =========================================================================


@dataclass(frozen=True, slots=True)
class Constant:
    """A finite numeric literal."""

    value: float


@dataclass(frozen=True, slots=True)
class Reference:
    """A qualified parameter reference."""

    ref: ParameterRef


@dataclass(frozen=True, slots=True)
class Negate:
    """Unary minus."""

    operand: Any


@dataclass(frozen=True, slots=True)
class Binary:
    """One of ``+``, ``-``, ``*``, ``/``."""

    op: str
    left: Any
    right: Any


@dataclass(frozen=True, slots=True)
class Power:
    """A node raised to a constant integer exponent."""

    base: Any
    exponent: int


Node = Constant | Reference | Negate | Binary | Power


@dataclass(frozen=True, slots=True)
class Expression:
    """A validated constraint expression and the references it reads."""

    text: str
    root: Node
    refs: frozenset[ParameterRef]

    def evaluate(self, values: Mapping[ParameterRef, float]) -> float:
        """Value of the expression at the given reference values."""
        return _evaluate(self.root, values)

    def gradient(self, values: Mapping[ParameterRef, float]) -> dict[ParameterRef, float]:
        """Partial derivative with respect to each reference it reads."""
        partials: dict[ParameterRef, float] = dict.fromkeys(self.refs, 0.0)
        _accumulate_gradient(self.root, values, 1.0, partials)
        return partials

    def interval(
        self,
        boxes: Mapping[Hashable, tuple[float, float]],
        key: Callable[[ParameterRef], Hashable],
    ) -> tuple[float, float]:
        """Conservative range over a box of dependency ranges.

        *key* maps a reference onto whatever the box is indexed by — the
        equivalence class it belongs to, in practice, so two aliases of one
        shared parameter share a single interval instead of being treated as
        independently varying.
        """
        return _interval(self.root, boxes, key)

    def affine(self, key: Callable[[ParameterRef], Hashable]) -> 'AffineForm | None':
        """Normalized ``constant + Σ coefficient·key`` form, or None if nonlinear.

        Keyed by equivalence class rather than by reference, so
        ``h2o.radius - d2o.radius`` is recognised as identically zero once the
        two are shared. Repeated coefficients are combined here and nowhere
        else, which is what makes that cancellation exact rather than an
        interval straddling zero.
        """
        return _affine(self.root, key)


@dataclass(frozen=True, slots=True)
class AffineForm:
    """``constant + Σ coefficients[k]·k``, with zero coefficients dropped."""

    constant: float
    coefficients: dict[Hashable, float]

    def range_over(self, boxes: Mapping[Hashable, tuple[float, float]]) -> tuple[float, float]:
        """Exact extrema over a box. Exact because the form is linear in each key."""
        low = high = self.constant
        for key, coefficient in self.coefficients.items():
            lo, hi = boxes[key]
            products = (coefficient * lo, coefficient * hi)
            low += min(products)
            high += max(products)
        return low, high


def parse_expression(text: str, resolve: Callable[[str, str], ParameterRef]) -> Expression:
    """Parse and validate one constraint expression.

    The text is parsed with :func:`ast.parse` and then walked node by node
    against an allowlist. It is never handed to ``eval`` or ``exec``, and no
    symbol table is exposed: an expression is data that this module interprets,
    which is also what lets a saved analysis carry one safely.

    Supported: finite numeric literals, qualified ``dataset.parameter``
    references, parentheses, unary ``+``/``-``, the four arithmetic operators,
    and ``**`` by a constant integer exponent up to :data:`MAX_ABS_POWER`.
    Rejected: calls, indexing, comparisons, comprehensions, attribute chains
    deeper than one dot, names without a dataset, and every form of assignment
    or import.

    Args:
        text: The expression source.
        resolve: Turns ``(dataset, local_name)`` into a :class:`ParameterRef`,
            raising for an unknown dataset or parameter. Supplied by the caller
            because only it knows the datasets and their alias layers.

    Raises:
        ExpressionError: On anything outside the grammar above.
    """
    if not isinstance(text, str):
        raise ExpressionError(
            f'A constraint expression must be a string, got {type(text).__name__}.'
        )
    source = text.strip()
    if not source:
        raise ExpressionError('A constraint expression must not be empty.')
    if len(source) > MAX_EXPRESSION_LENGTH:
        raise ExpressionError(
            f'Constraint expression is {len(source)} characters long; the limit is '
            f'{MAX_EXPRESSION_LENGTH}. Constraints describe a relationship between a '
            'few parameters, not a computation.'
        )

    try:
        tree = ast.parse(source, mode='eval')
    except SyntaxError as error:
        raise ExpressionError(
            f"Could not parse constraint expression '{text}': {error.msg}"
        ) from error

    node = _convert(tree.body, source, resolve, depth=0)
    return Expression(text=source, root=node, refs=frozenset(_node_refs(node)))


def _convert(
    node: ast.AST, source: str, resolve: Callable[[str, str], ParameterRef], depth: int
) -> Node:
    """Translate one validated AST node, rejecting everything not allowlisted."""
    if depth > MAX_EXPRESSION_DEPTH:
        raise ExpressionError(
            f"Constraint expression '{source}' nests deeper than {MAX_EXPRESSION_DEPTH} levels."
        )

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionError(
                f"Constraint expression '{source}' contains the literal {node.value!r}; "
                'only finite numbers are allowed.'
            )
        value = float(node.value)
        if not math.isfinite(value):
            raise ExpressionError(
                f"Constraint expression '{source}' contains the non-finite literal {value}."
            )
        return Constant(value)

    if isinstance(node, ast.Attribute):
        # 'h2o.radius' parses as Attribute(Name('h2o'), 'radius'). Requiring the
        # value to be a bare Name is what stops 'a.b.c' and any attribute walk
        # into a real object from being mistaken for a reference.
        if not isinstance(node.value, ast.Name):
            raise ExpressionError(
                f"Constraint expression '{source}' contains an attribute chain. "
                "References are exactly 'dataset.parameter'."
            )
        return Reference(resolve(node.value.id, node.attr))

    if isinstance(node, ast.Name):
        raise ExpressionError(
            f"Constraint expression '{source}' uses the bare name '{node.id}'. "
            "Qualify every parameter with its dataset, for example 'h2o.{0}'.".format(node.id)
        )

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.UAdd):
            return _convert(node.operand, source, resolve, depth + 1)
        if isinstance(node.op, ast.USub):
            return Negate(_convert(node.operand, source, resolve, depth + 1))
        raise ExpressionError(
            f"Constraint expression '{source}' uses an unsupported unary operator."
        )

    if isinstance(node, ast.BinOp):
        operators = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/'}
        for node_type, symbol in operators.items():
            if isinstance(node.op, node_type):
                return Binary(
                    symbol,
                    _convert(node.left, source, resolve, depth + 1),
                    _convert(node.right, source, resolve, depth + 1),
                )
        if isinstance(node.op, ast.Pow):
            return Power(
                _convert(node.left, source, resolve, depth + 1),
                _integer_exponent(node.right, source),
            )
        raise ExpressionError(
            f"Constraint expression '{source}' uses an unsupported operator. "
            'Supported: + - * / and ** with a constant integer exponent.'
        )

    raise ExpressionError(
        f"Constraint expression '{source}' contains {type(node).__name__}, which is not "
        'part of the constraint grammar. Calls, indexing, comparisons and comprehensions '
        'are all rejected.'
    )


def _integer_exponent(node: ast.AST, source: str) -> int:
    """The exponent of a ``**``, which must be a small constant integer."""
    sign = 1
    while isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        if isinstance(node.op, ast.USub):
            sign = -sign
        node = node.operand
    if not isinstance(node, ast.Constant) or isinstance(node.value, bool):
        raise ExpressionError(
            f"Constraint expression '{source}' raises to a non-constant power. "
            'Only constant integer exponents are supported.'
        )
    value = node.value
    if isinstance(value, float) and not value.is_integer():
        raise ExpressionError(
            f"Constraint expression '{source}' uses the fractional exponent {value}. "
            'Only integer exponents are supported; a fractional power has a domain '
            'restriction this release does not certify.'
        )
    exponent = sign * int(value)
    if abs(exponent) > MAX_ABS_POWER:
        raise ExpressionError(
            f"Constraint expression '{source}' uses the exponent {exponent}; "
            f'the supported range is -{MAX_ABS_POWER}..{MAX_ABS_POWER}.'
        )
    return exponent


def _node_refs(node: Node) -> set[ParameterRef]:
    if isinstance(node, Reference):
        return {node.ref}
    if isinstance(node, Constant):
        return set()
    if isinstance(node, Negate):
        return _node_refs(node.operand)
    if isinstance(node, Power):
        return _node_refs(node.base)
    return _node_refs(node.left) | _node_refs(node.right)


def _evaluate(node: Node, values: Mapping[ParameterRef, float]) -> float:
    if isinstance(node, Constant):
        return node.value
    if isinstance(node, Reference):
        return float(values[node.ref])
    if isinstance(node, Negate):
        return -_evaluate(node.operand, values)
    if isinstance(node, Power):
        base = _evaluate(node.base, values)
        if base == 0.0 and node.exponent < 0:
            raise ExpressionError('Constraint evaluates 0 raised to a negative power.')
        return float(base**node.exponent)
    left = _evaluate(node.left, values)
    right = _evaluate(node.right, values)
    if node.op == '+':
        return left + right
    if node.op == '-':
        return left - right
    if node.op == '*':
        return left * right
    if right == 0.0:
        raise ExpressionError('Constraint evaluates a division by zero.')
    return left / right


def _accumulate_gradient(
    node: Node,
    values: Mapping[ParameterRef, float],
    seed: float,
    out: dict[ParameterRef, float],
) -> None:
    """Reverse-mode accumulation of ∂expression/∂reference.

    Exact differentiation of the parsed tree, not a finite difference: the
    grammar is four operators and an integer power, so the derivative rules are
    short, and a numerical step would add an error term to an uncertainty that
    is supposed to be exact for the affine constraints users actually write.
    """
    if isinstance(node, Constant):
        return
    if isinstance(node, Reference):
        out[node.ref] = out.get(node.ref, 0.0) + seed
        return
    if isinstance(node, Negate):
        _accumulate_gradient(node.operand, values, -seed, out)
        return
    if isinstance(node, Power):
        base = _evaluate(node.base, values)
        if base == 0.0 and node.exponent < 1:
            raise ExpressionError('Constraint derivative is undefined at a zero base.')
        _accumulate_gradient(
            node.base, values, seed * node.exponent * base ** (node.exponent - 1), out
        )
        return
    if node.op in ('+', '-'):
        _accumulate_gradient(node.left, values, seed, out)
        _accumulate_gradient(node.right, values, seed if node.op == '+' else -seed, out)
        return
    left = _evaluate(node.left, values)
    right = _evaluate(node.right, values)
    if node.op == '*':
        _accumulate_gradient(node.left, values, seed * right, out)
        _accumulate_gradient(node.right, values, seed * left, out)
        return
    if right == 0.0:
        raise ExpressionError('Constraint derivative divides by zero.')
    _accumulate_gradient(node.left, values, seed / right, out)
    _accumulate_gradient(node.right, values, -seed * left / (right * right), out)


def _interval(
    node: Node,
    boxes: Mapping[Hashable, tuple[float, float]],
    key: Callable[[ParameterRef], Hashable],
) -> tuple[float, float]:
    """Interval arithmetic, conservative by construction.

    Repeated references are *not* correlated here — the interval of ``x - x``
    is ``[lo - hi, hi - lo]``, not zero. That over-estimate is why
    :meth:`Expression.affine` is tried first and this is only the fallback: an
    honest over-estimate that may refuse a feasible constraint beats a
    simplification that silently drops a domain restriction.
    """
    if isinstance(node, Constant):
        return node.value, node.value
    if isinstance(node, Reference):
        return boxes[key(node.ref)]
    if isinstance(node, Negate):
        lo, hi = _interval(node.operand, boxes, key)
        return -hi, -lo
    if isinstance(node, Power):
        lo, hi = _interval(node.base, boxes, key)
        return _power_interval(lo, hi, node.exponent)

    left = _interval(node.left, boxes, key)
    right = _interval(node.right, boxes, key)
    if node.op == '+':
        return left[0] + right[0], left[1] + right[1]
    if node.op == '-':
        return left[0] - right[1], left[1] - right[0]
    if node.op == '*':
        products = (
            left[0] * right[0],
            left[0] * right[1],
            left[1] * right[0],
            left[1] * right[1],
        )
        return min(products), max(products)

    if right[0] <= 0.0 <= right[1]:
        raise ExpressionError(
            'the divisor can reach zero over the allowed parameter ranges, so the '
            'constraint has no finite bound. Tighten the ranges of the parameters in '
            'the denominator so they exclude zero.'
        )
    quotients = (
        left[0] / right[0],
        left[0] / right[1],
        left[1] / right[0],
        left[1] / right[1],
    )
    return min(quotients), max(quotients)


def _power_interval(lo: float, hi: float, exponent: int) -> tuple[float, float]:
    """Range of ``x**exponent`` for ``x`` in ``[lo, hi]``."""
    if exponent == 0:
        if lo <= 0.0 <= hi:
            raise ExpressionError('a zero base raised to the power 0 is undefined.')
        return 1.0, 1.0
    if exponent < 0:
        if lo <= 0.0 <= hi:
            raise ExpressionError(
                'the base of a negative power can reach zero over the allowed ranges. '
                'Tighten that parameter so its range excludes zero.'
            )
        candidates = (lo**exponent, hi**exponent)
        return min(candidates), max(candidates)
    candidates = [lo**exponent, hi**exponent]
    if exponent % 2 == 0 and lo <= 0.0 <= hi:
        # An even power turns around at zero, which is inside the range.
        candidates.append(0.0)
    return min(candidates), max(candidates)


def _affine(node: Node, key: Callable[[ParameterRef], Hashable]) -> AffineForm | None:
    """Normalize to ``constant + Σ coefficient·key``, or None when nonlinear."""
    if isinstance(node, Constant):
        return AffineForm(node.value, {})
    if isinstance(node, Reference):
        return AffineForm(0.0, {key(node.ref): 1.0})
    if isinstance(node, Negate):
        inner = _affine(node.operand, key)
        return None if inner is None else _scaled(inner, -1.0)

    if isinstance(node, Power):
        base = _affine(node.base, key)
        if base is None:
            return None
        if node.exponent == 1:
            return base
        if not base.coefficients:
            try:
                return AffineForm(float(base.constant**node.exponent), {})
            except ZeroDivisionError:
                return None
        return None

    left = _affine(node.left, key)
    right = _affine(node.right, key)
    if left is None or right is None:
        return None

    if node.op in ('+', '-'):
        sign = 1.0 if node.op == '+' else -1.0
        combined = dict(left.coefficients)
        for name, coefficient in right.coefficients.items():
            combined[name] = combined.get(name, 0.0) + sign * coefficient
        return _pruned(AffineForm(left.constant + sign * right.constant, combined))

    if node.op == '*':
        if not right.coefficients:
            return _scaled(left, right.constant)
        if not left.coefficients:
            return _scaled(right, left.constant)
        return None  # a genuine product of two varying quantities

    if right.coefficients or right.constant == 0.0:
        # Division by something that varies is not affine, and division by an
        # exact zero is a domain error the interval path reports properly.
        return None
    return _scaled(left, 1.0 / right.constant)


def _scaled(form: AffineForm, factor: float) -> AffineForm:
    return _pruned(
        AffineForm(
            form.constant * factor,
            {name: coefficient * factor for name, coefficient in form.coefficients.items()},
        )
    )


def _pruned(form: AffineForm) -> AffineForm:
    """Drop coefficients that cancelled exactly.

    This is where ``h2o.radius - d2o.radius`` becomes the constant zero once the
    two refs share a class: both contribute to the same key and the sum is 0.
    """
    return AffineForm(
        form.constant,
        {name: value for name, value in form.coefficients.items() if value != 0.0},
    )


# =========================================================================
# Graph inputs
# =========================================================================


@dataclass(frozen=True, slots=True)
class ParameterDescriptor:
    """Everything the graph needs to know about one parameter of one dataset."""

    ref: ParameterRef
    value: float
    vary: bool
    minimum: float
    maximum: float
    #: Distribution type for a polydispersity width; '' for ordinary parameters.
    pd_type: str = ''
    #: Whether the owning entry has polydispersity switched on. A PD-width
    #: relationship on an entry with PD disabled is rejected rather than
    #: silently switching it on, which would change what the fit evaluates.
    pd_enabled: bool = True
    units: str = ''


@dataclass(frozen=True, slots=True)
class ShareGroup:
    """A symmetric equality group created by ``share()``."""

    members: tuple[ParameterRef, ...]
    #: Member whose value/vary flag wins when members disagree, or None to
    #: require agreement.
    source: ParameterRef | None = None
    #: The unqualified name the user shared, kept for messages and ``describe()``.
    label: str = ''


@dataclass(frozen=True, slots=True)
class ConstraintSpec:
    """One explicit definition of a parameter's value.

    ``kind`` is ``'constant'`` (release A), ``'equality'`` (release A, a bare
    reference) or ``'expression'`` (release B). Storing the original *text*
    alongside the parsed form keeps error messages and ``describe()`` in the
    user's own spelling.
    """

    target: ParameterRef
    kind: str
    text: str
    value: float | None = None
    source: ParameterRef | None = None
    expression: Expression | None = None
    #: Independent configuration saved so ``unconstrain()`` can restore it.
    previous: dict[str, Any] = field(default_factory=dict)

    @property
    def dependencies(self) -> frozenset[ParameterRef]:
        if self.kind == 'equality' and self.source is not None:
            return frozenset({self.source})
        if self.expression is not None:
            return self.expression.refs
        return frozenset()


# =========================================================================
# Compiled graph
# =========================================================================


@dataclass(slots=True)
class ParameterClass:
    """One quantity: a set of refs that are equal, and how its value arises."""

    index: int
    members: tuple[ParameterRef, ...]
    representative: ParameterRef
    #: 'free' | 'fixed' | 'constant' | 'derived'
    kind: str
    value: float
    minimum: float
    maximum: float
    expression: Expression | None = None
    constraint_text: str = ''
    #: Index into the free-root vector, or -1 for everything else.
    free_index: int = -1
    #: Whether the class holds polydispersity widths.
    pd: bool = False

    @property
    def label(self) -> str:
        """The covariance label and public name of this quantity."""
        return self.representative.qualified

    @property
    def shared(self) -> bool:
        return len(self.members) > 1


@dataclass(slots=True)
class CompiledGraph:
    """The resolved graph: classes, free roots, and how to evaluate both."""

    classes: list[ParameterClass]
    class_of: dict[ParameterRef, int]
    #: Class indices of the free roots, in optimizer-vector order.
    free_roots: list[int]
    #: Class indices of derived classes, already topologically sorted.
    derived_order: list[int]
    #: Ranges each class can reach, after target bounds were propagated back.
    ranges: dict[int, tuple[float, float]]

    @property
    def n_free(self) -> int:
        return len(self.free_roots)

    @property
    def labels(self) -> list[str]:
        """Free-root labels, in the order the optimizer vector uses."""
        return [self.classes[index].label for index in self.free_roots]

    def start_vector(self) -> np.ndarray:
        """Configured starting values of the free roots."""
        return np.array([self.classes[i].value for i in self.free_roots], dtype=float)

    def class_values(self, root_values: Sequence[float] | None = None) -> list[float]:
        """Value of every class, computing derived ones in dependency order."""
        values = [entry.value for entry in self.classes]
        if root_values is not None:
            for position, index in enumerate(self.free_roots):
                values[index] = float(root_values[position])
        for index in self.derived_order:
            entry = self.classes[index]
            assert entry.expression is not None  # kind == 'derived' guarantees it
            values[index] = entry.expression.evaluate(
                {ref: values[self.class_of[ref]] for ref in entry.expression.refs}
            )
        return values

    def resolve(self, root_values: Sequence[float] | None = None) -> dict[ParameterRef, float]:
        """Every reference's value, including derived and shared members."""
        values = self.class_values(root_values)
        return {ref: values[index] for ref, index in self.class_of.items()}

    def gradients(self, root_values: Sequence[float] | None = None) -> dict[int, np.ndarray]:
        """∂(class value)/∂(free roots), one vector per class.

        Built in the same topological order as the values, so a derived class
        that depends on another derived class chains correctly. Constants and
        fixed roots get a zero vector, which is the truthful statement that they
        do not move with the fit — as distinct from an *unknown* uncertainty,
        which the result layer reports separately.
        """
        values = self.class_values(root_values)
        jacobians = {index: np.zeros(self.n_free) for index in range(len(self.classes))}
        for position, index in enumerate(self.free_roots):
            jacobians[index][position] = 1.0
        for index in self.derived_order:
            entry = self.classes[index]
            assert entry.expression is not None
            local = {ref: values[self.class_of[ref]] for ref in entry.expression.refs}
            total = np.zeros(self.n_free)
            for ref, partial in entry.expression.gradient(local).items():
                total += partial * jacobians[self.class_of[ref]]
            jacobians[index] = total
        return jacobians


# =========================================================================
# Compilation
# =========================================================================


class _DisjointSet:
    """Union-find over references, the equality half of the graph."""

    def __init__(self, items: Iterable[ParameterRef]) -> None:
        self._parent: dict[ParameterRef, ParameterRef] = {item: item for item in items}

    def find(self, item: ParameterRef) -> ParameterRef:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:  # path compression
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: ParameterRef, right: ParameterRef) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            # Order by qualified name so the structure does not depend on the
            # order calls arrived in, which is what makes labels stable.
            if right_root.qualified < left_root.qualified:
                left_root, right_root = right_root, left_root
            self._parent[right_root] = left_root


def compile_graph(
    descriptors: Mapping[ParameterRef, ParameterDescriptor],
    *,
    local_links: Mapping[ParameterRef, ParameterRef],
    directed_links: Mapping[ParameterRef, ParameterRef],
    share_groups: Sequence[ShareGroup],
    constraints: Mapping[ParameterRef, ConstraintSpec],
) -> CompiledGraph:
    """Resolve declarations into a graph, or refuse with a specific reason.

    Nothing is evaluated by an optimizer here. Every error this raises is one a
    user can act on before a fit starts, which is the point: a graph mistake
    that survives to the optimizer shows up as a puzzling parameter count or a
    constraint that quietly stopped holding.

    Args:
        descriptors: Every referencable parameter, keyed by ref.
        local_links: Per-entry equality links a fitter already carried
            (follower -> target). Mandatory edges, not suggestions.
        directed_links: Cross-dataset ``link_params`` (follower -> target).
        share_groups: Symmetric groups from ``share()``.
        constraints: Explicit constant/equality/expression definitions.

    Raises:
        ConstraintError: On an empty bound intersection, disagreeing share
            members, two definitions of one quantity, a dependency cycle, an
            infeasible starting point, or a constraint whose value cannot be
            kept inside its target's bounds.
    """
    union = _DisjointSet(descriptors)

    # Two different kinds of "follower", because they forfeit different things.
    #
    # `value_followers` give up their value and vary flag: their configuration is
    # a shadow of what they follow, so it cannot vote on the class.
    #
    # `bound_followers` give up their limits as well. Only the *directed* forms
    # do — `link_params()` documents that the follower adopts the target's
    # configuration wholesale, which is what the single-dataset API has always
    # meant. `constrain(target, 'other.parameter')` is not one of them: it
    # documents the opposite guarantee, that the target's limits are enforced.
    # Treating the two alike would let `constrain(x, 'y')` and
    # `constrain(x, '1 * y')` — the same relationship — permit different values.
    value_followers: set[ParameterRef] = set()
    bound_followers: set[ParameterRef] = set()
    for edges in (local_links, directed_links):
        for follower, target in edges.items():
            union.union(follower, target)
            value_followers.add(follower)
            bound_followers.add(follower)
    for spec in constraints.values():
        if spec.kind == 'equality' and spec.source is not None:
            union.union(spec.target, spec.source)
            value_followers.add(spec.target)
    for group in share_groups:
        for member in group.members[1:]:
            union.union(group.members[0], member)

    _reject_unit_mismatch(share_groups, descriptors)

    members_by_root: dict[ParameterRef, list[ParameterRef]] = {}
    for ref in descriptors:
        members_by_root.setdefault(union.find(ref), []).append(ref)

    sources = {
        union.find(group.source): group.source for group in share_groups if group.source is not None
    }

    classes: list[ParameterClass] = []
    class_of: dict[ParameterRef, int] = {}
    for root in sorted(members_by_root, key=lambda ref: ref.qualified):
        members = tuple(sorted(members_by_root[root], key=lambda ref: ref.qualified))
        entry = _build_class(
            len(classes),
            members,
            value_followers,
            bound_followers,
            sources.get(root),
            descriptors,
        )
        classes.append(entry)
        for member in members:
            class_of[member] = entry.index

    _apply_constraints(classes, class_of, constraints)
    derived_order = _topological_order(classes, class_of)

    free_roots: list[int] = []
    for entry in classes:
        if entry.kind == 'free':
            entry.free_index = len(free_roots)
            free_roots.append(entry.index)

    graph = CompiledGraph(
        classes=classes,
        class_of=class_of,
        free_roots=free_roots,
        derived_order=derived_order,
        ranges={},
    )
    graph.ranges = _validate_and_tighten(graph)
    return graph


def _build_class(
    index: int,
    members: tuple[ParameterRef, ...],
    value_followers: set[ParameterRef],
    bound_followers: set[ParameterRef],
    source: ParameterRef | None,
    descriptors: Mapping[ParameterRef, ParameterDescriptor],
) -> ParameterClass:
    """Pick a class's representative, value, vary flag and bounds."""
    # A follower's own value and vary flag are shadows of its target's, so they
    # cannot vote on the class configuration. Sharing creates no followers,
    # which is exactly why a share group has to agree instead.
    authoritative = [ref for ref in members if ref not in value_followers] or list(members)
    # Limits are kept by everything except a directed link's follower: a
    # constrained target forfeits its *value*, not the range it is allowed to
    # take. See the two-follower-set comment in compile_graph.
    bounded = [ref for ref in members if ref not in bound_followers] or list(members)

    if source is not None and source in members:
        representative = source
    else:
        representative = min(authoritative, key=lambda ref: ref.qualified)

    _reject_incompatible_pd(members, descriptors)

    if source is None and len(authoritative) > 1:
        _require_agreement(authoritative, descriptors)

    chosen = descriptors[representative]
    minimum, maximum = chosen.minimum, chosen.maximum
    if len(bounded) > 1:
        minimum = max(descriptors[ref].minimum for ref in bounded)
        maximum = min(descriptors[ref].maximum for ref in bounded)
        if minimum > maximum:
            details = ', '.join(
                f'{ref.qualified} [{descriptors[ref].minimum:g}, {descriptors[ref].maximum:g}]'
                for ref in bounded
            )
            raise ConstraintError(
                f'{chosen.ref.qualified} and the parameters related to it have no allowed '
                f'range in common ({details}). Widen one of them with set_param(), or use '
                'link_params() if the follower is meant to adopt its target’s limits.'
            )

    return ParameterClass(
        index=index,
        members=members,
        representative=representative,
        kind='free' if chosen.vary else 'fixed',
        value=float(chosen.value),
        minimum=float(minimum),
        maximum=float(maximum),
        pd=representative.pd,
    )


def _reject_unit_mismatch(
    share_groups: Sequence[ShareGroup],
    descriptors: Mapping[ParameterRef, ParameterDescriptor],
) -> None:
    """Refuse to share parameters the kernel declares in different units.

    Applied to ``share()`` only, which asserts that the members are one
    measurement — a claim that cannot hold if the models disagree about what is
    being measured. Directed links are left alone: they relate two parameters on
    purpose, often ones with different names and meanings, and forbidding that
    would remove the reason they exist.

    Nothing is converted. A mismatch is reported so the caller can decide, since
    this release has no unit algebra and guessing a conversion factor would be
    worse than refusing.
    """
    for group in share_groups:
        units = {
            descriptors[ref].units: ref
            for ref in group.members
            if ref in descriptors and descriptors[ref].units
        }
        if len(units) > 1:
            listed = ', '.join(f'{ref.qualified} in {unit}' for unit, ref in sorted(units.items()))
            raise ConstraintError(
                f"Cannot share '{group.label or group.members[0].name}': the models declare "
                f'its members in different units ({listed}). Sharing asserts they are one '
                'measurement, and no conversion is applied. Relate them with '
                'link_params() or constrain() if the relationship is deliberate.'
            )


def _reject_incompatible_pd(
    members: tuple[ParameterRef, ...], descriptors: Mapping[ParameterRef, ParameterDescriptor]
) -> None:
    """Refuse a relationship between widths that do not mean the same thing.

    A polydispersity width is only a number once the distribution is fixed: 0.15
    of a lognormal and 0.15 of a Schulz describe different size distributions, so
    tying them together would share a symbol rather than a quantity. Quadrature
    settings are configuration and stay per entry.
    """
    if len(members) < 2:
        return
    pd_members = [ref for ref in members if ref.pd]
    if not pd_members:
        return
    if len(pd_members) != len(members):
        ordinary = next(ref for ref in members if not ref.pd)
        raise ConstraintError(
            f'Cannot relate the polydispersity width {pd_members[0].qualified} to the '
            f'ordinary parameter {ordinary.qualified}: a relative width and a parameter '
            'value are different quantities.'
        )
    disabled = [ref for ref in pd_members if not descriptors[ref].pd_enabled]
    if disabled:
        raise ConstraintError(
            f'Polydispersity is disabled for {disabled[0].dataset}, so its width '
            f'{disabled[0].qualified} takes no part in the fit. Call '
            f"fit['{disabled[0].dataset}'].enable_polydispersity(True) first."
        )
    types = {descriptors[ref].pd_type for ref in pd_members}
    if len(types) > 1:
        listed = ', '.join(
            f'{ref.qualified}={descriptors[ref].pd_type}' for ref in sorted(pd_members)
        )
        raise ConstraintError(
            f'Cannot relate polydispersity widths with different distributions ({listed}). '
            'A width means something different under each distribution. Set one '
            'distribution type with set_pd_param(pd_type=...) on every entry first.'
        )


def _require_agreement(
    members: Sequence[ParameterRef], descriptors: Mapping[ParameterRef, ParameterDescriptor]
) -> None:
    """Refuse to silently pick one member's starting value or vary flag."""
    first = descriptors[members[0]]
    for ref in members[1:]:
        other = descriptors[ref]
        if other.vary != first.vary:
            raise ConstraintError(
                f'{members[0].qualified} and {ref.qualified} are being shared but '
                f'disagree about whether to vary ({first.vary} vs {other.vary}). '
                "Set them the same, or name the one that wins with source='...'."
            )
        if not _values_agree(first.value, other.value):
            raise ConstraintError(
                f'{members[0].qualified} and {ref.qualified} are being shared but start '
                f'from different values ({first.value:g} vs {other.value:g}). Set them '
                "the same, or name the one that wins with source='...'."
            )


def _values_agree(left: float, right: float) -> bool:
    return (
        bool(math.isclose(left, right, rel_tol=VALUE_AGREEMENT_RTOL, abs_tol=0.0)) or left == right
    )


def _apply_constraints(
    classes: list[ParameterClass],
    class_of: Mapping[ParameterRef, int],
    constraints: Mapping[ParameterRef, ConstraintSpec],
) -> None:
    """Turn constant and expression definitions into class kinds."""
    defined: dict[int, ParameterRef] = {}
    for target, spec in constraints.items():
        if spec.kind == 'equality':
            continue  # already merged into an equivalence class
        index = class_of[target]
        if index in defined:
            raise ConstraintError(
                f'{target.qualified} and {defined[index].qualified} are the same quantity '
                'and cannot both be given a definition. Remove one with unconstrain(), or '
                'detach the member with unshare() before constraining it separately.'
            )
        defined[index] = target
        entry = classes[index]
        entry.constraint_text = spec.text
        if spec.kind == 'constant':
            entry.kind = 'constant'
            entry.value = float(spec.value if spec.value is not None else entry.value)
        else:
            entry.kind = 'derived'
            entry.expression = spec.expression


def _topological_order(
    classes: Sequence[ParameterClass], class_of: Mapping[ParameterRef, int]
) -> list[int]:
    """Order derived classes so each is computed after everything it reads."""
    dependencies: dict[int, set[int]] = {}
    for entry in classes:
        if entry.kind != 'derived' or entry.expression is None:
            continue
        dependencies[entry.index] = {class_of[ref] for ref in entry.expression.refs}

    order: list[int] = []
    state: dict[int, int] = {}  # 0 = visiting, 1 = done

    def visit(index: int, path: list[int]) -> None:
        if state.get(index) == 1:
            return
        if state.get(index) == 0:
            cycle = path[path.index(index) :] + [index]
            names = ' -> '.join(classes[i].label for i in cycle)
            raise ConstraintError(
                f'Constraints form a cycle: {names}. A parameter cannot be defined, '
                'directly or indirectly, in terms of itself.'
            )
        state[index] = 0
        for dependency in sorted(dependencies.get(index, ())):
            if dependency in dependencies:
                visit(dependency, path + [index])
            elif dependency == index:
                raise ConstraintError(f'{classes[index].label} is defined in terms of itself.')
        state[index] = 1
        order.append(index)

    for index in sorted(dependencies):
        visit(index, [])
    return order


def _validate_and_tighten(graph: CompiledGraph) -> dict[int, tuple[float, float]]:
    """Check feasibility and push target bounds back onto the roots that drive them.

    Binding an expression into bumps replaces the target's own parameter object,
    so bumps stops enforcing the target's limits — see the plan's section 6.4 and
    the probe recorded there. The limits are kept instead by proving, before the
    fit starts, that the expression cannot leave them over the ranges the roots
    are allowed to explore. Where the relationship is affine in a single root,
    that proof is constructive: invert it and narrow the root's range until the
    target is guaranteed. Anything the arithmetic cannot certify is refused with
    the parameters to tighten named, rather than fitted and hoped over.
    """
    ranges: dict[int, tuple[float, float]] = {}
    for entry in graph.classes:
        if entry.kind == 'free':
            _check_free_root(entry)
            ranges[entry.index] = (entry.minimum, entry.maximum)
        elif entry.kind in ('fixed', 'constant'):
            _check_point(entry)
            ranges[entry.index] = (entry.value, entry.value)

    for _ in range(MAX_BOUND_TIGHTENING_PASSES):
        if not _tightening_pass(graph, ranges):
            break
    else:
        raise ConstraintError(
            'Constraint bounds could not be resolved to a stable set of parameter '
            'ranges. Simplify the chain of constraints or tighten the ranges by hand.'
        )

    # Starting values last: a tightened range can exclude a start that was inside
    # the configured one, and the optimizer must not be launched from outside its
    # own box.
    for entry in graph.classes:
        if entry.kind != 'free':
            continue
        low, high = ranges[entry.index]
        if not low <= entry.value <= high:
            raise ConstraintError(
                f'{entry.label} starts at {entry.value:g}, outside the range '
                f'[{low:g}, {high:g}] that the constraints leave it. Move the starting '
                'value inside that range, or relax the constraint that narrowed it.'
            )
    return ranges


def _check_free_root(entry: ParameterClass) -> None:
    if not (math.isfinite(entry.minimum) and math.isfinite(entry.maximum)):
        raise ConstraintError(
            f'{entry.label} varies but has a non-finite bound '
            f'[{entry.minimum}, {entry.maximum}]. Give it finite limits with set_param().'
        )
    if entry.minimum == entry.maximum:
        raise ConstraintError(
            f'{entry.label} varies but its allowed range collapsed to the single value '
            f'{entry.minimum:g}. Fix it explicitly with vary=False instead.'
        )
    if entry.minimum > entry.maximum:
        raise ConstraintError(
            f'{entry.label} has an empty range [{entry.minimum:g}, {entry.maximum:g}].'
        )
    if not math.isfinite(entry.value):
        raise ConstraintError(f'{entry.label} has a non-finite starting value.')
    if not entry.minimum <= entry.value <= entry.maximum:
        raise ConstraintError(
            f'{entry.label} starts at {entry.value:g}, outside its range '
            f'[{entry.minimum:g}, {entry.maximum:g}].'
        )


def _check_point(entry: ParameterClass) -> None:
    if not math.isfinite(entry.value):
        raise ConstraintError(f'{entry.label} has a non-finite value ({entry.value}).')
    if entry.kind == 'constant' and not entry.minimum <= entry.value <= entry.maximum:
        raise ConstraintError(
            f"Constraint '{entry.label} = {entry.constraint_text}' sets {entry.value:g}, "
            f'outside the allowed range [{entry.minimum:g}, {entry.maximum:g}]. Widen the '
            'bounds with set_param(min=..., max=...) if the value is intended.'
        )


def _tightening_pass(graph: CompiledGraph, ranges: dict[int, tuple[float, float]]) -> bool:
    """One sweep over the derived classes. Returns True if anything narrowed."""
    changed = False
    for index in graph.derived_order:
        entry = graph.classes[index]
        assert entry.expression is not None
        affine = entry.expression.affine(lambda ref: graph.class_of[ref])

        if affine is not None:
            low, high = affine.range_over(ranges)
        else:
            try:
                low, high = entry.expression.interval(ranges, lambda ref: graph.class_of[ref])
            except ExpressionError as error:
                raise ConstraintError(
                    f"Constraint '{entry.label} = {entry.constraint_text}' cannot be "
                    f'bounded: {error}'
                ) from error

        if not (math.isfinite(low) and math.isfinite(high)):
            raise ConstraintError(
                f"Constraint '{entry.label} = {entry.constraint_text}' can reach a "
                'non-finite value over the allowed parameter ranges.'
            )

        if low < entry.minimum or high > entry.maximum:
            if affine is not None and len(affine.coefficients) == 1:
                root_index, coefficient = next(iter(affine.coefficients.items()))
                if _narrow_root(
                    graph, ranges, int(root_index), affine.constant, coefficient, entry
                ):
                    changed = True
                    continue
            raise ConstraintError(
                f"Constraint '{entry.label} = {entry.constraint_text}' can reach "
                f"[{low:g}, {high:g}], outside {entry.label}'s allowed range "
                f'[{entry.minimum:g}, {entry.maximum:g}]. Tighten the range of '
                f'{_dependency_names(graph, entry)} so the constraint stays inside it, '
                'or widen the target bounds with set_param().'
            )

        ranges[index] = (max(low, entry.minimum), min(high, entry.maximum))
    return changed


def _narrow_root(
    graph: CompiledGraph,
    ranges: dict[int, tuple[float, float]],
    root_index: int,
    constant: float,
    coefficient: float,
    target: ParameterClass,
) -> bool:
    """Invert a one-root affine constraint onto that root's range.

    ``target = constant + coefficient·root`` is monotonic in the root, so the
    target's limits map exactly onto root limits. This is the constructive half
    of bound enforcement: the narrowed range is what the root's bumps parameter
    is given, which is why the target's limits survive an expression binding
    that discards them.
    """
    root = graph.classes[root_index]
    if root.kind != 'free' or coefficient == 0.0:
        return False

    implied = sorted(
        (
            (target.minimum - constant) / coefficient,
            (target.maximum - constant) / coefficient,
        )
    )
    low, high = ranges[root_index]
    new_low = max(low, implied[0])
    new_high = min(high, implied[1])
    if new_low > new_high:
        raise ConstraintError(
            f"Constraint '{target.label} = {target.constraint_text}' cannot be satisfied: "
            f'keeping {target.label} inside [{target.minimum:g}, {target.maximum:g}] would '
            f'require {root.label} in [{implied[0]:g}, {implied[1]:g}], which does not '
            f'overlap its allowed range [{low:g}, {high:g}].'
        )
    if (new_low, new_high) == (low, high):
        return False
    ranges[root_index] = (new_low, new_high)
    return True


def _dependency_names(graph: CompiledGraph, entry: ParameterClass) -> str:
    assert entry.expression is not None
    labels = sorted({graph.classes[graph.class_of[ref]].label for ref in entry.expression.refs})
    return ', '.join(labels) or 'its inputs'

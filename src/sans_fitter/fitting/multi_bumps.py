"""Binding a compiled parameter graph onto one bumps ``FitProblem``.

The numerical half of a simultaneous fit, and the only module that knows both
the graph from :mod:`sans_fitter.modeling.constraints` and bumps. Three things
happen here that have no single-dataset equivalent:

**One parameter object per class, not per model.** bumps discovers fit
parameters by walking the object graph and deduplicating *by identity*
(``bumps.parameter.unique``), so two datasets sharing a radius is literally the
same ``Parameter`` instance reached from two models. That is why shared
parameters cost one coordinate rather than two plus a constraint. Every class
gets exactly one object here, named with its qualified label, and every member
reference is bound to it.

**Labels are assigned, not inherited.** bumps names a parameter after the kernel
slot it came from, so two datasets fitting a sphere each produce a parameter
called ``radius``. Nothing complains: ``problem.labels()`` simply returns
``['radius', 'radius']``, and any code keying a dictionary by label silently
loses one. Each root is renamed to its ``dataset.parameter`` label before the
problem is built, and the labels are checked for uniqueness afterwards.

**Covariance depends on what the weights mean.** With unit weights this reuses
the same unscaled Jacobian covariance the single-dataset engine reports. With
explicit dataset factors it computes the sandwich form from section 7 of the
plan, because those factors are fitting priorities and the supplied ``dI``
remains the error model — the inverse curvature of the reweighted objective
would answer a different question.
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from bumps.fitters import fit as bumps_fit
from bumps.names import FitProblem
from bumps.parameter import Parameter
from sasmodels.bumps_model import Experiment

from ..console import CHI_SQUARED, logger
from ..modeling.constraints import CompiledGraph, Constant, Negate, Power, Reference
from ..results import ParameterStateSnapshot
from .base import extract_fit_index, validate_covariance
from .bumps_engine import _configured_budget, build_bumps_model

__all__ = ['MultiEntrySpec', 'MultiFitOutcome', 'DatasetOutcome', 'run_multi_bumps']


@dataclass(slots=True)
class MultiEntrySpec:
    """Everything one dataset contributes to the joint problem."""

    name: str
    data: Any
    kernel: Any
    fit_state: ParameterStateSnapshot
    weight: float
    #: Polydispersity base names whose block must be written even when this
    #: entry's own width looks inactive — a shared or derived width is driven
    #: from the graph, not from the local configuration.
    forced_pd: tuple[str, ...] = ()
    #: Entry-local parameter name -> the canonical sasmodels slots it drives.
    #: One name can drive several slots under ``set_models(shared=...)``.
    canonical: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: Entry-local parameter name -> canonical base name of its PD width.
    canonical_pd: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetOutcome:
    """Per-dataset numbers extracted at the best point."""

    name: str
    n_points: int
    chisq: float
    weight: float
    fitted_curve: np.ndarray
    fit_index: np.ndarray | None
    residuals: np.ndarray

    @property
    def objective_contribution(self) -> float:
        return self.weight * self.chisq

    @property
    def rms_residual(self) -> float:
        return math.sqrt(self.chisq / self.n_points) if self.n_points else float('nan')


@dataclass(slots=True)
class MultiFitOutcome:
    """Everything the joint fit produced, before it is shaped into a result."""

    method: str
    message: str
    root_values: np.ndarray
    datasets: list[DatasetOutcome]
    chisq: float
    objective: float
    n_points: int
    cov: np.ndarray | None
    cov_labels: list[str]
    cov_source: str | None
    cov_note: str = ''
    problem: Any = None


# =========================================================================
# Construction
# =========================================================================


def _build_root_parameters(graph: CompiledGraph) -> dict[int, Any]:
    """One bumps object per class, in dependency order.

    Free roots get a range; fixed roots and constants stay plain values, which
    bumps leaves out of the fit vector. Derived classes become native bumps
    expressions so the relationship is re-evaluated on every trial point by
    bumps itself rather than by a hook this package would have to install.
    """
    objects: dict[int, Any] = {}

    for entry in graph.classes:
        if entry.kind == 'derived':
            continue
        parameter = Parameter(value=entry.value, name=entry.label)
        if entry.kind == 'free':
            low, high = graph.ranges[entry.index]
            parameter.range(low, high)
        objects[entry.index] = parameter

    for index in graph.derived_order:
        entry = graph.classes[index]
        assert entry.expression is not None
        objects[index] = _to_bumps_expression(entry.expression.root, graph, objects)

    return objects


def _to_bumps_expression(node: Any, graph: CompiledGraph, objects: dict[int, Any]) -> Any:
    """Translate a parsed constraint node into bumps arithmetic objects.

    The tree is walked, never executed: each node maps onto an operator that
    bumps' ``Parameter`` overloads into a live ``Expression``. Constant
    sub-trees collapse to plain floats on the way, which keeps a constraint like
    ``2 * cold.radius + 10`` down to one expression object.
    """
    if isinstance(node, Constant):
        return node.value
    if isinstance(node, Reference):
        return objects[graph.class_of[node.ref]]
    if isinstance(node, Negate):
        return -_to_bumps_expression(node.operand, graph, objects)
    if isinstance(node, Power):
        return _to_bumps_expression(node.base, graph, objects) ** node.exponent

    left = _to_bumps_expression(node.left, graph, objects)
    right = _to_bumps_expression(node.right, graph, objects)
    if node.op == '+':
        return left + right
    if node.op == '-':
        return left - right
    if node.op == '*':
        return left * right
    return left / right


def _bind(
    entries: list[MultiEntrySpec],
    models: dict[str, Any],
    graph: CompiledGraph,
    objects: dict[int, Any],
) -> None:
    """Point every member reference at its class's single object.

    Assignment replaces the model's own parameter object, which is how bumps'
    identity-based discovery collapses the members into one coordinate. It also
    discards the replaced object's bounds — that is expected, and why the graph
    proved the bounds before this point rather than relying on bumps to enforce
    them (plan section 6.4).
    """
    specs = {entry.name: entry for entry in entries}
    for entry_class in graph.classes:
        target = objects[entry_class.index]
        for ref in entry_class.members:
            spec = specs[ref.dataset]
            model = models[ref.dataset]
            if ref.pd:
                slots: tuple[str, ...] = (f'{spec.canonical_pd[ref.name]}_pd',)
            else:
                slots = spec.canonical[ref.name]
            for slot in slots:
                setattr(model, slot, target)


def build_multi_problem(
    entries: list[MultiEntrySpec], graph: CompiledGraph
) -> tuple[Any, dict[str, Any], dict[int, Any]]:
    """Assemble the joint problem. Returns (problem, experiments, root objects)."""
    models = {
        entry.name: build_bumps_model(entry.kernel, entry.fit_state, entry.forced_pd)
        for entry in entries
    }
    objects = _build_root_parameters(graph)
    _bind(entries, models, graph, objects)

    experiments = {
        entry.name: Experiment(data=entry.data, model=models[entry.name], name=entry.name)
        for entry in entries
    }
    # sqrt(a): bumps multiplies residuals by the weight, so the objective picks
    # up its square. Passing the public factor straight through would minimise
    # a different function than the one documented.
    problem = FitProblem(
        [experiments[entry.name] for entry in entries],
        weights=[math.sqrt(entry.weight) for entry in entries],
    )

    _verify_discovery(problem, graph)
    return problem, experiments, objects


def _verify_discovery(problem: Any, graph: CompiledGraph) -> None:
    """Check that bumps found exactly the coordinates the graph predicted.

    A mismatch means a binding did not take effect — a member still holding its
    own object, or a shared parameter counted twice. Both produce a fit that
    runs and reports a wrong number of degrees of freedom, so this is checked
    rather than assumed.
    """
    labels = list(problem.labels())
    if len(set(labels)) != len(labels):
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        raise RuntimeError(
            f'Internal error: the joint problem has duplicate parameter labels '
            f'{duplicates}. Every independent root must carry its qualified name.'
        )
    if sorted(labels) != sorted(graph.labels):
        missing = sorted(set(graph.labels) - set(labels))
        unexpected = sorted(set(labels) - set(graph.labels))
        raise RuntimeError(
            'Internal error: bumps discovered a different parameter set than the '
            f'constraint graph compiled. Missing: {missing or "none"}. '
            f'Unexpected: {unexpected or "none"}.'
        )


# =========================================================================
# Covariance
# =========================================================================


def _point_weights(outcomes: list[DatasetOutcome]) -> np.ndarray:
    """The dataset factor of every selected point, in concatenation order."""
    return np.concatenate(
        [np.full(entry.n_points, entry.weight, dtype=float) for entry in outcomes]
    )


def _rank_note(curvature: np.ndarray, labels: list[str]) -> str:
    """A diagnosis when the free coordinates are not separately identifiable.

    Computed on both covariance paths, so the diagnosis does not appear and
    disappear with the dataset weights. What each path then *does* with it
    differs, and deliberately: see :func:`_covariance`.
    """
    size = curvature.shape[0]
    rank = int(np.linalg.matrix_rank(curvature))
    if rank >= size:
        return ''
    return (
        f'the curvature matrix is rank deficient ({rank} of {size} independent '
        f'directions over {", ".join(labels)}), so at least one combination of these '
        'parameters is not determined by the data. Fix or relate a redundant '
        'parameter and refit.'
    )


def _covariance(
    problem: Any, outcomes: list[DatasetOutcome], labels: list[str]
) -> tuple[np.ndarray | None, str | None, str]:
    """Covariance over the free roots, and an honest account of where it came from.

    With every factor at 1 this is the same estimate the single-dataset engine
    reports, so a one-entry ``MultiFitter`` and a ``SANSFitter`` agree. bumps'
    ``jacobian_cov`` clamps singular values, so it returns a finite matrix even
    when the problem is degenerate; that matrix is kept — diverging from the
    single-fit engine would be a worse surprise than a large variance — but the
    deficiency is stated in ``cov_note`` and warned about, rather than left for
    the reader to infer from an implausible error bar.

    With factors other than 1 the objective is no longer the likelihood of the
    supplied uncertainties, and the inverse curvature ``H⁻¹`` would only be a
    covariance if ``dI/√a`` were the real error bars. The documented meaning is
    the opposite — the factors are priorities and ``dI`` is trusted — so the
    sandwich ``H⁻¹ (Jᵀ W² J) H⁻¹`` is used instead. It is invariant under
    multiplying every factor by one constant, which a bare ``H⁻¹`` is not. There
    is no clamped fallback to inherit here, so a deficient problem yields no
    covariance at all.
    """
    from bumps import lsqerror

    jacobian = np.asarray(lsqerror.jacobian(problem, problem.getp()), dtype=float)
    weights = np.array([entry.weight for entry in outcomes], dtype=float)
    # jacobian() differentiates problem.residuals(), which already carries the
    # sqrt(a) factors, so this is J_w and the curvature needs no reweighting.
    curvature = jacobian.T @ jacobian
    note = _rank_note(curvature, labels)

    if np.allclose(weights, 1.0):
        if note:
            warnings.warn(
                f'Uncertainties may be meaningless: {note}',
                RuntimeWarning,
                stacklevel=4,
            )
        return (
            validate_covariance(lsqerror.jacobian_cov(jacobian), labels),
            'jacobian',
            note,
        )

    if note:
        return None, None, f'{note} No weighted covariance is available.'

    point_weights = _point_weights(outcomes)
    middle = jacobian.T @ (point_weights[:, None] * jacobian)
    try:
        left = np.linalg.solve(curvature, middle)
        cov = np.linalg.solve(curvature, left.T).T
    except np.linalg.LinAlgError as error:
        return None, None, f'the weighted curvature matrix could not be inverted ({error}).'

    return validate_covariance(cov, labels), 'weighted-sandwich-known-dI', ''


# =========================================================================
# Running
# =========================================================================


def _collect(entries: list[MultiEntrySpec], experiments: dict[str, Any]) -> list[DatasetOutcome]:
    """Per-dataset curves, residuals and chi-squared at the current point."""
    outcomes: list[DatasetOutcome] = []
    for entry in entries:
        experiment = experiments[entry.name]
        # Experiment.residuals() is (theory - I)/dI, the sign this package keeps
        # throughout the multi-fit path and states on every export.
        residuals = np.asarray(experiment.residuals(), dtype=float)
        outcomes.append(
            DatasetOutcome(
                name=entry.name,
                n_points=int(residuals.size),
                chisq=float(np.sum(residuals**2)),
                weight=float(entry.weight),
                fitted_curve=np.asarray(experiment.theory(), dtype=float),
                fit_index=extract_fit_index(experiment),
                residuals=residuals,
            )
        )
    return outcomes


def run_multi_bumps(
    entries: list[MultiEntrySpec],
    graph: CompiledGraph,
    method: str = 'amoeba',
    **kwargs: Any,
) -> MultiFitOutcome:
    """Optimize the joint problem and read everything back at the best point."""
    problem, experiments, objects = build_multi_problem(entries, graph)

    logger.info(
        f'\nJoint fit: {len(entries)} datasets, {graph.n_free} free parameters\n'
        f'Initial {CHI_SQUARED}/dof = {problem.chisq():.4f}\n'
        f'Fitting with BUMPS (method: {method})...'
    )

    if graph.n_free == 0:
        # Nothing to optimize, but the constraints still have to be resolved and
        # the curves evaluated, so this is a real (single-point) evaluation
        # rather than an error.
        outcomes = _collect(entries, experiments)
        return _outcome(
            entries,
            graph,
            problem,
            outcomes,
            method,
            'no free parameters; the model was evaluated once at the configured values',
            None,
            [],
            None,
            '',
            objects,
        )

    result = bumps_fit(problem, method=method, **kwargs)

    # Explicitly re-set the best point before reading anything: the problem's
    # state after a fit is wherever the optimizer last evaluated, which is not
    # necessarily the best point, and setp() also invalidates the experiment
    # caches so the curves below are the ones that belong to these parameters.
    problem.setp(np.asarray(result.x, dtype=float))
    outcomes = _collect(entries, experiments)

    labels = list(problem.labels())
    cov, cov_source, cov_note = _covariance(problem, outcomes, labels)

    budget = _configured_budget(method, problem, kwargs)
    steps = getattr(result, 'nit', None)
    message = (
        'bumps does not report convergence; '
        f'iterations reported: {"unknown" if steps is None else steps}, '
        f'configured maximum: {"unknown" if budget is None else budget}'
    )

    return _outcome(
        entries,
        graph,
        problem,
        outcomes,
        method,
        message,
        cov,
        labels,
        cov_source,
        cov_note,
        objects,
    )


def _outcome(
    entries: list[MultiEntrySpec],
    graph: CompiledGraph,
    problem: Any,
    outcomes: list[DatasetOutcome],
    method: str,
    message: str,
    cov: np.ndarray | None,
    labels: list[str],
    cov_source: str | None,
    cov_note: str,
    objects: dict[int, Any],
) -> MultiFitOutcome:
    """Assemble the outcome, reordering covariance into the graph's root order."""
    chisq = float(sum(entry.chisq for entry in outcomes))
    objective = float(sum(entry.objective_contribution for entry in outcomes))

    residual_vector = np.asarray(problem.residuals(), dtype=float)
    reconstructed = float(np.dot(residual_vector, residual_vector))
    if not math.isclose(objective, reconstructed, rel_tol=1e-9, abs_tol=1e-12):
        raise RuntimeError(
            f'Internal error: the weighted objective from per-dataset contributions '
            f'({objective:.12g}) disagrees with the residual vector the optimizer '
            f'minimized ({reconstructed:.12g}).'
        )

    ordered_cov = cov
    if cov is not None and labels:
        permutation = [labels.index(label) for label in graph.labels]
        ordered_cov = np.asarray(cov)[np.ix_(permutation, permutation)]

    # Read the roots back from the objects bumps actually moved, rather than
    # from the result vector, so the graph and the problem cannot disagree about
    # which coordinate is which.
    root_values = np.array([float(objects[index].value) for index in graph.free_roots], dtype=float)

    return MultiFitOutcome(
        method=method,
        message=message,
        root_values=root_values,
        datasets=outcomes,
        chisq=chisq,
        objective=objective,
        n_points=int(sum(entry.n_points for entry in outcomes)),
        cov=ordered_cov,
        cov_labels=list(graph.labels),
        cov_source=cov_source,
        cov_note=cov_note,
        problem=problem,
    )

"""Simultaneous, constrained fitting of several datasets.

``MultiFitter`` owns a set of named datasets, each with its own model, Q range,
resolution and parameters, and fits them together through one bumps
``FitProblem``. Contrast variation, a temperature series, or one sample measured
on three instrument configurations all have the same shape: most of the physics
is common, a few parameters are not, and fitting the curves one at a time throws
away the constraint that makes the common part identifiable.

    >>> from sans_fitter import MultiFitter
    >>> fit = MultiFitter()
    >>> fit.add('h2o', data_h2o, model='sphere')
    >>> fit.add('d2o', data_d2o, model='sphere')
    >>> for name in ('h2o', 'd2o'):
    ...     fit[name].set_param('radius', value=45, min=10, max=100, vary=True)
    ...     fit[name].set_param('scale', value=0.02, min=0.001, max=0.1, vary=True)
    >>> fit.share('radius')
    >>> fit.constrain('h2o.sld_solvent', -0.56)
    >>> fit.constrain('d2o.sld_solvent', 6.34)
    >>> result = fit.fit()

Every parameter is addressed as ``dataset.parameter``. Three relationships can
hold between them — :meth:`share` (one quantity, several datasets),
:meth:`link_params` (one parameter follows another) and :meth:`constrain` (a
constant or an arithmetic expression) — and all three are resolved by
:mod:`sans_fitter.modeling.constraints` into a graph before any model runs, so a
contradiction is reported at the call that creates it rather than as a strange
fit.

Two design choices are worth knowing before reading further:

**Datasets are owned, not borrowed.** :meth:`add` copies the data it is given
and builds a private ``SANSFitter`` behind each handle. Editing the array you
passed in cannot change a configured fit, and there is no child fitter to call
``.fit()`` on by mistake while it is part of a joint analysis.

**Every mutation recompiles the graph.** A call that would leave the analysis
inconsistent — sharing parameters whose bounds do not overlap, constraining a
value outside its limits, changing a model out from under a constraint — is
rejected and rolls back, so the fitter is never in a state that cannot be fitted.
"""

import copy
import csv
import io
import keyword
import math
import os
import re
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from plotly.graph_objects import Figure

from . import multi_plotting
from .console import OK, logger
from .data.loader import get_fit_index, has_real_data
from .data.provenance import fingerprint_arrays
from .data.resolution import ResolutionSetting
from .fileio import atomic_write
from .fitter import SANSFitter
from .fitting.base import at_bound, reduced_chisq
from .fitting.multi_bumps import MultiEntrySpec, run_multi_bumps
from .fitting.theory import build_model_parameters, evaluate_theory, theory_data
from .modeling.constraints import (
    CompiledGraph,
    ConstraintError,
    ConstraintSpec,
    ParameterDescriptor,
    ParameterRef,
    ShareGroup,
    compile_graph,
    parse_expression,
    parse_reference,
)
from .multi_results import (
    DatasetResult,
    MultiFitReport,
    MultiFitResult,
    MultiParameter,
    format_estimate,
)
from .persistence import config_digest

__all__ = ['MultiFitter', 'DatasetHandle']

#: Dataset names are identifiers so they can appear unquoted in a constraint
#: expression, where ``h2o.radius`` has to parse as Python.
_NAME_PATTERN = re.compile(r'[A-Za-z][A-Za-z0-9_]*')

#: Polydispersity widths are fitted on this range, matching the single-dataset
#: engine. Recorded in the graph rather than left implicit so a shared width has
#: bounds like any other quantity.
PD_WIDTH_RANGE = (0.0, 1.0)

#: Name of the export manifest, and the line above its file list. The list is
#: how a later export knows which artifacts were its own to retire.
MANIFEST_NAME = 'manifest.txt'
MANIFEST_FILE_MARKER = 'Files written by this export:'


class _Entry:
    """One owned dataset: its private fitter, weight and provenance."""

    __slots__ = ('name', 'fitter', 'weight', 'handle')

    def __init__(self, name: str, fitter: SANSFitter, weight: float) -> None:
        self.name = name
        self.fitter = fitter
        self.weight = weight
        self.handle: DatasetHandle | None = None

    def snapshot(self) -> tuple[Any, ...]:
        """Enough state to undo a rejected configuration change.

        The parameter manager is deep-copied because it is the only mutable
        thing a handle can reach; the kernel is held by reference, since
        rebuilding it would discard a compiled sasmodels kernel for no reason.
        """
        return (
            self.fitter.kernel,
            copy.deepcopy(self.fitter._param_manager),
            self.fitter.get_q_range(),
            self.fitter.get_resolution(),
            self.weight,
        )

    def restore(self, state: tuple[Any, ...]) -> None:
        kernel, manager, q_range, resolution, weight = state
        self.fitter.kernel = kernel
        self.fitter._param_manager = manager
        if q_range is not None and self.fitter.data is not None:
            self.fitter.data.qmin, self.fitter.data.qmax = q_range
        self.fitter._resolution = ResolutionSetting(**resolution)
        self.weight = weight


class DatasetHandle:
    """Configure one dataset of a :class:`MultiFitter`.

    A deliberate facade over the private fitter, not the fitter itself. The
    methods below are the ones that describe *this* dataset; everything that
    describes the analysis as a whole — fitting, sharing, constraints, plots,
    export — belongs to the parent, because a child that could fit itself while
    enrolled in a joint analysis would produce a result nobody asked for and
    nothing would notice.

    Every mutation is routed back through the parent so the parameter graph is
    revalidated and, if the change is inconsistent, rolled back.
    """

    __slots__ = ('_parent', '_name')

    def __init__(self, parent: 'MultiFitter', name: str) -> None:
        self._parent = parent
        self._name = name

    def __repr__(self) -> str:
        return f'<DatasetHandle {self._name!r} model={self.model_name!r}>'

    @property
    def _fitter(self) -> SANSFitter:
        return self._parent._entry(self._name).fitter

    @property
    def name(self) -> str:
        """The dataset's identifier within the analysis."""
        return self._name

    @property
    def model_name(self) -> str | None:
        """The model expression this dataset is fitted with."""
        return self._fitter.model_name

    @property
    def weight(self) -> float:
        """This dataset's objective coefficient."""
        return self._parent._entry(self._name).weight

    @property
    def data(self) -> Any:
        """A copy of the dataset, so edits cannot reach the configured fit."""
        return copy.deepcopy(self._fitter.data)

    @property
    def params(self) -> dict[str, dict[str, Any]]:
        """A snapshot of the parameter table, with values resolved by the graph."""
        resolved = self._parent._resolved_values()
        table: dict[str, dict[str, Any]] = {}
        for name, info in self._fitter.params.items():
            entry = dict(info)
            ref = ParameterRef(self._name, name)
            if ref in resolved:
                entry['value'] = resolved[ref]
            table[name] = entry
        return table

    # -- model configuration ------------------------------------------------

    def set_model(self, model_name: str, platform: str = 'cpu') -> None:
        """Replace this dataset's model. Rejected while a constraint needs a
        parameter the new model does not have."""
        self._parent._mutate(self._name, lambda: self._fitter.set_model(model_name, platform))

    def set_models(self, *model_names: str, **kwargs: Any) -> None:
        """Combine several models for this dataset (see :meth:`SANSFitter.set_models`)."""
        self._parent._mutate(self._name, lambda: self._fitter.set_models(*model_names, **kwargs))

    def set_structure_factor(
        self, structure_factor_name: str, radius_effective_mode: str = 'unconstrained'
    ) -> None:
        """Apply a structure factor to this dataset's model."""
        self._parent._mutate(
            self._name,
            lambda: self._fitter.set_structure_factor(structure_factor_name, radius_effective_mode),
        )

    def remove_structure_factor(self) -> None:
        """Revert to the form factor alone."""
        self._parent._mutate(self._name, self._fitter.remove_structure_factor)

    # -- parameters ---------------------------------------------------------

    def set_param(
        self,
        name: str,
        value: float | None = None,
        min: float | None = None,
        max: float | None = None,
        vary: bool | None = None,
    ) -> None:
        """Configure a parameter of this dataset.

        Routed through the parent, so setting a value or vary flag on a member
        of a shared group updates the whole group and keeps it consistent.
        """
        self._parent.set_param(f'{self._name}.{name}', value=value, min=min, max=max, vary=vary)

    def set_pd_param(
        self,
        param_name: str,
        pd_width: float | None = None,
        pd_n: int | None = None,
        pd_nsigma: float | None = None,
        pd_type: str | None = None,
        vary: bool | None = None,
    ) -> None:
        """Configure polydispersity for a parameter of this dataset.

        One call is one transaction. The width and vary flag belong to the
        quantity and reach every dataset sharing it; the distribution type,
        quadrature count and truncation are this dataset's own configuration.
        If any part of the call is rejected, none of it is applied.
        """
        self._parent._configure_pd(
            self._name,
            param_name,
            pd_width=pd_width,
            pd_n=pd_n,
            pd_nsigma=pd_nsigma,
            pd_type=pd_type,
            vary=vary,
        )

    def enable_polydispersity(self, enabled: bool = True) -> None:
        """Enable or disable polydispersity for this dataset."""
        self._parent._mutate(self._name, lambda: self._fitter.enable_polydispersity(enabled))

    def link_params(self, name: str, to: str) -> None:
        """Link two parameters *within* this dataset.

        Applied to the dataset's own parameter manager, exactly as
        :meth:`SANSFitter.link_params` would, so this handle's ``get_links()``
        and the graph agree about it afterwards. The graph picks it up as a
        mandatory local edge. The cross-dataset form is
        :meth:`MultiFitter.link_params`, which takes qualified names.
        """
        self._parent._mutate(self._name, lambda: self._fitter.link_params(name, to))

    def unlink_params(self, name: str) -> None:
        """Remove a link created by :meth:`link_params`."""
        self._parent._mutate(self._name, lambda: self._fitter.unlink_params(name))

    # -- selection and resolution -------------------------------------------

    def set_q_range(self, qmin: float | None = None, qmax: float | None = None) -> None:
        """Restrict the Q range used for this dataset."""
        self._parent._mutate(self._name, lambda: self._fitter.set_q_range(qmin, qmax))

    def reset_q_range(self) -> None:
        """Use this dataset's full Q range again."""
        self._parent._mutate(self._name, self._fitter.reset_q_range)

    def set_resolution(self, mode: str = 'data', **kwargs: Any) -> None:
        """Choose how instrument resolution is applied to this dataset."""
        self._parent._mutate(self._name, lambda: self._fitter.set_resolution(mode, **kwargs))

    # -- inspection ---------------------------------------------------------

    def get_params(self) -> None:
        """Print this dataset's parameter table."""
        self._fitter.get_params()

    def get_pd_param(self, param_name: str) -> dict[str, Any]:
        """Polydispersity configuration for one parameter.

        ``pd`` is the width the analysis will actually use, resolved through the
        constraint graph, so a shared or constrained width reads the same here
        as it does in :meth:`MultiFitter.get_parameter_table`. The distribution
        type, quadrature count and truncation are per-dataset configuration and
        are reported as this dataset holds them.
        """
        config = dict(self._fitter.get_pd_param(param_name))
        resolved = self._parent._resolved_value(
            ParameterRef(self._name, self._fitter._param_manager.resolve_name(param_name), pd=True)
        )
        if resolved is not None:
            config['pd'] = resolved
            config['active'] = resolved > 0
        return config

    def get_pd_params(self) -> None:
        """Print this dataset's polydispersity table."""
        self._fitter.get_pd_params()

    def get_polydisperse_parameters(self) -> list[str]:
        """Parameters of this model that support polydispersity."""
        return self._fitter.get_polydisperse_parameters()

    def get_varying_pd_params(self) -> list[str]:
        """Polydispersity widths set to vary for this dataset."""
        return self._fitter.get_varying_pd_params()

    def supports_polydispersity(self) -> bool:
        return self._fitter.supports_polydispersity()

    def is_polydispersity_enabled(self) -> bool:
        return self._fitter.is_polydispersity_enabled()

    def get_q_range(self) -> tuple[float, float] | None:
        return self._fitter.get_q_range()

    def get_resolution(self) -> dict[str, Any]:
        return self._fitter.get_resolution()

    def get_structure_factor(self) -> str | None:
        return self._fitter.get_structure_factor()

    def get_components(self) -> list[tuple[str, str, str]]:
        return self._fitter.get_components()

    def get_links(self) -> dict[str, str]:
        """Every equality relationship whose follower is a parameter of this dataset.

        Covers this dataset's own links *and* the cross-dataset ones declared on
        the parent, because a reader asking what constrains this dataset needs
        both. A target in another dataset is reported qualified
        (``{'radius': 'h2o.radius'}``); a target in this one is reported bare
        (``{'radius_effective': 'radius'}``), matching
        :meth:`SANSFitter.get_links`.
        """
        links = dict(self._fitter.get_links())
        for follower, target in self._parent._cross_dataset_equalities().items():
            if follower.dataset != self._name:
                continue
            local = f'{follower.name}_pd' if follower.pd else follower.name
            links[local] = (
                (f'{target.name}_pd' if target.pd else target.name)
                if target.dataset == self._name
                else target.qualified
            )
        return links


class MultiFitter:
    """Fit several datasets at once, with parameters shared or related between them.

    See the module docstring for the workflow. The public surface divides into
    configuration (:meth:`add`, :meth:`share`, :meth:`link_params`,
    :meth:`constrain`, :meth:`set_dataset_weight`), inspection
    (:meth:`describe`, :meth:`get_constraints`, :meth:`get_sharing`,
    :meth:`calculate`, :meth:`plot_model`) and results (:meth:`fit`,
    :meth:`get_fit_report`, :meth:`plot_results`, :meth:`save_results`).

    Only the bumps engine is supported. The parameter graph is kept free of
    bumps so the scipy backend and joint DREAM sampling can be added without
    changing this API.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._share_groups: list[ShareGroup] = []
        self._directed: dict[ParameterRef, ParameterRef] = {}
        self._constraints: dict[ParameterRef, ConstraintSpec] = {}
        self._result: MultiFitResult | None = None
        self._graph: CompiledGraph | None = None

    # =====================================================================
    # Dataset registry
    # =====================================================================

    def add(
        self,
        name: str,
        data: Any,
        *,
        model: str | None = None,
        dataset: int | str = 0,
        weight: float = 1.0,
        platform: str = 'cpu',
    ) -> DatasetHandle:
        """Add a named dataset to the analysis.

        Args:
            name: Identifier for this dataset. Must match ``[A-Za-z][A-Za-z0-9_]*``
                and not be a Python keyword, because it appears unquoted on the
                left of the dot in every reference to one of its parameters.
            data: A file path, or an in-memory ``Data1D``. The dataset is copied,
                so later edits to the object passed here do not reach the fit.
            model: Model expression, e.g. ``'sphere'`` or ``'sphere@hardsphere'``.
                May be set later through the handle's ``set_model``.
            dataset: Which entry to take from a file holding several.
            weight: Objective coefficient for this dataset. Leave at 1 unless you
                mean to change how much it counts; see :meth:`set_dataset_weight`.
            platform: Computation platform passed to sasmodels.

        Returns:
            A :class:`DatasetHandle` for configuring this dataset.

        Raises:
            ValueError: If the name is invalid or already used, or the weight is
                not positive and finite.
        """
        self._validate_name(name)
        weight = self._validate_weight(weight)

        fitter = SANSFitter()
        if isinstance(data, (str, os.PathLike)):
            fitter.load_data(os.fspath(data), dataset=dataset)
        else:
            # Copied at ingestion: a joint analysis holding a reference to a
            # caller's array would silently change meaning when that array did.
            fitter.set_data(copy.deepcopy(data))
        if model is not None:
            fitter.set_model(model, platform=platform)

        entry = _Entry(name, fitter, weight)
        entry.handle = DatasetHandle(self, name)
        self._entries[name] = entry
        self._invalidate()
        logger.info(
            f"{OK} Added dataset '{name}'"
            + (f" with model '{model}'" if model else '')
            + f' ({len(fitter.data.x)} points, weight {weight:g})'
        )
        return entry.handle

    def add_fitter(self, name: str, fitter: SANSFitter, *, weight: float = 1.0) -> DatasetHandle:
        """Add a dataset from an already configured :class:`~sans_fitter.SANSFitter`.

        The data, model, parameters, polydispersity, links, structure factor, Q
        range and resolution are copied into a new owned entry. The source
        fitter's fit result and engine handles are deliberately not: they belong
        to a different fit of a different problem.

        Args:
            name: Identifier for this dataset.
            fitter: The fitter to copy. It is left untouched.
            weight: Objective coefficient for this dataset.

        Raises:
            ValueError: If the name is invalid or taken, the weight is invalid,
                or the source fitter has no data or no model.
        """
        self._validate_name(name)
        weight = self._validate_weight(weight)
        if fitter.data is None:
            raise ValueError(f"Cannot add '{name}': the fitter has no data loaded.")
        if fitter.kernel is None:
            raise ValueError(f"Cannot add '{name}': the fitter has no model set.")

        owned = SANSFitter()
        owned.set_data(copy.deepcopy(fitter.data))
        q_range = fitter.get_q_range()
        if q_range is not None:
            owned.data.qmin, owned.data.qmax = q_range
        owned._resolution = ResolutionSetting(**fitter.get_resolution())
        # The kernel is rebuilt from the model expression rather than shared:
        # a sasmodels kernel can own C or OpenCL runtime state, and two entries
        # holding one object is exactly the aliasing this class avoids.
        owned.kernel = fitter.kernel
        owned._param_manager = copy.deepcopy(fitter._param_manager)
        owned._data_source = copy.deepcopy(fitter._data_source)

        entry = _Entry(name, owned, weight)
        entry.handle = DatasetHandle(self, name)
        self._entries[name] = entry
        self._invalidate()
        logger.info(f"{OK} Added dataset '{name}' from an existing fitter (weight {weight:g})")
        return entry.handle

    def remove(self, name: str) -> None:
        """Remove a dataset.

        Raises:
            KeyError: If the dataset is unknown.
            ValueError: If sharing or a constraint still refers to it. The
                message names the references to remove first, rather than
                silently dropping relationships the user set up.
        """
        self._entry(name)
        references = sorted(
            ref.qualified for ref in self._declared_references() if ref.dataset == name
        )
        if references:
            raise ValueError(
                f"Cannot remove '{name}': it is still referenced by {', '.join(references)}. "
                'Remove those with unshare(), unlink_params() or unconstrain() first.'
            )
        del self._entries[name]
        self._invalidate()
        logger.info(f"{OK} Removed dataset '{name}'")

    def set_dataset_weight(self, name: str, weight: float) -> None:
        """Set a dataset's objective coefficient ``a_d``.

        The objective becomes ``Σ a_d·χ²_d``. The default of 1 means every valid
        observation counts according to its own uncertainty, which is what makes
        the total a χ². A coefficient other than 1 states a **fitting priority**:
        it changes which compromise the optimizer prefers, while the supplied
        ``dI`` stays the error model. Uncertainties then come from the
        known-error sandwich covariance rather than the curvature of the
        reweighted objective — the reported ``chisq`` and the reported
        ``objective`` are different numbers, and the report labels both.

        Raises:
            ValueError: If the weight is not positive and finite. Zero is
                rejected: to leave a dataset out, remove it.
        """
        entry = self._entry(name)
        previous = entry.weight
        entry.weight = self._validate_weight(weight)
        self._invalidate()
        logger.info(f"{OK} Dataset '{name}' weight: {previous:g} -> {entry.weight:g}")

    # -- access -------------------------------------------------------------

    def __getitem__(self, name: str) -> DatasetHandle:
        handle = self._entry(name).handle
        assert handle is not None
        return handle

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __repr__(self) -> str:
        return f'<MultiFitter {len(self._entries)} dataset(s): {", ".join(self._entries)}>'

    @property
    def names(self) -> list[str]:
        """The dataset names, in the order they were added."""
        return list(self._entries)

    @property
    def result(self) -> MultiFitResult | None:
        """The last fit result, or None."""
        return self._result

    # =====================================================================
    # Relationships
    # =====================================================================

    def share(
        self,
        *parameters: str,
        datasets: Sequence[str] | None = None,
        source: str | None = None,
    ) -> None:
        """Make one or more parameters a single quantity across datasets.

        Sharing is symmetric: the members must already agree on their starting
        value and vary flag, or *source* must name the dataset whose settings
        win. Insertion order deliberately does not decide — which dataset's
        starting radius the fit uses is a scientific choice, not a consequence
        of the order two files were loaded in. The shared bounds are the
        intersection of the members', so a limit set on any member still holds.

        Args:
            *parameters: Unqualified names, e.g. ``'radius'``, ``'radius_pd'``.
            datasets: Which datasets take part. Defaults to all of them.
                Membership is fixed here: a dataset added later is not enrolled.
            source: Dataset whose value and vary flag the group adopts.

        Raises:
            KeyError: If a dataset or parameter is unknown.
            ConstraintError: If members disagree with no *source*, their bounds
                do not overlap, or the result contradicts another relationship.
        """
        if not parameters:
            raise ValueError('share() needs at least one parameter name.')
        chosen = list(self._entries) if datasets is None else list(datasets)
        for name in chosen:
            self._entry(name)
        if len(chosen) < 2:
            raise ValueError(
                f'share() needs at least two datasets, got {len(chosen)}. '
                'Add another dataset, or use set_param() for a single one.'
            )
        if source is not None and source not in chosen:
            raise ValueError(
                f"source='{source}' is not among the datasets being shared ({', '.join(chosen)})."
            )

        groups = []
        for parameter in parameters:
            members = tuple(self._reference(f'{name}.{parameter}') for name in chosen)
            picked = self._reference(f'{source}.{parameter}') if source else None
            groups.append(ShareGroup(members=members, source=picked, label=parameter))

        self._transaction(lambda: self._share_groups.extend(groups))
        logger.info(
            f'{OK} Sharing {", ".join(parameters)} across {", ".join(chosen)}'
            + (f' (source: {source})' if source else '')
        )

    def unshare(self, parameter: str, *, dataset: str) -> None:
        """Detach one dataset's parameter from a shared group.

        It keeps the value the group currently holds — that is the value the
        rest of the analysis was built around — along with its own bounds and
        vary flag. A group left with one member is dropped.

        Raises:
            KeyError: If the dataset or parameter is unknown.
            ValueError: If the parameter is not shared.
        """
        ref = self._reference(f'{dataset}.{parameter}')
        remaining: list[ShareGroup] = []
        found = False
        for group in self._share_groups:
            if ref not in group.members:
                remaining.append(group)
                continue
            found = True
            members = tuple(member for member in group.members if member != ref)
            if len(members) < 2:
                continue
            source = group.source if group.source in members else None
            remaining.append(ShareGroup(members=members, source=source, label=group.label))
        if not found:
            raise ValueError(f'{ref.qualified} is not part of a shared group.')

        members = self._class_members(ref)
        touched = {member.dataset for member in members}

        def apply() -> None:
            # Every member first, not just the one leaving: the group's value
            # may never have been written down anywhere, and the ones staying
            # behind must not move either.
            self._materialize(members)
            self._replace_shares(remaining)

        self._transaction(apply, entries=tuple(touched))
        logger.info(f'{OK} Detached {ref.qualified} from its shared group')

    def link_params(self, target: str, *, to: str) -> None:
        """Make one parameter follow another, in either dataset.

        Directed, unlike :meth:`share`: *target* adopts *to*'s value, vary flag
        **and bounds**, and its own settings are discarded. That is what the
        single-dataset :meth:`SANSFitter.link_params` has always meant. Use it
        to relate parameters with different names (``'warm.length'`` following
        ``'cold.length'``); use :meth:`share` when the parameters are the same
        measurement seen from several datasets, and
        ``constrain(target, 'other.parameter')`` when the target's own limits
        must continue to hold.

        Raises:
            KeyError: If either reference is unknown.
            ValueError: If the two references are the same parameter, or the
                follower is already defined by a constraint.
            ConstraintError: If the link contradicts another relationship.
        """
        follower = self._reference(target)
        leader = self._reference(to)
        if follower == leader:
            raise ValueError(f'Cannot link {follower.qualified} to itself.')
        spec = self._constraints.get(follower)
        if spec is not None:
            raise ValueError(
                f"{follower.qualified} is already defined by the constraint '{spec.text}'. "
                'Two definitions of one parameter would both have to hold; remove that '
                'one with unconstrain() first if the link is what you want.'
            )
        self._transaction(lambda: self._directed.__setitem__(follower, leader))
        logger.info(f'{OK} Linked {follower.qualified} -> {leader.qualified}')

    def unlink_params(self, target: str) -> None:
        """Remove a link created by :meth:`link_params`.

        Raises:
            ValueError: If the reference does not follow anything.
        """
        follower = self._reference(target)
        if follower not in self._directed:
            raise ValueError(f'{follower.qualified} does not follow another parameter.')

        def apply() -> None:
            # Same reason as unshare(): the follower's value lives in the graph
            # until the link is gone, so write it down before removing it.
            self._materialize([follower])
            self._directed.pop(follower)

        self._transaction(apply, entries=(follower.dataset,))
        logger.info(f'{OK} Unlinked {follower.qualified}')

    def constrain(self, target: str, expression: float | str) -> None:
        """Define a parameter's value, removing it from the fitted coordinates.

        Three forms are accepted:

        - a **number**, or a string holding one: ``constrain('d2o.sld_solvent', 6.34)``
          pins the parameter;
        - a **reference**: ``constrain('d2o.background', 'h2o.background')`` is
          equality, the same relationship :meth:`link_params` creates;
        - an **expression**: ``constrain('d2o.scale', '0.8 * h2o.scale')`` or
          ``constrain('warm.length', '2 * cold.radius + 10')``.

        The expression grammar is small on purpose — numbers, qualified
        references, ``+ - * /`` and integer powers — and is interpreted, never
        executed. Binding an expression replaces the target's parameter object
        inside bumps, which discards the target's own limits, so those limits are
        proved here instead: a constraint is accepted only if it cannot leave
        them over the ranges its inputs are allowed to explore. Where the
        relationship is linear in one parameter that proof is constructive and
        the input's range is narrowed to guarantee it; where the arithmetic
        cannot certify it, the constraint is refused and the message names the
        parameters to tighten. That is conservative — some feasible nonlinear
        relationships need tighter bounds than they strictly must — and it is
        preferred to a limit that quietly stops applying.

        **The target keeps its own limits under every form**, including a bare
        reference. That is the difference from :meth:`link_params`, where the
        follower deliberately adopts its target's configuration: the two
        spellings of one relationship would otherwise permit different values.

        Constraining the same parameter again replaces the definition. A
        parameter that already follows another through :meth:`link_params`
        cannot also be constrained — unlink it first — because two definitions
        of one value would both have to hold.

        Raises:
            KeyError: If a reference is unknown.
            ExpressionError: If the expression is outside the grammar.
            ValueError: If the target already follows another parameter.
            ConstraintError: If the value or range cannot be kept inside the
                target's bounds, or the definition conflicts with another.
        """
        ref = self._reference(target)
        if ref in self._directed:
            raise ValueError(
                f'{ref.qualified} already follows {self._directed[ref].qualified} through '
                'link_params(). Remove that link with unlink_params() before constraining '
                'it, so the parameter has one definition rather than two.'
            )
        spec = self._build_constraint(ref, expression)
        self._transaction(lambda: self._constraints.__setitem__(ref, spec))
        logger.info(f'{OK} Constrained {ref.qualified} = {spec.text}')

    def unconstrain(self, target: str) -> None:
        """Remove a constraint, restoring an independent parameter.

        The parameter keeps the value the constraint last resolved to — that is
        the state the rest of the analysis was built around — and gets back the
        bounds and vary flag it had before being constrained.

        Raises:
            ValueError: If the parameter is not constrained.
        """
        ref = self._reference(target)
        spec = self._constraints.get(ref)
        if spec is None:
            raise ValueError(f'{ref.qualified} is not constrained.')

        resolved = self._resolved_values().get(ref)
        # Bounds and the vary flag come back; the value does not. The rest of the
        # analysis was configured around what the constraint resolved to, so
        # reinstating the value from before it was applied would silently move
        # the parameter as a side effect of deleting a relationship.
        previous = {key: spec.previous.get(key) for key in ('vary', 'min', 'max')}

        def apply() -> None:
            del self._constraints[ref]
            if resolved is not None:
                self._write_value(ref, resolved)
            self._write_configuration(ref, previous)

        self._transaction(apply, entries=(ref.dataset,))
        logger.info(f'{OK} Removed the constraint on {ref.qualified}')

    def set_param(
        self,
        reference: str,
        value: float | None = None,
        min: float | None = None,
        max: float | None = None,
        vary: bool | None = None,
    ) -> None:
        """Configure a parameter through the graph.

        *reference* is normally qualified (``'h2o.radius'``). An unqualified name
        is accepted when exactly one shared group carries it, which is the common
        case after ``share('radius')``.

        A value or vary flag set on a member of a shared group is applied to
        every member, so the group keeps agreeing. A bound is applied to the
        named member only; the group's effective range is the intersection, and
        is recomputed here.

        Raises:
            KeyError: If the reference is unknown.
            ValueError: If the parameter is defined by a constraint — change or
                remove the constraint instead.
        """
        ref = self._reference(reference)
        spec = self._constraints.get(ref)
        if spec is not None and (value is not None or vary is not None):
            raise ValueError(
                f"{ref.qualified} is defined by the constraint '{spec.text}', so its value "
                'is not free to set. Use unconstrain() first, or change the constraint.'
            )

        # Value and vary belong to the quantity, bounds to the individual
        # parameter, so the two travel different distances through the graph.
        siblings = self._class_members(ref) if (value is not None or vary is not None) else (ref,)
        touched = {member.dataset for member in siblings} | {ref.dataset}

        def apply() -> None:
            for member in siblings:
                self._write_configuration(member, {'value': value, 'vary': vary})
            self._write_configuration(ref, {'min': min, 'max': max})

        self._transaction(apply, entries=tuple(touched))

    # -- inspection ---------------------------------------------------------

    def get_constraints(self) -> list[dict[str, Any]]:
        """Every explicit constraint, as inspectable data."""
        return [
            {
                'target': ref.qualified,
                'kind': spec.kind,
                'text': spec.text,
                'depends_on': sorted(dep.qualified for dep in spec.dependencies),
            }
            for ref, spec in sorted(self._constraints.items(), key=lambda item: item[0].qualified)
        ] + [
            {
                'target': follower.qualified,
                'kind': 'equality',
                'text': leader.qualified,
                'depends_on': [leader.qualified],
            }
            for follower, leader in sorted(
                self._directed.items(), key=lambda item: item[0].qualified
            )
        ]

    def get_sharing(self) -> list[dict[str, Any]]:
        """Every quantity held by more than one parameter, after compilation."""
        graph = self._compiled()
        return [
            {
                'root': entry.label,
                'members': [ref.qualified for ref in entry.members],
                'status': entry.kind,
            }
            for entry in graph.classes
            if entry.shared
        ]

    def get_parameter_table(self) -> list[dict[str, Any]]:
        """Every parameter of every dataset, with its status in the graph."""
        graph = self._compiled()
        values = graph.class_values()
        rows: list[dict[str, Any]] = []
        for entry_class in graph.classes:
            for ref in entry_class.members:
                rows.append(
                    {
                        'parameter': ref.qualified,
                        'value': values[entry_class.index],
                        'status': self._status(entry_class, ref),
                        'root': entry_class.label,
                        'expression': entry_class.constraint_text,
                        'min': entry_class.minimum,
                        'max': entry_class.maximum,
                    }
                )
        return sorted(rows, key=lambda row: row['parameter'])

    def describe(self) -> None:
        """Print the datasets, relationships and free-parameter count.

        The summary to read before fitting: it states how many independent
        coordinates the graph leaves, which is the number that shared parameters
        are supposed to reduce and the easiest thing to get wrong.
        """
        graph = self._compiled()
        print(f'\n{"=" * 78}')
        print(f'MultiFitter: {len(self._entries)} dataset(s), {graph.n_free} free parameter(s)')
        print('=' * 78)

        for entry in self._entries.values():
            q_range = entry.fitter.get_q_range()
            selected = (
                int(get_fit_index(entry.fitter.data).sum()) if entry.fitter.data is not None else 0
            )
            print(
                f'  {entry.name:<14} {entry.fitter.model_name or "(no model)":<24} '
                f'{selected:>5} pts  weight {entry.weight:<6g} '
                f'Q [{q_range[0]:.4g}, {q_range[1]:.4g}]  '
                f'{entry.fitter._resolution.describe()}'
            )

        rows = [row for row in self.get_parameter_table() if row['status'] != 'fixed']
        if rows:
            print('\nParameters taking part in the fit:')
            width = max(len(row['parameter']) for row in rows)
            for row in rows:
                detail = f' = {row["expression"]}' if row['expression'] else ''
                root = '' if row['parameter'] == row['root'] else f'  [= {row["root"]}]'
                print(
                    f'  {row["parameter"]:<{width}}  {row["value"]:>12.6g}  '
                    f'{row["status"]}{detail}{root}'
                )

        constraints = self.get_constraints()
        if constraints:
            print('\nConstraints:')
            for item in constraints:
                print(f'  {item["target"]} = {item["text"]}  ({item["kind"]})')

        print(f'\nFree parameters: {", ".join(graph.labels) or "none"}')
        print(f'{"=" * 78}\n')

    # =====================================================================
    # Evaluation
    # =====================================================================

    def calculate(
        self, q: Mapping[str, np.ndarray] | None = None, dq: Mapping[str, float] | None = None
    ) -> dict[str, np.ndarray]:
        """Evaluate every model at the current, constraint-resolved parameters.

        Shared values and expressions hold here exactly as they do during a fit,
        so a preview is a preview of the constrained model rather than of
        whatever the individual parameter tables happen to say.

        Args:
            q: Optional per-dataset Q grids, ``{'h2o': array, ...}``. Datasets
                left out use their own data grid and resolution.
            dq: Optional per-dataset relative resolution widths for those grids.

        Returns:
            Dataset name -> intensities. On a data grid the array is full length,
            with NaN where a point is excluded from the fit.
        """
        self._require_ready()
        graph = self._compiled()
        resolved = graph.resolve()

        curves: dict[str, np.ndarray] = {}
        for entry in self._entries.values():
            grid = None if q is None else q.get(entry.name)
            width = None if dq is None else dq.get(entry.name)
            target = (
                entry.fitter._evaluation_data(warn=False)
                if grid is None
                else theory_data(grid, width)
            )
            curve, fit_index = self._evaluate_entry(entry, target, resolved)
            full = np.full(len(fit_index), np.nan)
            full[fit_index] = curve
            curves[entry.name] = full
        return curves

    def plot_model(
        self,
        show_residuals: bool = True,
        log_scale: bool = True,
        show: bool | None = None,
        show_components: bool = False,
    ) -> Figure:
        """Plot every dataset with its model at the current parameters, without fitting.

        One data-and-model panel per dataset, each with its own axes because the
        Q ranges need not match. Constraints are resolved first, so this shows
        the model the fit will start from. Unlike :meth:`plot_results`, this
        reads the datasets as they are now — that is the point of a preview.

        Args:
            show_components: For a '+' mixture model, overlay one dashed curve
                per component. A documented no-op elsewhere.
        """
        self._require_ready()
        graph = self._compiled()
        resolved = graph.resolve()

        panels = []
        for entry in self._entries.values():
            evaluation_data = entry.fitter._evaluation_data(warn=False)
            curve, fit_index = self._evaluate_entry(entry, evaluation_data, resolved)
            components = None
            if show_components:
                components = entry.fitter._compute_component_curves(
                    self._canonical_values(entry, resolved)
                )
            panels.append(
                multi_plotting.DatasetPanel.from_dataset(
                    name=entry.name,
                    model=entry.fitter.model_name or '',
                    data=entry.fitter.data,
                    fitted_curve=curve,
                    fit_index=fit_index,
                    residuals=_preview_residuals(evaluation_data, curve, fit_index),
                    weight=entry.weight,
                    component_curves=components,
                )
            )
        return multi_plotting.plot_multi(
            panels,
            title=f'Model preview: {len(panels)} datasets, {graph.n_free} free parameters',
            show_residuals=show_residuals,
            log_scale=log_scale,
            show=show,
            preview=True,
            show_components=show_components,
        )

    def plot_results(
        self,
        show_residuals: bool = True,
        log_scale: bool = True,
        show: bool | None = None,
        objective_residuals: bool = False,
        show_components: bool = False,
    ) -> Figure:
        """Plot the fitted model for every dataset.

        Drawn entirely from the result's own snapshot — the observations, the
        selection and the model curves it was built with — so the figure shows
        one coherent fit even if a dataset has since been reconfigured or
        removed. Use :meth:`plot_model` to see the datasets as they are now.

        Args:
            objective_residuals: Show ``sqrt(weight)·residual`` — what the
                optimizer minimised — instead of the residual in the dataset's
                own sigma units. Only differs when a dataset factor is set, and
                is labelled differently so the two cannot be confused.
            show_components: For a '+' mixture model, overlay one dashed curve
                per component. A documented no-op elsewhere.
        """
        result = self._require_result()
        self._warn_if_stale()
        panels = [
            multi_plotting.DatasetPanel(
                name=entry.name,
                model=entry.model,
                q=entry.observed_q,
                intensity=entry.observed_intensity,
                uncertainty=entry.observed_uncertainty,
                dq=entry.observed_dq,
                fitted_curve=entry.fitted_curve,
                fit_index=entry.fit_index,
                residuals=(entry.objective_residuals if objective_residuals else entry.residuals),
                weight=entry.weight,
                component_curves=entry.component_curves,
            )
            for entry in result.datasets.values()
        ]
        symbol = 'χ²'
        return multi_plotting.plot_multi(
            panels,
            title=(
                f'Simultaneous fit: {result.n_datasets} datasets, '
                f'{symbol}/dof = {result.reduced_chisq:.4f}'
            ),
            show_residuals=show_residuals,
            log_scale=log_scale,
            show=show,
            preview=False,
            residual_label=('Residuals (√w·σ)' if objective_residuals else 'Residuals (σ)'),
            show_components=show_components,
        )

    # =====================================================================
    # Fitting
    # =====================================================================

    def fit(
        self,
        engine: Literal['bumps'] = 'bumps',
        method: str | None = None,
        **kwargs: Any,
    ) -> MultiFitResult:
        """Fit every dataset together, through one bumps problem.

        Args:
            engine: Only ``'bumps'`` is supported. The scipy backend and joint
                DREAM sampling are planned and will keep this signature.
            method: bumps optimizer — ``'amoeba'`` (default), ``'lm'``,
                ``'newton'``, ``'de'``.
            **kwargs: Passed to the bumps fitter.

        Returns:
            A :class:`~sans_fitter.multi_results.MultiFitResult`.

        Raises:
            ValueError: If the analysis is not ready to fit, an engine other
                than bumps is requested, or a dataset fails the data preflight.
            ConstraintError: If the parameter graph cannot be compiled.
        """
        if engine != 'bumps':
            raise ValueError(
                f"Unknown engine '{engine}'. Simultaneous fitting currently supports "
                "'bumps' only; the scipy path is not implemented, and reporting one "
                'as if it were would give results no test has checked.'
            )
        self._require_ready()
        graph = self._compiled()
        self._preflight(graph)

        specs = [self._entry_spec(entry) for entry in self._entries.values()]
        outcome = run_multi_bumps(specs, graph, method=method or 'amoeba', **kwargs)

        result = self._build_result(graph, outcome, method or 'amoeba')
        # Commit last: a failure anywhere above leaves every dataset exactly as
        # it was, which is what makes a failed fit safe to retry. The context is
        # taken afterwards, because it digests the parameter values and would
        # otherwise describe the configuration the fit started from — making
        # every result look stale the moment it was produced.
        self._commit(graph, outcome.root_values)
        self._attach_component_curves(result)
        result.fit_context = self._context()
        self._result = result

        logger.info(f'\n{OK} Simultaneous fit completed!\n{MultiFitReport(result)}')
        self._warn_about_bounds(result)
        return result

    def get_fit_report(self) -> MultiFitReport:
        """A self-rendering report for the last fit.

        Prints as a table in a terminal, renders as HTML in a notebook, and
        converts to Markdown with ``to_markdown()``.

        Raises:
            ValueError: If no fit has been run.
        """
        return MultiFitReport(self._require_result())

    def save_results(self, directory: str) -> None:
        """Write the fit to a directory of CSV files.

        Produces ``parameters.csv`` (every parameter with its status, root and
        uncertainty), ``datasets.csv`` (per-dataset diagnostics),
        ``covariance.csv`` (over the free roots, when available), one
        ``<dataset>_curve.csv`` per dataset, and ``manifest.txt`` tying them to
        one fit. Curve files carry Q, I, dI, the fitted intensity, the residual
        in sigma units and the objective residual, for the fitted points only.

        Raises:
            ValueError: If no fit has been run.
        """
        result = self._require_result()
        self._warn_if_stale()

        # Everything is rendered before anything is written, so a failure
        # half-way cannot leave a directory holding some files from this fit and
        # some from the last one.
        files: dict[str, str] = {
            'parameters.csv': _csv_text(
                ['parameter', 'value', 'stderr', 'formatted', 'status', 'root', 'expression'],
                [
                    [
                        entry.qualified,
                        _csv_number(entry.value),
                        '' if entry.stderr is None else _csv_number(entry.stderr),
                        entry.formatted,
                        entry.status,
                        entry.root,
                        entry.expression,
                    ]
                    for entry in result.parameters.values()
                ],
            ),
            'datasets.csv': _csv_text(
                [
                    'dataset',
                    'model',
                    'n_points',
                    'chisq',
                    'objective_contribution',
                    'rms_residual',
                    'mean_squared_residual',
                    'weight',
                    'qmin',
                    'qmax',
                    'resolution',
                ],
                [
                    [
                        entry.name,
                        entry.model,
                        str(entry.n_points),
                        _csv_number(entry.chisq),
                        _csv_number(entry.objective_contribution),
                        _csv_number(entry.rms_residual),
                        _csv_number(entry.mean_squared_residual),
                        _csv_number(entry.weight),
                        '' if entry.q_range is None else _csv_number(entry.q_range[0]),
                        '' if entry.q_range is None else _csv_number(entry.q_range[1]),
                        entry.resolution,
                    ]
                    for entry in result.datasets.values()
                ],
            ),
        }

        if result.cov is not None:
            files['covariance.csv'] = _csv_text(
                ['parameter', *result.cov_labels],
                [
                    [label, *[_csv_number(value) for value in row]]
                    for label, row in zip(result.cov_labels, np.asarray(result.cov), strict=True)
                ],
            )

        for entry in result.datasets.values():
            files[f'{entry.name}_curve.csv'] = self._curve_text(entry, result)

        files[MANIFEST_NAME] = self._manifest_text(result, sorted(files))

        os.makedirs(directory, exist_ok=True)
        obsolete = _previously_exported(directory) - set(files)
        for name, text in files.items():
            atomic_write(os.path.join(directory, name), text)
        # Only files a previous export of ours listed are removed, and only once
        # the new ones are safely in place. A stale covariance.csv beside a fit
        # that has none would otherwise read as this fit's covariance.
        for name in sorted(obsolete):
            try:
                os.remove(os.path.join(directory, name))
            except OSError:
                logger.debug(f'Could not remove the superseded export file {name}.')

        logger.info(f'{OK} Results saved to {directory}')

    # =====================================================================
    # Internals: references and state
    # =====================================================================

    def _entry(self, name: str) -> _Entry:
        try:
            return self._entries[name]
        except KeyError:
            available = ', '.join(self._entries) or 'none'
            raise KeyError(f"Unknown dataset '{name}'. Available: {available}") from None

    def _validate_name(self, name: str) -> None:
        if name in self._entries:
            raise ValueError(
                f"A dataset called '{name}' has already been added. Names identify "
                'parameters, so reusing one would make every reference to it ambiguous.'
            )
        if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name):
            raise ValueError(
                f'Invalid dataset name {name!r}. Use a letter followed by letters, '
                "digits or underscores, e.g. 'h2o' — the name appears unquoted in "
                "every reference such as 'h2o.radius'."
            )
        if keyword.iskeyword(name):
            raise ValueError(f"Dataset name '{name}' is a Python keyword.")

    def _validate_weight(self, weight: float) -> float:
        value = float(weight)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f'Dataset weight must be positive and finite, got {weight!r}. '
                'To leave a dataset out of the fit, remove it rather than '
                'weighting it to zero.'
            )
        return value

    def _reference(self, text: str) -> ParameterRef:
        """Turn ``'h2o.radius'`` into a resolved reference.

        An unqualified name is accepted when exactly one shared group carries
        it, which is what makes ``set_param('radius', value=45)`` work after
        ``share('radius')`` without naming a dataset arbitrarily.
        """
        if isinstance(text, str) and '.' not in text:
            return self._unqualified(text)
        dataset, local = parse_reference(text)
        entry = self._entry(dataset)
        return self._resolve_local(entry, local)

    def _unqualified(self, name: str) -> ParameterRef:
        matches = [group for group in self._share_groups if group.label == name]
        if len(matches) == 1:
            return matches[0].members[0]
        if not matches:
            raise KeyError(
                f"'{name}' is not a shared parameter, so it does not identify one "
                f"quantity. Qualify it with a dataset, for example 'h2o.{name}'."
            )
        raise KeyError(f"'{name}' names {len(matches)} shared groups. Qualify it with a dataset.")

    def _resolve_local(self, entry: _Entry, local: str) -> ParameterRef:
        """Resolve one dataset-local parameter name, including PD widths."""
        manager = entry.fitter._param_manager
        if not manager.params:
            raise KeyError(
                f"Dataset '{entry.name}' has no model, so it has no parameters yet. "
                'Call set_model() on its handle first.'
            )

        if local.endswith('_pd'):
            base = local[:-3]
            try:
                key = manager.resolve_name(base)
            except KeyError:
                key = None
            if key is not None and manager._resolve_canonical(key) in (
                manager.get_polydisperse_parameters()
            ):
                return ParameterRef(entry.name, key, pd=True)

        try:
            key = manager.resolve_name(local)
        except KeyError as error:
            raise KeyError(f"Dataset '{entry.name}': {error.args[0]}") from None
        if key not in manager.params:
            raise KeyError(
                f"Dataset '{entry.name}': parameter '{local}' is not independently "
                'settable under this model.'
            )
        return ParameterRef(entry.name, key, pd=False)

    def _declared_references(self) -> set[ParameterRef]:
        """Every reference named by a share, link or constraint."""
        references: set[ParameterRef] = set()
        for group in self._share_groups:
            references.update(group.members)
        for follower, leader in self._directed.items():
            references.update((follower, leader))
        for ref, spec in self._constraints.items():
            references.add(ref)
            references.update(spec.dependencies)
        return references

    def _build_constraint(self, ref: ParameterRef, expression: float | str) -> ConstraintSpec:
        """Classify a constraint and remember what to restore on unconstrain()."""
        previous = self._configuration(ref)

        if isinstance(expression, (int, float)) and not isinstance(expression, bool):
            value = float(expression)
            if not math.isfinite(value):
                raise ConstraintError(f'Cannot constrain {ref.qualified} to {expression}.')
            return ConstraintSpec(
                target=ref, kind='constant', text=f'{value:g}', value=value, previous=previous
            )

        # Collapsed to one line: the text is echoed into reports and CSV exports,
        # and a constraint may legally span lines inside parentheses. The same
        # reasoning as fitting.base.normalize_message.
        text = ' '.join(str(expression).split())
        try:
            value = float(text)
        except ValueError:
            pass
        else:
            if not math.isfinite(value):
                raise ConstraintError(f"Cannot constrain {ref.qualified} to '{text}'.")
            return ConstraintSpec(
                target=ref, kind='constant', text=text, value=value, previous=previous
            )

        parsed = parse_expression(
            text, lambda dataset, local: self._reference(f'{dataset}.{local}')
        )
        if ref in parsed.refs:
            raise ConstraintError(
                f"Constraint '{ref.qualified} = {text}' defines the parameter in terms of itself."
            )

        if not parsed.refs:
            # Arithmetic over literals alone is a number, and is folded into one
            # here so that every layer agrees it is one: a derived node with no
            # inputs would translate to a bare float, which the optimizer backend
            # cannot bind, and would be reported as 'derived' with a gradient of
            # nothing. Evaluating also enforces the domain, so '1 / 0' is
            # rejected here rather than surviving as a definition.
            value = parsed.evaluate({})
            if not math.isfinite(value):
                raise ConstraintError(
                    f"Constraint '{ref.qualified} = {text}' does not evaluate to a finite number."
                )
            return ConstraintSpec(
                target=ref, kind='constant', text=text, value=float(value), previous=previous
            )

        # A lone reference is equality, which the graph merges into one class
        # rather than carrying as an expression — same relationship, one fewer
        # moving part, and an uncertainty that is copied rather than propagated.
        # The target keeps its own bounds either way; see compile_graph.
        from .modeling.constraints import Reference as _Reference

        if isinstance(parsed.root, _Reference):
            return ConstraintSpec(
                target=ref,
                kind='equality',
                text=text,
                source=parsed.root.ref,
                previous=previous,
            )
        return ConstraintSpec(
            target=ref, kind='expression', text=text, expression=parsed, previous=previous
        )

    def _configuration(self, ref: ParameterRef) -> dict[str, Any]:
        """The independent value/vary/bounds of one reference."""
        fitter = self._entry(ref.dataset).fitter
        if ref.pd:
            config = fitter.get_pd_param(ref.name)
            low, high = PD_WIDTH_RANGE
            return {
                'value': float(config['pd']),
                'vary': bool(config.get('vary', False)),
                'min': low,
                'max': high,
            }
        info = fitter.params[ref.name]
        return {
            'value': float(info['value']),
            'vary': bool(info['vary']),
            'min': float(info['min']),
            'max': float(info['max']),
        }

    def _write_configuration(self, ref: ParameterRef, changes: Mapping[str, Any]) -> None:
        """Apply value/vary/min/max to one reference, bypassing link guards.

        The graph, not the per-dataset manager, decides what follows what here,
        so a write to a local link follower is legitimate and the manager's own
        refusal to set one would be wrong.
        """
        fitter = self._entry(ref.dataset).fitter
        if ref.pd:
            fitter.set_pd_param(
                ref.name,
                pd_width=changes.get('value'),
                vary=changes.get('vary'),
            )
            return
        info = fitter.params[ref.name]
        for field in ('value', 'min', 'max', 'vary'):
            if changes.get(field) is not None:
                info[field] = changes[field]

    def _write_value(self, ref: ParameterRef, value: float) -> None:
        self._write_configuration(ref, {'value': float(value)})

    def _class_members(self, ref: ParameterRef) -> tuple[ParameterRef, ...]:
        """Every reference that is the same quantity as *ref*."""
        graph = self._compiled()
        index = graph.class_of.get(ref)
        return graph.classes[index].members if index is not None else (ref,)

    def _replace_shares(self, groups: list[ShareGroup]) -> None:
        self._share_groups[:] = groups

    # -- transactions -------------------------------------------------------

    def _invalidate(self) -> None:
        self._graph = None

    def _declarations(self) -> tuple[Any, ...]:
        return (
            list(self._share_groups),
            dict(self._directed),
            dict(self._constraints),
        )

    def _restore_declarations(self, state: tuple[Any, ...]) -> None:
        shares, directed, constraints = state
        self._share_groups[:] = shares
        self._directed.clear()
        self._directed.update(directed)
        self._constraints.clear()
        self._constraints.update(constraints)

    def _transaction(self, action, entries: Sequence[str] = ()) -> None:
        """Apply a configuration change, or leave nothing behind if it fails.

        Recompiling after every change is what lets this class promise that a
        configured analysis is always fittable: the error arrives at the call
        that caused it, with both sides of the contradiction in scope, instead
        of at ``fit()`` where the cause is several steps away.
        """
        saved_declarations = self._declarations()
        saved_entries = {name: self._entry(name).snapshot() for name in entries}
        try:
            action()
            self._graph = None
            # Keep what this compile produced: it already describes the state
            # the change left behind, and discarding it only means the next read
            # compiles the same graph a second time.
            self._compiled()
        except Exception:
            self._restore_declarations(saved_declarations)
            for name, state in saved_entries.items():
                self._entry(name).restore(state)
            self._graph = None
            raise

    def _mutate(self, name: str, action) -> None:
        """A handle-level change to one dataset, revalidated and rolled back."""
        self._transaction(action, entries=(name,))

    # -- graph --------------------------------------------------------------

    def _descriptors(self) -> dict[ParameterRef, ParameterDescriptor]:
        """Every parameter the graph can address.

        Polydispersity widths appear only when they matter: the entry has PD
        enabled and the width is either active already or named by a
        relationship. Listing every width of every model would bury the
        parameter table under quantities nobody is fitting.
        """
        declared = self._declared_references()
        descriptors: dict[ParameterRef, ParameterDescriptor] = {}

        for entry in self._entries.values():
            manager = entry.fitter._param_manager
            units = _kernel_units(entry.fitter.kernel)
            for name, info in manager.params.items():
                ref = ParameterRef(entry.name, name)
                descriptors[ref] = ParameterDescriptor(
                    ref=ref,
                    value=float(info['value']),
                    vary=bool(info['vary']),
                    minimum=float(info['min']),
                    maximum=float(info['max']),
                    units=units.get(manager._resolve_canonical(name), ''),
                )

            if not manager.is_pd_enabled():
                continue
            for canonical in manager.get_polydisperse_parameters():
                display = manager.to_display_name(canonical)
                try:
                    key = manager.resolve_name(display)
                except KeyError:
                    continue
                ref = ParameterRef(entry.name, key, pd=True)
                config = manager.get_pd_param(canonical)
                active = config['pd'] > 0 or config.get('vary', False)
                if not (active or ref in declared):
                    continue
                low, high = PD_WIDTH_RANGE
                descriptors[ref] = ParameterDescriptor(
                    ref=ref,
                    value=float(config['pd']),
                    vary=bool(config.get('vary', False)),
                    minimum=low,
                    maximum=high,
                    pd_type=str(config.get('pd_type', '')),
                    pd_enabled=True,
                )
        return descriptors

    def _local_links(self) -> dict[ParameterRef, ParameterRef]:
        """Per-dataset equality links, promoted into the global graph.

        Mandatory edges: a link a dataset already carried (``link_params``,
        ``shared=`` on ``set_models``, ``radius_effective_mode='link_radius'``)
        still holds inside a joint fit, and compiling it here rather than
        letting each engine reapply it is what keeps one binding map for
        everything.
        """
        links: dict[ParameterRef, ParameterRef] = {}
        for entry in self._entries.values():
            for follower, target in entry.fitter.get_links().items():
                links[ParameterRef(entry.name, follower)] = ParameterRef(entry.name, target)
        return links

    def _compiled(self) -> CompiledGraph:
        if self._graph is None:
            descriptors = self._descriptors()
            self._check_references(descriptors)
            self._graph = compile_graph(
                descriptors,
                local_links=self._local_links(),
                directed_links=self._directed,
                share_groups=self._share_groups,
                constraints=self._constraints,
            )
        return self._graph

    def _check_references(self, descriptors: Mapping[ParameterRef, ParameterDescriptor]) -> None:
        """Fail with names, not a KeyError, when a relationship lost its parameter."""
        missing = sorted(
            (ref for ref in self._declared_references() if ref not in descriptors),
            key=lambda ref: ref.qualified,
        )
        if not missing:
            return

        disabled = [
            ref
            for ref in missing
            if ref.pd and not self._entries[ref.dataset].fitter.is_polydispersity_enabled()
        ]
        if disabled:
            names = ', '.join(sorted({ref.dataset for ref in disabled}))
            raise ConstraintError(
                f'{disabled[0].qualified} is part of a relationship, but polydispersity '
                f'is switched off for {names}, so the width takes no part in the fit. '
                f"Call fit['{disabled[0].dataset}'].enable_polydispersity(True), or drop "
                'the relationship with unshare()/unconstrain().'
            )
        raise ConstraintError(
            f'These references no longer exist: '
            f'{", ".join(ref.qualified for ref in missing)}. A model changed underneath '
            'a relationship. Remove the relationship with unshare(), unlink_params() or '
            'unconstrain(), or restore the parameter.'
        )

    def _resolved_values(self) -> dict[ParameterRef, float]:
        """Graph-resolved values, or the raw ones if the graph does not compile.

        The single read path for numeric configuration. A shared, linked or
        constrained parameter's value lives in the graph, not in the dataset
        that happens to hold a copy of it, so every public getter that reports a
        number goes through here — otherwise two getters answer the same
        question differently depending on which layer they consulted.

        Returns an empty mapping rather than raising when the graph does not
        compile: read-only views must stay usable mid-edit.
        """
        try:
            return self._compiled().resolve()
        except ConstraintError:
            return {}

    def _resolved_value(self, ref: ParameterRef) -> float | None:
        """One graph-resolved value, or None when the graph cannot supply it."""
        return self._resolved_values().get(ref)

    def _cross_dataset_equalities(self) -> dict[ParameterRef, ParameterRef]:
        """Equality relationships declared on the parent, follower -> target.

        Both spellings that create one: :meth:`link_params` and a
        ``constrain(target, 'other.parameter')`` whose expression is a bare
        reference.
        """
        edges = dict(self._directed)
        for ref, spec in self._constraints.items():
            if spec.kind == 'equality' and spec.source is not None:
                edges[ref] = spec.source
        return edges

    def _materialize(self, refs: Sequence[ParameterRef]) -> None:
        """Write each reference's currently resolved value into its dataset.

        Called before a relationship is removed. Until that point a member's
        value lives in the graph and the dataset may still hold whatever it was
        configured with; detaching without materializing first would make the
        parameter jump back to a value the analysis stopped using when the
        relationship was created.
        """
        resolved = self._resolved_values()
        for ref in refs:
            value = resolved.get(ref)
            if value is not None:
                self._write_value(ref, value)

    def _configure_pd(self, dataset: str, param_name: str, **changes: Any) -> None:
        """Apply one polydispersity setter call as a single transaction."""
        ref = self._reference(f'{dataset}.{param_name}_pd')
        spec = self._constraints.get(ref)
        width, vary = changes.pop('pd_width', None), changes.pop('vary', None)
        if spec is not None and (width is not None or vary is not None):
            raise ValueError(
                f"{ref.qualified} is defined by the constraint '{spec.text}', so its width "
                'is not free to set. Use unconstrain() first, or change the constraint.'
            )

        # The width is the quantity and travels to every member; the
        # distribution and quadrature settings describe how this dataset
        # evaluates it and stay here.
        settings = {key: value for key, value in changes.items() if value is not None}
        siblings = self._class_members(ref) if (width is not None or vary is not None) else ()
        touched = {dataset} | {member.dataset for member in siblings}

        def apply() -> None:
            if settings:
                self._entry(dataset).fitter.set_pd_param(param_name, **settings)
            for member in siblings:
                self._write_configuration(member, {'value': width, 'vary': vary})

        self._transaction(apply, entries=tuple(touched))

    @staticmethod
    def _status(entry_class: Any, ref: ParameterRef) -> str:
        if entry_class.kind == 'derived':
            return 'derived'
        if entry_class.kind == 'constant':
            return 'fixed'
        if entry_class.kind == 'fixed':
            return 'fixed'
        return 'shared' if entry_class.shared else 'free'

    # -- evaluation helpers -------------------------------------------------

    def _overrides(self, entry: _Entry, resolved: Mapping[ParameterRef, float]) -> dict[str, float]:
        """Graph values as canonical sasmodels keyword arguments for one entry."""
        manager = entry.fitter._param_manager
        overrides: dict[str, float] = {}
        for ref, value in resolved.items():
            if ref.dataset != entry.name:
                continue
            if ref.pd:
                overrides[f'{manager._resolve_canonical(ref.name)}_pd'] = value
            else:
                for canonical in _canonical_names(manager, ref.name):
                    overrides[canonical] = value
        return overrides

    def _evaluate_entry(
        self, entry: _Entry, target: Any, resolved: Mapping[ParameterRef, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        fit_state = entry.fitter._param_manager.snapshot_fit_state()
        pars = build_model_parameters(fit_state, self._overrides(entry, resolved))
        return evaluate_theory(target, entry.fitter.kernel, pars)

    def _entry_spec(self, entry: _Entry) -> MultiEntrySpec:
        manager = entry.fitter._param_manager
        graph = self._compiled()
        forced = tuple(
            manager._resolve_canonical(ref.name)
            for ref in graph.class_of
            if ref.dataset == entry.name and ref.pd
        )
        return MultiEntrySpec(
            name=entry.name,
            data=entry.fitter._evaluation_data(),
            kernel=entry.fitter.kernel,
            fit_state=manager.snapshot_fit_state(),
            weight=entry.weight,
            forced_pd=forced,
            canonical={name: tuple(_canonical_names(manager, name)) for name in manager.params},
            canonical_pd={name: manager._resolve_canonical(name) for name in manager.params},
        )

    # -- preflight ----------------------------------------------------------

    def _require_ready(self) -> None:
        if not self._entries:
            raise ValueError('No datasets added. Use add() first.')
        for entry in self._entries.values():
            if entry.fitter.data is None:
                raise ValueError(f"Dataset '{entry.name}' has no data.")
            if entry.fitter.kernel is None:
                raise ValueError(
                    f"Dataset '{entry.name}' has no model. "
                    f"Call fit['{entry.name}'].set_model(...) first."
                )

    def _preflight(self, graph: CompiledGraph) -> None:
        """Check the data and the graph before an optimizer is built.

        Every failure here is one that would otherwise surface as an infinite
        chi-squared, a silent reweighting, or a fit of fewer points than the
        user believes, so each is reported with the dataset that caused it.
        """
        fingerprints: dict[str, list[str]] = {}
        for entry in self._entries.values():
            data = entry.fitter.data
            index = get_fit_index(data)
            selected = int(index.sum())
            if selected == 0:
                raise ValueError(
                    f"Dataset '{entry.name}' has no points inside its Q range and mask."
                )
            dy = getattr(data, 'dy', None)
            if not has_real_data(dy):
                raise ValueError(
                    f"Dataset '{entry.name}' has no intensity uncertainties (dI). The "
                    'bumps engine cannot weight such points. Supply dI or remove the '
                    'dataset.'
                )
            values = np.asarray(dy, dtype=float)[index]
            bad = int(np.sum(~np.isfinite(values) | (values <= 0)))
            if bad:
                raise ValueError(
                    f"Dataset '{entry.name}' has {bad} of {selected} fitted points with a "
                    'non-positive or non-finite dI. Exclude them with set_q_range() or a '
                    'mask, or supply valid uncertainties.'
                )
            fingerprints.setdefault(fingerprint_arrays(data), []).append(entry.name)

        for names in fingerprints.values():
            if len(names) > 1:
                warnings.warn(
                    f'Datasets {", ".join(names)} hold identical data. Fitting the same '
                    'measurement twice does not make it two independent measurements: '
                    'the joint chi-squared counts it twice and the uncertainties will be '
                    'optimistic. This is intentional when comparing two models on one '
                    'dataset — but that comparison is better done as separate fits.',
                    stacklevel=3,
                )

        if graph.n_free == 0:
            warnings.warn(
                'No parameters are free to vary, so the fit evaluates the model once at '
                'the configured values. Set vary=True on something, or remove a '
                'constraint.',
                stacklevel=3,
            )
        self._check_degeneracies(graph)

    def _check_degeneracies(self, graph: CompiledGraph) -> None:
        """Warn about coordinates the data cannot separate.

        Extended to the graph's roots rather than each dataset's parameter list:
        sharing can hide a degeneracy (two datasets whose scales are shared with
        one another *and* with an SLD) that neither dataset shows on its own.
        """
        free_labels = set(graph.labels)
        for entry in self._entries.values():
            manager = entry.fitter._param_manager
            if not manager.get_components():
                continue
            scale = ParameterRef(entry.name, 'scale')
            if scale not in graph.class_of:
                continue
            if graph.classes[graph.class_of[scale]].label not in free_labels:
                continue
            component_scales = [
                name
                for name in manager.params
                if name.endswith('_scale')
                and name != 'scale'
                and graph.classes[graph.class_of[ParameterRef(entry.name, name)]].label
                in free_labels
            ]
            if component_scales:
                warnings.warn(
                    f"In dataset '{entry.name}' both the global 'scale' and the component "
                    f'scale(s) {", ".join(component_scales)} vary. Only their product is '
                    'fitted, so the split between them is degenerate. Fix one of them.',
                    stacklevel=4,
                )

    # -- results ------------------------------------------------------------

    def _build_result(self, graph: CompiledGraph, outcome: Any, method: str) -> MultiFitResult:
        values = graph.class_values(outcome.root_values)
        gradients = graph.gradients(outcome.root_values)
        cov = outcome.cov

        def error_of(index: int) -> tuple[float | None, str]:
            entry_class = graph.classes[index]
            if entry_class.kind in ('fixed', 'constant'):
                return 0.0, 'fixed'
            if cov is None:
                return None, outcome.cov_note or 'no covariance was available'
            gradient = gradients[index]
            variance = float(gradient @ np.asarray(cov) @ gradient)
            if not math.isfinite(variance) or variance < 0:
                return None, 'the propagated variance was not a positive number'
            source = (
                'root covariance'
                if entry_class.kind == 'free'
                else (
                    'propagated through the constraint graph'
                    if entry_class.kind == 'derived'
                    else 'shared with its root'
                )
            )
            return math.sqrt(variance), source

        parameters: dict[str, MultiParameter] = {}
        by_dataset: dict[str, dict[str, MultiParameter]] = {name: {} for name in self._entries}
        for entry_class in graph.classes:
            stderr, source = error_of(entry_class.index)
            value = values[entry_class.index]
            for ref in entry_class.members:
                parameter = MultiParameter(
                    qualified=ref.qualified,
                    dataset=ref.dataset,
                    name=f'{ref.name}_pd' if ref.pd else ref.name,
                    value=value,
                    stderr=stderr,
                    formatted=format_estimate(value, stderr),
                    status=self._status(entry_class, ref),
                    root=entry_class.label,
                    members=tuple(member.qualified for member in entry_class.members),
                    expression=entry_class.constraint_text,
                    uncertainty_source=source,
                )
                parameters[ref.qualified] = parameter
                by_dataset[ref.dataset][parameter.name] = parameter

        datasets: dict[str, DatasetResult] = {}
        for dataset_outcome in outcome.datasets:
            entry = self._entries[dataset_outcome.name]
            root = math.sqrt(dataset_outcome.weight)
            data = entry.fitter.data
            datasets[entry.name] = DatasetResult(
                name=entry.name,
                model=entry.fitter.model_name or '',
                n_points=dataset_outcome.n_points,
                chisq=dataset_outcome.chisq,
                objective_contribution=dataset_outcome.objective_contribution,
                weight=dataset_outcome.weight,
                q_range=entry.fitter.get_q_range(),
                resolution=entry.fitter._resolution.describe(),
                parameters=by_dataset[entry.name],
                # Copied, not referenced: this result must keep describing one
                # coherent fit even after the dataset is reconfigured, replaced
                # or removed. See DatasetResult.
                observed_q=np.array(data.x, dtype=float, copy=True),
                observed_intensity=np.array(data.y, dtype=float, copy=True),
                observed_uncertainty=(
                    None if data.dy is None else np.array(data.dy, dtype=float, copy=True)
                ),
                observed_dq=(
                    None
                    if getattr(data, 'dx', None) is None
                    else np.array(data.dx, dtype=float, copy=True)
                ),
                fitted_curve=dataset_outcome.fitted_curve,
                fit_index=dataset_outcome.fit_index,
                residuals=dataset_outcome.residuals,
                objective_residuals=root * dataset_outcome.residuals,
            )

        dof = outcome.n_points - graph.n_free
        on_bounds: list[tuple[str, str]] = []
        for index in graph.free_roots:
            entry_class = graph.classes[index]
            low, high = graph.ranges[index]
            value = values[index]
            if at_bound(value, low):
                on_bounds.append((entry_class.label, 'min'))
            elif at_bound(value, high):
                on_bounds.append((entry_class.label, 'max'))

        return MultiFitResult(
            engine='bumps',
            method=method,
            # bumps hard-codes success on every fitter, so there is no verdict to
            # report and inventing one would be worse than saying nothing.
            converged=None,
            message=outcome.message,
            n_datasets=len(datasets),
            n_points=outcome.n_points,
            n_free=graph.n_free,
            dof=dof,
            chisq=outcome.chisq,
            reduced_chisq=reduced_chisq(outcome.chisq, dof),
            objective=outcome.objective,
            reduced_objective=reduced_chisq(outcome.objective, dof),
            weighting={name: entry.weight for name, entry in self._entries.items()},
            constraints=self.get_constraints(),
            sharing=[
                {
                    'root': entry_class.label,
                    'members': [ref.qualified for ref in entry_class.members],
                    'status': entry_class.kind,
                }
                for entry_class in graph.classes
                if entry_class.shared
            ],
            parameters=parameters,
            cov=outcome.cov,
            cov_labels=list(outcome.cov_labels),
            cov_source=outcome.cov_source,
            cov_note=outcome.cov_note,
            on_bounds=on_bounds,
            datasets=datasets,
            fit_context=None,
        )

    def _commit(self, graph: CompiledGraph, root_values: np.ndarray) -> None:
        """Write the fitted values back into every dataset, all at once.

        The compiled graph is dropped afterwards: it carries the configured
        starting values, and leaving it cached would make every read-only view
        report the state the fit began from rather than where it ended.
        """
        for ref, value in graph.resolve(root_values).items():
            self._write_value(ref, value)
        self._invalidate()

    def _attach_component_curves(self, result: MultiFitResult) -> None:
        """Add per-component curves to a '+' mixture dataset's result.

        Run after the fitted values are committed, so the existing single-fit
        evaluator sees the parameters the joint fit landed on. A mismatched
        length is dropped rather than plotted: it would mean the component was
        evaluated on a different selection than the total, and a component curve
        that does not stack onto its total is worse than none.
        """
        for name, entry in result.datasets.items():
            fitter = self._entries[name].fitter
            curves = fitter._compute_component_curves()
            if not curves:
                continue
            expected = len(entry.fitted_curve)
            usable = {
                label: curve
                for label, curve in curves.items()
                if len(np.asarray(curve)) == expected
            }
            if len(usable) != len(curves):
                warnings.warn(
                    f"Dataset '{name}': some component curves did not match the "
                    f'{expected} fitted points and were dropped.',
                    RuntimeWarning,
                    stacklevel=4,
                )
            entry.component_curves = usable or None

    def _context(self) -> dict[str, Any]:
        """A digest of everything a result depends on, for stale detection."""
        entries = {
            entry.name: {
                'model': entry.fitter.model_name,
                'weight': entry.weight,
                'q_range': list(entry.fitter.get_q_range() or ()),
                'resolution': entry.fitter.get_resolution(),
                'data': fingerprint_arrays(entry.fitter.data),
                'config': config_digest(entry.fitter._param_manager.export_config()),
            }
            for entry in self._entries.values()
        }
        graph = {
            'sharing': [[ref.qualified for ref in group.members] for group in self._share_groups],
            'links': {
                follower.qualified: leader.qualified for follower, leader in self._directed.items()
            },
            'constraints': {ref.qualified: spec.text for ref, spec in self._constraints.items()},
        }
        payload = {'entries': entries, 'graph': graph}
        return {'digest': config_digest(payload), 'entries': entries, 'graph': graph}

    def _require_result(self) -> MultiFitResult:
        if self._result is None:
            raise ValueError('No fit result available. Run fit() first.')
        return self._result

    def _warn_if_stale(self) -> None:
        result = self._result
        if result is None or result.fit_context is None:
            return
        if result.fit_context.get('digest') != self._context()['digest']:
            warnings.warn(
                'The configuration has changed since this fit ran, so the result no '
                'longer describes the current analysis. It is still available on '
                '.result; refit to bring the two back together.',
                stacklevel=3,
            )

    @staticmethod
    def _warn_about_bounds(result: MultiFitResult) -> None:
        if not result.on_bounds:
            return
        hits = ', '.join(f'{name} ({side})' for name, side in result.on_bounds)
        warnings.warn(
            f'These parameters stopped on a bound: {hits}. The optimizer wanted to go '
            'further, so the value is a limit rather than a measurement, and its '
            'uncertainty understates the truth.',
            stacklevel=3,
        )

    # -- export helpers -----------------------------------------------------

    def _curve_text(self, entry: DatasetResult, result: MultiFitResult) -> str:
        """One dataset's measured and fitted curve, from the result's snapshot.

        Reads the observations the fit was run against rather than whatever the
        dataset holds now, so every exported row is internally consistent:
        ``Residual`` always equals ``(I_fit - I_exp) / dI_exp``.
        """
        index = entry.selection()
        q = entry.observed_q[index]
        intensity = entry.observed_intensity[index]
        uncertainty = (
            np.full(len(q), np.nan)
            if entry.observed_uncertainty is None
            else entry.observed_uncertainty[index]
        )

        header = [
            'SANS simultaneous fit',
            f'Dataset: {entry.name}',
            f'Model: {entry.model}',
            f'Points: {entry.n_points} of {len(index)}',
            f'Chi-squared (this dataset): {entry.chisq:.6f}',
            f'Objective contribution: {entry.objective_contribution:.6f}',
            f'Weight: {entry.weight:g}',
            f'Resolution: {entry.resolution}',
            'Residual convention: (model - I) / dI',
            f'Joint fit: {result.n_datasets} datasets, {result.n_free} free parameters, '
            f'dof {result.dof}',
        ]
        rows = [
            [_csv_number(value) for value in row]
            for row in zip(
                q,
                intensity,
                uncertainty,
                entry.fitted_curve,
                entry.residuals,
                entry.objective_residuals,
                strict=True,
            )
        ]
        return _csv_text(
            ['Q', 'I_exp', 'dI_exp', 'I_fit', 'Residual', 'Objective_residual'],
            rows,
            header=header,
        )

    def _manifest_text(self, result: MultiFitResult, filenames: Sequence[str]) -> str:
        """The manifest, which also records what this export owns.

        The file list is read back by the next export so it can retire artifacts
        that no longer belong — a covariance file beside a fit that has no
        covariance would otherwise be read as this fit's.
        """
        lines = [
            'SANS-fitter simultaneous fit',
            f'Engine: {result.engine}/{result.method}',
            f'Datasets: {result.n_datasets}',
            f'Points: {result.n_points}   Free parameters: {result.n_free}   dof: {result.dof}',
            f'Chi-squared: {result.chisq:.6f}   Reduced: {result.reduced_chisq:.6f}',
        ]
        if result.weighted:
            lines.append(
                f'Weighted objective: {result.objective:.6f}   '
                f'Per dof: {result.reduced_objective:.6f}'
            )
            lines.append('Dataset factors are fitting priorities; dI remains the error model.')
        lines.append(f'Covariance: {result.cov_source or "unavailable"}')
        if result.cov_note:
            lines.append(f'  {result.cov_note}')
        lines.append(f'Context digest: {(result.fit_context or {}).get("digest", "n/a")}')
        lines.append('')
        lines.append(MANIFEST_FILE_MARKER)
        lines.extend(f'  {name}' for name in filenames)
        lines.append(f'  {MANIFEST_NAME}')
        return '\n'.join(lines) + '\n'


# =========================================================================
# Module helpers
# =========================================================================


def _kernel_units(kernel: Any) -> dict[str, str]:
    """Declared units per canonical parameter name, for the sharing check.

    Read from the kernel rather than stored on the parameter manager: units are
    documentation derived from the model definition, and pinning them into saved
    configuration would make a file go stale against a sasmodels upgrade — the
    same reason ``description`` is not exported.
    """
    try:
        parameters = kernel.info.parameters.kernel_parameters
    except AttributeError:  # a kernel shape this package does not model
        return {}
    return {parameter.name: str(getattr(parameter, 'units', '') or '') for parameter in parameters}


def _canonical_names(manager: Any, name: str) -> list[str]:
    """Every canonical sasmodels slot a user-facing parameter name drives.

    One name drives several slots under ``set_models(shared=[...])``, where a
    single ``sld`` stands for ``A_sld`` and ``B_sld``.
    """
    resolved = manager.resolve_name(name)
    shared = manager._shared_to_canonicals.get(resolved)
    if shared:
        return list(shared)
    return [manager._resolve_canonical(resolved)]


def _preview_residuals(data: Any, curve: np.ndarray, fit_index: np.ndarray) -> np.ndarray | None:
    """Residuals for a preview panel, or None when the data carries no dI."""
    dy = getattr(data, 'dy', None)
    if not has_real_data(dy):
        return None
    selected = np.asarray(dy, dtype=float)[fit_index]
    if np.any(selected == 0):
        return None
    intensity = np.asarray(data.y, dtype=float)[fit_index]
    # The same sign convention the fit uses, so a preview panel and a result
    # panel can be compared without a mental flip.
    return (curve - intensity) / selected


def _csv_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f'{number:.8e}' if math.isfinite(number) else ''


def _csv_text(
    columns: Sequence[str], rows: Sequence[Sequence[str]], header: Sequence[str] = ()
) -> str:
    """Render a CSV as text, quoting whatever needs quoting.

    Through ``csv.writer`` rather than joining on commas: cells carry model
    expressions, monikers and constraint text, and a hand-rolled writer that
    substitutes commas silently corrupts the value it was trying to protect.
    Quoting preserves it instead, and ``csv.reader`` gets it back unchanged.

    *header* lines are written as ``#`` comments above the column row, which is
    the convention the single-fit export already uses.
    """
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer, lineterminator='\n')
    for line in header:
        buffer.write(f'# {line}\n')
    writer.writerow(list(columns))
    writer.writerows([list(row) for row in rows])
    return buffer.getvalue()


def _previously_exported(directory: str) -> set[str]:
    """Filenames the last export into *directory* recorded in its manifest.

    Only files this package wrote are ever considered for removal, and only
    because it said so in writing. Anything else in the directory belongs to
    the user and is left alone.
    """
    path = os.path.join(directory, MANIFEST_NAME)
    try:
        with open(path, encoding='utf-8') as handle:
            lines = handle.read().splitlines()
    except OSError:
        return set()

    if MANIFEST_FILE_MARKER not in lines:
        return set()
    listed = lines[lines.index(MANIFEST_FILE_MARKER) + 1 :]
    return {
        name
        for name in (line.strip() for line in listed)
        # A bare filename, so a path separator cannot direct a delete elsewhere.
        if name and name != MANIFEST_NAME and os.path.basename(name) == name
    }

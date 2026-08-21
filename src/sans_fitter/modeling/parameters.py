"""
Parameter Manager - Handles model parameter management for SANS fitting.

This module encapsulates all parameter-related operations including initialization,
validation, bounds management, structure factor parameter linking, polydispersity,
composite-model component metadata, the friendly-name alias layer, and generic
parameter equality links.
"""

import warnings
from typing import Any

import numpy as np

from ..results import ParameterStateSnapshot
from .polydispersity import PolydispersityManager
from .structure_factor import StructureFactorManager, default_parameter_bounds


def derive_mixture_components(kernel: Any) -> list[tuple[str, str, str]]:
    """Derive ``(prefix, moniker, part_model_name)`` triples for a mixture kernel.

    The derivation walks the kernel's composition tree and reads each part's
    actual prefix from the prefixed names in
    ``kernel.info.parameters.kernel_parameters`` — the user's expression string
    is never the prefix authority (sasmodels renumbers prefixes for nested
    mixture plugins). For flat atomic/product parts the result equals
    part-order prefixing. The moniker is set to the prefix; the alias layer
    (:meth:`ParameterManager.register_aliases`) replaces it for the
    ``set_models`` path.

    Returns an empty list for atomic (non-mixture) models.
    """
    info = getattr(kernel, 'info', None)
    composition = getattr(info, 'composition', None) if info is not None else None
    # Defensive: mock kernels or atomic models may expose anything here. Only
    # a real ('mixture', parts) tuple proceeds.
    if not isinstance(composition, (tuple, list)) or len(composition) != 2:
        return []
    if composition[0] != 'mixture':
        return []
    operation = getattr(info, 'operation', '+')
    parts = composition[1]
    try:
        kernel_names = [param.name for param in info.parameters.kernel_parameters]
    except Exception as exc:  # noqa: BLE001 - mock kernels may expose anything
        warnings.warn(
            f'Could not read kernel parameters of mixture model: {exc!r}. '
            'Treating the model as atomic — component features (aliases, '
            'monikers, component curves) are disabled.',
            stacklevel=2,
        )
        return []

    components: list[tuple[str, str, str]] = []
    try:
        cursor = 0
        for part in parts:
            prefix = ''
            if operation == '+':
                # Every '+' part contributes a leading {prefix}_scale parameter;
                # its name carries the part's actual prefix ('A_scale', or the
                # combined 'AB_scale' form for nested product-mixture plugins).
                scale_name = kernel_names[cursor] if cursor < len(kernel_names) else ''
                if scale_name.endswith('scale'):
                    prefix = scale_name[: -len('scale')].rstrip('_')
                cursor += 1
            n_part_params = len(part.parameters.kernel_parameters)
            block = kernel_names[cursor : cursor + n_part_params]
            cursor += n_part_params
            if not prefix and block:
                prefix = block[0].split('_')[0]
            part_name = getattr(part, 'name', '') or ''
            components.append((prefix, prefix, part_name))
    except Exception as exc:  # noqa: BLE001 - mock kernels may expose anything
        warnings.warn(
            f'Could not derive components of mixture model: {exc!r}. '
            'Treating the model as atomic — component features (aliases, '
            'monikers, component curves) are disabled.',
            stacklevel=2,
        )
        return []
    return components


class ParameterManager:
    """
    Manages model parameters for SANS fitting.

    Handles parameter initialization, validation, bounds management,
    special logic for structure factor parameter linking, and polydispersity support.

    Attributes:
        params: Dictionary of parameter configurations
        model_name: Name of the current model
        structure_factor_name: Name of applied structure factor (if any)
        radius_effective_mode: Mode for handling radius_effective ('unconstrained' or 'link_radius')
        polydisperse_params: Dictionary of polydispersity parameters
        pd_enabled: Whether polydispersity is globally enabled
    """

    def __init__(self):
        """Initialize the parameter manager."""
        self.params: dict[str, dict[str, Any]] = {}
        self.model_name: str | None = None
        self._sf_manager = StructureFactorManager()
        self._pd_manager = PolydispersityManager()

        # Composite-model state (see 46_COMPOSITE_MODELS.md)
        # _components: ordered (prefix, moniker, part_model_name) triples.
        self._components: list[tuple[str, str, str]] = []
        # _links: equality links, follower -> target, stored under whatever
        # names self.params uses (aliases on the set_models path, canonical
        # names on the raw set_model path).
        self._links: dict[str, str] = {}
        # Alias layer (set_models path only): alias -> canonical name. Shared
        # parameters additionally map one alias to several canonical names.
        self._alias_to_canonical: dict[str, str] = {}
        self._canonical_to_alias: dict[str, str] = {}
        self._shared_to_canonicals: dict[str, list[str]] = {}

    @property
    def _structure_factor_name(self) -> str | None:
        return self._sf_manager.name

    @_structure_factor_name.setter
    def _structure_factor_name(self, value: str | None) -> None:
        self._sf_manager.name = value

    @property
    def _radius_effective_mode(self) -> str:
        return self._sf_manager.radius_effective_mode

    @_radius_effective_mode.setter
    def _radius_effective_mode(self, value: str) -> None:
        self._sf_manager.radius_effective_mode = value

    @property
    def _form_factor_params(self) -> dict[str, dict[str, Any]]:
        return self._sf_manager.backed_up_params

    @_form_factor_params.setter
    def _form_factor_params(self, value: dict[str, dict[str, Any]]) -> None:
        self._sf_manager.backed_up_params = value

    @property
    def _polydisperse_param_names(self) -> list[str]:
        return self._pd_manager.param_names

    @_polydisperse_param_names.setter
    def _polydisperse_param_names(self, value: list[str]) -> None:
        self._pd_manager.param_names = value

    @property
    def polydisperse_params(self) -> dict[str, dict[str, Any]]:
        return self._pd_manager.params

    @polydisperse_params.setter
    def polydisperse_params(self, value: dict[str, dict[str, Any]]) -> None:
        self._pd_manager.params = value

    @property
    def _pd_enabled(self) -> bool:
        return self._pd_manager.enabled

    @_pd_enabled.setter
    def _pd_enabled(self, value: bool) -> None:
        self._pd_manager.enabled = value

    @property
    def _backed_up_pd_state(self) -> dict[str, Any] | None:
        return self._pd_manager.backup_state

    @_backed_up_pd_state.setter
    def _backed_up_pd_state(self, value: dict[str, Any] | None) -> None:
        self._pd_manager.backup_state = value

    def initialize_from_kernel(
        self,
        kernel: Any,
        model_name: str,
        components: list[tuple[str, str, str]] | None = None,
    ) -> None:
        """
        Initialize parameters from a SasModels kernel.

        Args:
            kernel: SasModels kernel object
            model_name: Name of the model
            components: Optional ordered ``(prefix, moniker, part_model_name)``
                triples for composite models. Defaults to deriving components
                from the kernel's composition tree (empty for atomic models).

        Raises:
            ValueError: If kernel is invalid
        """
        if kernel is None:
            raise ValueError('Kernel cannot be None')

        # Clear all state first to ensure clean initialization
        self.clear()

        self.model_name = model_name
        if components is None:
            components = derive_mixture_components(kernel)
        self._components = [tuple(entry) for entry in components]

        # Extract parameters from kernel
        for param in kernel.info.parameters.kernel_parameters:
            lo, hi = default_parameter_bounds(param.default, param.limits)
            self.params[param.name] = {
                'value': param.default,
                'min': lo,
                'max': hi,
                'vary': False,  # By default, parameters are fixed
                'description': param.description,
            }

            # Track polydisperse parameters
            if getattr(param, 'polydisperse', False):
                self._polydisperse_param_names.append(param.name)

        # Add implicit scale and background parameters (present in all models)
        if 'scale' not in self.params:
            self.params['scale'] = {
                'value': 1.0,
                'min': 0.0,
                'max': np.inf,
                'vary': False,
                'description': 'Scale factor for the model intensity',
            }

        if 'background' not in self.params:
            self.params['background'] = {
                'value': 0.0,
                'min': 0.0,
                'max': np.inf,
                'vary': False,
                'description': 'Constant background level',
            }

        self._initialize_polydispersity_params()

    def get_param_dict(self) -> dict[str, dict[str, Any]]:
        """
        Get the full parameter dictionary.

        Returns:
            Dictionary of parameter configurations
        """
        return self.params

    def get_param_values(self) -> dict[str, float]:
        """
        Get dictionary of parameter names to current values.

        Returns:
            Dictionary mapping parameter names to their current values
        """
        return {name: info['value'] for name, info in self.params.items()}

    def get_canonical_param_values(self) -> dict[str, float]:
        """Get current parameter values keyed by canonical sasmodels names.

        Shared parameters expand to every canonical name they drive. Used by
        post-fit evaluation (component curves) that speaks sasmodels names.
        """
        values: dict[str, float] = {}
        for name, info in self.params.items():
            canonicals = self._shared_to_canonicals.get(name)
            if canonicals:
                for canonical in canonicals:
                    values[canonical] = info['value']
            else:
                values[self._resolve_canonical(name)] = info['value']
        return values

    def snapshot_fit_state(self) -> ParameterStateSnapshot:
        """Capture a stable snapshot of parameter state for fitting engines.

        The snapshot carries **canonical sasmodels names only**: alias-keyed
        entries are emitted under their canonical names, shared parameters
        expand to their first canonical name as the link target plus equality
        links from the remaining canonical names, and ``_links`` (stored under
        user-facing names) is translated to canonical names here — the one and
        only translation site for links.
        """
        canonical_params: dict[str, dict[str, Any]] = {}
        linked_params: dict[str, str] = {}

        for name, info in self.params.items():
            canonicals = self._shared_to_canonicals.get(name)
            if canonicals:
                # Shared parameter: one user-facing entry drives several
                # canonical parameters. Emit the first as the target and link
                # the rest to it.
                target = canonicals[0]
                canonical_params[target] = dict(info)
                for follower in canonicals[1:]:
                    canonical_params[follower] = dict(info)
                    canonical_params[follower]['vary'] = False
                    linked_params[follower] = target
            else:
                canonical = self._alias_to_canonical.get(name, name)
                canonical_params[canonical] = dict(info)

        # Translate equality links (stored in user-facing names) to canonical.
        for follower, target in self._links.items():
            follower_canonicals = self._shared_to_canonicals.get(follower, [follower])
            target_canonicals = self._shared_to_canonicals.get(target, [target])
            for follower_canonical in follower_canonicals:
                follower_canonical = self._alias_to_canonical.get(
                    follower_canonical, follower_canonical
                )
                target_canonical = self._alias_to_canonical.get(
                    target_canonicals[0], target_canonicals[0]
                )
                linked_params[follower_canonical] = target_canonical
                if follower_canonical in canonical_params:
                    canonical_params[follower_canonical]['vary'] = False

        varying = [
            name
            for name, info in canonical_params.items()
            if info['vary'] and name not in linked_params
        ]

        return ParameterStateSnapshot(
            params=canonical_params,
            polydisperse_param_names=self._pd_manager.get_parameters(),
            polydisperse_params={
                name: dict(info) for name, info in self._pd_manager.params.items()
            },
            pd_enabled=self._pd_manager.is_enabled(),
            radius_effective_mode=self._radius_effective_mode,
            structure_factor_name=self._structure_factor_name,
            varying_params=varying,
            varying_pd_params=self.get_varying_pd_params(),
            linked_params=linked_params,
            components=tuple(self._components),
        )

    def apply_fitted_values(self, fitted_values: dict[str, float]) -> None:
        """Apply fitted values back into regular and PD parameter state.

        Engine results carry canonical names; they are translated back through
        the reverse alias map before write-back so a fitted ``A_sld`` lands on
        the user-facing ``sld`` entry. The reverse map is consulted *before*
        the raw ``params`` membership check: on the alias path ``params`` is
        keyed by aliases, and an alias may equal an unrelated canonical name
        (moniker shadowing a sasmodels prefix) — the canonical spelling of an
        engine result must never be mistaken for such an alias.
        """
        for name, value in fitted_values.items():
            if name in self._canonical_to_alias:
                self.set_param(self._canonical_to_alias[name], value=value)
            elif name in self.params:
                self.set_param(name, value=value)
            elif name.endswith('_pd'):
                base_param = name[:-3]
                # PD state is keyed by canonical names end-to-end; no reverse
                # translation is needed here.
                if base_param in self._pd_manager.get_parameters():
                    self.set_pd_param(base_param, pd_width=value)

        # Propagate each link target's fitted value onto its followers.
        # Followers are excluded from the varying set, so engine results never
        # contain a follower name — this post-loop propagation is the
        # follower's *only* update path. Do not remove it as apparently dead.
        for follower, target in self._links.items():
            if target in self.params and follower in self.params:
                self.params[follower]['value'] = self.params[target]['value']

    def resolve_name(self, name: str) -> str:
        """Resolve a user-facing parameter name to the key used in ``params``.

        Resolution rule: try the alias map first, fall back to canonical names
        (so ``A_sld`` always works), then raise ``KeyError`` listing the
        user-facing (alias) names.
        """
        if name in self.params:
            return name
        if name in self._alias_to_canonical:
            canonical = self._alias_to_canonical[name]
            if canonical in self.params:
                return canonical
            # Alias of a suppressed (shared) prefixed entry: not in params.
            return name
        if name in self._canonical_to_alias:
            return self._canonical_to_alias[name]
        available = ', '.join(self.params.keys())
        raise KeyError(f"Parameter '{name}' not found. Available: {available}")

    def link_params(self, name: str, to: str) -> None:
        """Create an equality link: *name* (follower) mirrors *to* (target).

        The follower is forced to ``vary=False`` and always carries the
        target's value — before, during, and after the fit.

        Raises:
            KeyError: If either name does not exist.
            ValueError: On self-links, link chains, or conflicting links.
        """
        follower = self.resolve_name(name)
        target = self.resolve_name(to)
        for resolved, original in ((follower, name), (target, to)):
            if resolved not in self.params:
                available = ', '.join(self.params.keys())
                raise KeyError(f"Parameter '{original}' not found. Available: {available}")
        if follower == target:
            raise ValueError(f"Cannot link parameter '{name}' to itself.")
        if follower in self._links:
            raise ValueError(f"Parameter '{name}' is already linked to '{self._links[follower]}'.")
        if target in self._links:
            raise ValueError(
                f"Cannot link '{name}' to '{to}': '{to}' is itself a follower. "
                'Link chains are not supported — link both followers directly '
                'to the common target.'
            )
        if follower in self._links.values():
            raise ValueError(
                f"Cannot make '{name}' a follower: it is the target of another "
                'link. Link chains are not supported — link both followers '
                'directly to the common target.'
            )
        self._links[follower] = target
        self.params[follower]['vary'] = False
        self.params[follower]['value'] = self.params[target]['value']

    def unlink_params(self, name: str) -> None:
        """Remove an equality link, restoring the follower's independence.

        Raises:
            KeyError: If the name does not exist.
            ValueError: If the parameter is not a follower.
        """
        # Accept a raw link key first so a link can always be removed, even if
        # its follower no longer resolves to a live parameter.
        follower = name if name in self._links else self.resolve_name(name)
        if follower not in self._links:
            raise ValueError(f"Parameter '{name}' is not linked to another parameter.")
        del self._links[follower]

    def get_links(self) -> dict[str, str]:
        """Return the equality links (follower -> target) in user-facing names."""
        return dict(self._links)

    def get_components(self) -> list[tuple[str, str, str]]:
        """Return composite components as (prefix, moniker, part_model_name) triples.

        Empty for atomic models.
        """
        return [tuple(entry) for entry in self._components]

    # =========================================================================
    # Alias layer (set_models path) — see 46_COMPOSITE_MODELS.md §4.2b
    # =========================================================================

    def register_aliases(
        self, components: list[tuple[str, str]], shared: 'list[str] | tuple[str, ...]'
    ) -> None:
        """Build the friendly-name alias layer over a loaded composite model.

        Args:
            components: Ordered ``(moniker, model_name)`` pairs as given to
                ``set_models``. Monikers replace the prefix-derived monikers in
                the component triples by position.
            shared: Parameter names (unprefixed) that must exist in >= 2
                components and are collapsed into a single unprefixed parameter.

        Raises:
            ValueError: If a model entry expanded to more than one kernel
                component (monikers could not map 1:1), if a shared name is
                present in fewer than 2 components, or if the generated alias
                set has collisions.

        Note:
            This method is atomic: all validation runs against local state, and
            ``self.*`` is only mutated once every check has passed. On failure
            the manager is left exactly as ``set_model`` configured it.
        """
        if len(components) != len(self._components):
            raise ValueError(
                f'{len(components)} model entries expanded to {len(self._components)} kernel '
                'components (a nested mixture expression or mixture plugin). Pass each '
                'component separately so monikers map 1:1.'
            )

        # Overlay the user's monikers onto the kernel-derived triples by
        # position — into a local list; self._components is committed last.
        new_components = [
            (prefix, moniker, part_name)
            for (prefix, _old_moniker, part_name), (moniker, _model_name) in zip(
                self._components, components, strict=False
            )
        ]

        # Longest-prefix-first so nested combined prefixes (e.g. 'AB') win.
        comps = sorted(new_components, key=lambda c: len(c[0]), reverse=True)

        alias_to_canonical: dict[str, str] = {}
        canonical_to_alias: dict[str, str] = {}
        for canonical in self.params.keys():
            matched = None
            for prefix, moniker, _part_name in comps:
                if prefix and canonical.startswith(prefix + '_'):
                    matched = (prefix, moniker)
                    break
            if matched is None:
                continue  # global parameter (scale/background) — no alias
            prefix, moniker = matched
            stripped = canonical[len(prefix) + 1 :]
            alias = f'{moniker}_{stripped}'
            if alias in alias_to_canonical and alias_to_canonical[alias] != canonical:
                raise ValueError(
                    f"Component monikers produce a colliding parameter name '{alias}' "
                    f"(from both '{alias_to_canonical[alias]}' and '{canonical}'). "
                    'Choose distinct monikers.'
                )
            alias_to_canonical[alias] = canonical

        # Reject aliases that shadow an unrelated canonical name (e.g.
        # set_models(B='sphere', A='cylinder') maps prefix A -> moniker "B",
        # so the alias 'B_radius' equals the cylinder's canonical name).
        # Such shadowing makes name resolution ambiguous and would cross-wire
        # fitted values between components.
        for alias, canonical in alias_to_canonical.items():
            if alias in self.params and alias != canonical:
                raise ValueError(
                    f"Component moniker produces the alias '{alias}', which shadows "
                    f"the canonical parameter '{alias}' of another component "
                    f"(the alias maps to '{canonical}'). Choose monikers that do "
                    "not reuse sasmodels' A/B/C prefix letters in a different order."
                )

        # Shared parameters: one-to-many, must exist in >= 2 components.
        shared_to_canonicals: dict[str, list[str]] = {}
        for shared_name in shared:
            canonicals = []
            for prefix, _moniker, _part_name in new_components:
                candidate = f'{prefix}_{shared_name}' if prefix else shared_name
                if candidate in self.params:
                    canonicals.append(candidate)
            if len(canonicals) < 2:
                per_component = '; '.join(
                    f'{moniker}: '
                    + ', '.join(
                        name[len(prefix) + 1 :]
                        for name in self.params
                        if prefix and name.startswith(prefix + '_')
                    )
                    for prefix, moniker, _part in new_components
                )
                raise ValueError(
                    f"Shared parameter '{shared_name}' must exist in at least 2 "
                    f'components (found in {len(canonicals)}). '
                    f'Per-component parameters — {per_component}'
                )
            shared_to_canonicals[shared_name] = canonicals
            # The shared canonical names reverse-map to the shared alias.
            for canonical in canonicals:
                canonical_to_alias[canonical] = shared_name

        # Non-shared canonical names reverse-map to their prefixed alias.
        for alias, canonical in alias_to_canonical.items():
            if canonical not in canonical_to_alias:
                canonical_to_alias[canonical] = alias

        # Re-key the user-facing params dict to alias names.
        new_params: dict[str, dict[str, Any]] = {}
        shared_canonical_set = {
            canonical for canonicals in shared_to_canonicals.values() for canonical in canonicals
        }
        for canonical, info in self.params.items():
            if canonical not in alias_to_canonical.values():
                new_params[canonical] = info  # global scale/background
                continue
            if canonical in shared_canonical_set:
                # The prefixed alias of a shared parameter is suppressed: only
                # the shared name appears in the user-facing params dict.
                shared_name = canonical_to_alias[canonical]
                if shared_to_canonicals[shared_name][0] == canonical:
                    new_params[shared_name] = info
            else:
                new_params[canonical_to_alias[canonical]] = info

        self._components = new_components
        self.params = new_params
        self._alias_to_canonical = alias_to_canonical
        self._canonical_to_alias = canonical_to_alias
        self._shared_to_canonicals = shared_to_canonicals

    def _resolve_canonical(self, name: str) -> str:
        """Map an alias (or canonical) name to its canonical sasmodels name."""
        return self._alias_to_canonical.get(name, name)

    def to_display_name(self, canonical_name: str) -> str:
        """Translate a canonical sasmodels name to its user-facing name.

        Used for engine results, saved CSVs, and plot labels on the
        ``set_models`` path. On the raw ``set_model`` path the alias map is
        empty and names pass through unchanged.
        """
        return self._canonical_to_alias.get(canonical_name, canonical_name)

    def set_param(
        self,
        name: str,
        value: float | None = None,
        min: float | None = None,
        max: float | None = None,
        vary: bool | None = None,
    ) -> None:
        """
        Configure a model parameter.

        Args:
            name: Parameter name (alias or canonical)
            value: Initial value (optional)
            min: Minimum bound (optional)
            max: Maximum bound (optional)
            vary: Whether to vary during fit (optional)

        Raises:
            KeyError: If parameter name doesn't exist
            ValueError: If the parameter is a link follower and value/vary is
                written, or if vary=True is requested for a follower.
        """
        resolved = self.resolve_name(name)
        if resolved not in self.params:
            available = ', '.join(self.params.keys())
            raise KeyError(f"Parameter '{name}' not found. Available: {available}")

        if resolved in self._links:
            target = self._links[resolved]
            if value is not None or vary is True:
                raise ValueError(
                    f"Parameter '{name}' is linked to '{target}' and cannot be "
                    'set directly. Configure the target, or unlink_params() first.'
                )

        if value is not None:
            self.params[resolved]['value'] = value
            # Sync radius_effective when radius is updated in link_radius mode
            if (
                resolved == 'radius'
                and self._radius_effective_mode == 'link_radius'
                and 'radius_effective' in self.params
            ):
                self.params['radius_effective']['value'] = value
            # Propagate to equality-link followers of this parameter.
            for follower, link_target in self._links.items():
                if link_target == resolved:
                    self.params[follower]['value'] = value
        if min is not None:
            self.params[resolved]['min'] = min
        if max is not None:
            self.params[resolved]['max'] = max
        if vary is not None:
            self.params[resolved]['vary'] = vary

    def validate_param(self, name: str) -> bool:
        """
        Check if a parameter name exists.

        Args:
            name: Parameter name to validate

        Returns:
            True if parameter exists, False otherwise
        """
        return name in self.params

    def display_params(self) -> None:
        """Display current parameter values and settings in a readable format.

        For composite models the parameters are grouped: global parameters
        first, then shared parameters (from ``shared=``), then one block per
        component moniker.
        """
        if not self.params:
            print('No parameters available.')
            return

        print(f'\n{"=" * 80}')
        print(f'Model: {self.model_name}')
        if self._structure_factor_name:
            print(f'Structure Factor: {self._structure_factor_name}')
            print(f'Radius Effective Mode: {self._radius_effective_mode}')
        print(f'{"=" * 80}')

        if not self._components:
            self._print_param_table(self.params)
            print(f'{"=" * 80}\n')
            return

        global_names = {'scale', 'background'}
        shared_names = set(self._shared_to_canonicals.keys())

        def entry_line(name: str, info: dict[str, Any]) -> str:
            vary_str = '✓' if info['vary'] else '✗'
            if name == 'radius_effective' and self._radius_effective_mode == 'link_radius':
                vary_str = '→radius'
            if name in self._links:
                vary_str = f'→{self._links[name]}'
            return (
                f'{name:<28} {info["value"]:<12.4g} {info["min"]:<12.4g} '
                f'{info["max"]:<12.4g} {vary_str:<8}'
            )

        header = f'{"Parameter":<28} {"Value":<12} {"Min":<12} {"Max":<12} {"Vary":<8}'
        print(header)
        print(f'{"-" * 80}')

        print('Global:')
        for name in ('scale', 'background'):
            if name in self.params:
                print('  ' + entry_line(name, self.params[name]))
        if shared_names:
            print('Shared:')
            for name in sorted(shared_names):
                if name in self.params:
                    print('  ' + entry_line(name, self.params[name]))
        # Assign each parameter to the component with the longest matching
        # moniker prefix, so 'sphere_big_radius' files under a 'sphere_big'
        # moniker rather than also matching a plain 'sphere' moniker.
        monikers_by_length = sorted(
            (moniker for _prefix, moniker, _part in self._components), key=len, reverse=True
        )

        def owning_moniker(name: str) -> str | None:
            for candidate in monikers_by_length:
                if name.startswith(f'{candidate}_'):
                    return candidate
            return None

        for _prefix, moniker, part_name in self._components:
            label = moniker if moniker == part_name else f'{moniker} ({part_name})'
            print(f'{label}:')
            for name, info in self.params.items():
                if name in global_names or name in shared_names:
                    continue
                # Moniker alone covers both paths: params are keyed by alias
                # after register_aliases, and moniker == prefix on the raw
                # set_model path. Also matching the prefix would misfile
                # params when one component's moniker equals another's prefix.
                if owning_moniker(name) == moniker:
                    print('  ' + entry_line(name, info))
        print(f'{"=" * 80}\n')

    def _print_param_table(self, params: dict[str, dict[str, Any]]) -> None:
        """Print a flat parameter table (atomic models)."""
        print(f'{"Parameter":<20} {"Value":<12} {"Min":<12} {"Max":<12} {"Vary":<8}')
        print(f'{"-" * 80}')
        for name, info in params.items():
            vary_str = '✓' if info['vary'] else '✗'
            # Show linked indicator for radius_effective in link_radius mode
            if name == 'radius_effective' and self._radius_effective_mode == 'link_radius':
                vary_str = '→radius'
            if name in self._links:
                vary_str = f'→{self._links[name]}'
            print(
                f'{name:<20} {info["value"]:<12.4g} {info["min"]:<12.4g} '
                f'{info["max"]:<12.4g} {vary_str:<8}'
            )

    def backup_params(self) -> None:
        """Backup current parameters (used before applying structure factor)."""
        self._sf_manager.backup_params(self.params)

    def restore_params(self) -> None:
        """Restore backed up parameters (used when removing structure factor)."""
        if self._sf_manager.has_backup():
            self.params = self._sf_manager.restore_params()

    def has_backed_up_params(self) -> bool:
        """
        Check if there are backed up parameters.

        Returns:
            True if parameters have been backed up, False otherwise
        """
        return self._sf_manager.has_backup()

    def get_backed_up_params(self) -> dict[str, dict[str, Any]]:
        """
        Get the backed up form factor parameters.

        Returns:
            Dictionary of backed up parameters
        """
        return self._sf_manager.backed_up_params

    def update_for_product_model(
        self, kernel: Any, structure_factor_name: str, radius_effective_mode: str = 'unconstrained'
    ) -> None:
        """
        Update parameters for a product model (form factor @ structure factor).

        Args:
            kernel: New product model kernel
            structure_factor_name: Name of the structure factor
            radius_effective_mode: How to handle radius_effective
                - 'unconstrained': radius_effective is a separate parameter
                - 'link_radius': radius_effective is linked to radius

        Raises:
            ValueError: If radius_effective_mode is invalid
        """
        # Backup polydispersity state if not already done
        if not self._backed_up_pd_state:
            self.backup_pd_state()
        self.params = self._sf_manager.apply(
            kernel=kernel,
            sf_name=structure_factor_name,
            re_mode=radius_effective_mode,
            current_params=self.params,
        )
        self._prune_stale_links()

    def remove_structure_factor(self) -> str:
        """
        Remove structure factor and restore form factor parameters.

        Returns:
            Name of the removed structure factor

        Raises:
            ValueError: If no structure factor is currently set
        """
        sf_name, restored_params = self._sf_manager.remove()
        self.params = restored_params
        self.restore_pd_state()
        self._prune_stale_links()
        return sf_name

    def _prune_stale_links(self) -> None:
        """Drop equality links whose follower or target left ``params``.

        Called after every params rebuild (applying/removing a structure
        factor). Without this, a stale link survives as a phantom entry that
        cannot be unlinked and keeps blocking link-free engines.
        """
        stale = [
            follower
            for follower, target in self._links.items()
            if follower not in self.params or target not in self.params
        ]
        for follower in stale:
            target = self._links.pop(follower)
            warnings.warn(
                f"Removed parameter link '{follower}' -> '{target}': one of the "
                'parameters no longer exists after the model change.',
                stacklevel=3,
            )

    def get_structure_factor(self) -> str | None:
        """
        Get the name of the currently applied structure factor.

        Returns:
            Name of the structure factor, or None if no structure factor is set
        """
        return self._sf_manager.name

    def get_radius_effective_mode(self) -> str:
        """
        Get the current radius_effective mode.

        Returns:
            Current radius_effective mode ('unconstrained' or 'link_radius')
        """
        return self._sf_manager.radius_effective_mode

    def update_param_value(self, name: str, value: float) -> None:
        """
        Update a parameter's value.

        Args:
            name: Parameter name
            value: New value

        Raises:
            KeyError: If parameter doesn't exist
        """
        if name not in self.params:
            raise KeyError(f"Parameter '{name}' not found")
        self.params[name]['value'] = value

    def get_varying_params(self) -> list[str]:
        """
        Get list of parameter names that are set to vary.

        Returns:
            List of parameter names with vary=True
        """
        return [name for name, info in self.params.items() if info['vary']]

    # =========================================================================
    # Polydispersity Methods
    # =========================================================================

    def _initialize_polydispersity_params(self) -> None:
        """Initialize polydispersity parameters for all polydisperse parameters."""
        self._pd_manager.initialize(self._polydisperse_param_names)

    def get_polydisperse_parameters(self) -> list[str]:
        """
        Return list of parameter names that support polydispersity.

        Returns:
            List of parameter names that can have polydispersity applied
        """
        return self._pd_manager.get_parameters()

    def has_polydisperse_parameters(self) -> bool:
        """
        Check if the current model has any polydisperse parameters.

        Returns:
            True if model has polydisperse parameters, False otherwise
        """
        return self._pd_manager.has_parameters()

    def set_pd_param(
        self,
        base_param: str,
        pd_width: float | None = None,
        pd_n: int | None = None,
        pd_nsigma: float | None = None,
        pd_type: str | None = None,
        vary: bool | None = None,
    ) -> None:
        """
        Configure polydispersity for a specific parameter.

        Args:
            base_param: Name of the base parameter (e.g., 'radius')
            pd_width: Polydispersity width (relative, 0.0 = monodisperse)
            pd_n: Number of Gaussian quadrature points (default: 35)
            pd_nsigma: Number of sigmas to include (default: 3.0)
            pd_type: Distribution type ('gaussian', 'rectangle', 'lognormal', 'schulz', 'boltzmann')
            vary: Whether to vary the pd_width during fitting

        Raises:
            KeyError: If base_param is not a polydisperse parameter
            ValueError: If pd_type is not a valid distribution type
        """
        # Polydispersity state is keyed by canonical names end-to-end; resolve
        # aliases (e.g. 'small_radius' -> 'A_radius') before delegating.
        base_param = self._resolve_canonical(base_param)
        self._pd_manager.set_param(
            base_param,
            pd_width=pd_width,
            pd_n=pd_n,
            pd_nsigma=pd_nsigma,
            pd_type=pd_type,
            vary=vary,
        )

    def get_pd_param(self, base_param: str) -> dict[str, Any]:
        """
        Get polydispersity configuration for a specific parameter.

        Args:
            base_param: Name of the base parameter (e.g., 'radius')

        Returns:
            Dictionary with pd, pd_n, pd_nsigma, pd_type, vary, and active values.
            'active' indicates whether polydispersity is active for this parameter (pd > 0).

        Raises:
            KeyError: If base_param is not a polydisperse parameter
        """
        base_param = self._resolve_canonical(base_param)
        return self._pd_manager.get_param(base_param)

    def toggle_pd_visibility(self, enabled: bool) -> None:
        """
        Enable/disable polydispersity globally.

        When disabled, polydispersity parameters are excluded from fitting
        but their values are preserved for when PD is re-enabled.

        Args:
            enabled: Whether polydispersity should be enabled
        """
        self._pd_manager.set_enabled(enabled)

    def is_pd_enabled(self) -> bool:
        """
        Check if polydispersity is globally enabled.

        Returns:
            True if polydispersity is enabled, False otherwise
        """
        return self._pd_manager.is_enabled()

    def get_pd_params_for_fitting(self) -> dict[str, Any]:
        """
        Return polydispersity parameters to include in fitting.

        Only returns PD parameters when pd_enabled is True.
        Returns parameters in the format expected by SasModels:
        - {param}_pd: polydispersity width
        - {param}_pd_n: number of quadrature points
        - {param}_pd_nsigma: number of sigmas
        - {param}_pd_type: distribution type

        Returns:
            Dictionary of PD parameters ready for fitting
        """
        return self._pd_manager.get_fitting_params()

    def get_varying_pd_params(self) -> list[str]:
        """
        Get list of polydispersity parameter names set to vary.

        Only returns parameters when pd_enabled is True.

        Returns:
            List of base parameter names whose PD width should vary
        """
        return self._pd_manager.get_varying_params()

    def display_pd_params(self) -> None:
        """Display polydispersity parameter values and settings.

        On the ``set_models`` path the canonical prefixed names are translated
        to their user-facing aliases for display.
        """
        self._pd_manager.display(name_map=self._canonical_to_alias or None)

    def backup_pd_state(self) -> None:
        """Backup current polydispersity state (used before applying structure factor)."""
        self._pd_manager.backup()

    def restore_pd_state(self) -> None:
        """Restore backed up polydispersity state (used when removing structure factor)."""
        self._pd_manager.restore()

    def has_backed_up_pd_state(self) -> bool:
        """
        Check if there is backed up polydispersity state.

        Returns:
            True if polydispersity state has been backed up, False otherwise
        """
        return self._pd_manager.has_backup()

    def clear(self) -> None:
        """Clear all parameters and reset state."""
        self.params = {}
        self.model_name = None
        self._sf_manager.clear()

        # Reset composite-model state
        self._components = []
        self._links = {}
        self._alias_to_canonical = {}
        self._canonical_to_alias = {}
        self._shared_to_canonicals = {}

        # Reset polydispersity state
        self._pd_manager.clear()

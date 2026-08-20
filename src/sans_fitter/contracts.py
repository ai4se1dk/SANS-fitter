from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ParameterStateSnapshot:
    """Read-only parameter state passed into fitting engines.

    All parameter names carried by the snapshot are **canonical sasmodels
    names** (e.g. ``A_sld`` under a composite model). Any user-facing alias
    layer is translated away before the snapshot is built.
    """

    params: dict[str, dict[str, Any]]
    polydisperse_param_names: list[str]
    polydisperse_params: dict[str, dict[str, Any]]
    pd_enabled: bool
    radius_effective_mode: str
    structure_factor_name: str | None
    varying_params: list[str]
    varying_pd_params: list[str]
    # Equality links (follower -> target), canonical names. Populated by
    # link_params() and by the shared= sugar of set_models().
    linked_params: dict[str, str] = field(default_factory=dict)
    # Composite-model components as (prefix, moniker, part_model_name)
    # triples; empty for atomic models.
    components: tuple[tuple[str, str, str], ...] = ()

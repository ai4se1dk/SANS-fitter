from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ParameterStateSnapshot:
    """Read-only parameter state passed into fitting engines."""

    params: dict[str, dict[str, Any]]
    polydisperse_param_names: list[str]
    polydisperse_params: dict[str, dict[str, Any]]
    pd_enabled: bool
    radius_effective_mode: str
    structure_factor_name: str | None
    varying_params: list[str]
    varying_pd_params: list[str]

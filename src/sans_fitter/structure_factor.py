import warnings
from typing import Any

import numpy as np


def default_parameter_bounds(default: float, limits: tuple[float, float]) -> tuple[float, float]:
    """Finite, sign-permissive fallback bounds for a kernel parameter.

    Finite declared limits are kept as-is. Each infinite side of a
    ``(-inf, inf)``-style declaration is replaced by a range derived from the
    default value: ``default ± max(10·|default|, 1.0)``, with the lower side
    additionally capped at 0 so SLD-like parameters can go negative. This
    guarantees a non-degenerate range even for zero-default parameters.

    Note: this is a deliberate policy — parameters whose old fallback produced
    ``[0, 10·default]`` now get sign-permissive ranges; users who relied on
    the implicit positivity should set explicit bounds via ``set_param``.
    """
    lo, hi = limits
    if not np.isfinite(lo):
        lo = min(0.0, default - max(10 * abs(default), 1.0))
    if not np.isfinite(hi):
        hi = default + max(10 * abs(default), 1.0)
    return lo, hi


class StructureFactorManager:
    """Manage structure factor state and form-factor parameter backups."""

    def __init__(self) -> None:
        self._name: str | None = None
        self._radius_effective_mode = 'unconstrained'
        self._form_factor_params: dict[str, dict[str, Any]] = {}

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, value: str | None) -> None:
        self._name = value

    @property
    def radius_effective_mode(self) -> str:
        return self._radius_effective_mode

    @radius_effective_mode.setter
    def radius_effective_mode(self, value: str) -> None:
        self._radius_effective_mode = value

    @property
    def backed_up_params(self) -> dict[str, dict[str, Any]]:
        return self._form_factor_params

    @backed_up_params.setter
    def backed_up_params(self, value: dict[str, dict[str, Any]]) -> None:
        self._form_factor_params = {name: dict(info) for name, info in value.items()}

    def backup_params(self, current_params: dict[str, dict[str, Any]]) -> None:
        self._form_factor_params = {name: dict(info) for name, info in current_params.items()}

    def restore_params(self) -> dict[str, dict[str, Any]]:
        restored = {name: dict(info) for name, info in self._form_factor_params.items()}
        self._form_factor_params = {}
        return restored

    def has_backup(self) -> bool:
        return bool(self._form_factor_params)

    def apply(
        self,
        kernel: Any,
        sf_name: str,
        re_mode: str,
        current_params: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        if re_mode not in ['unconstrained', 'link_radius']:
            raise ValueError(
                f"Invalid radius_effective_mode '{re_mode}'. Use 'unconstrained' or 'link_radius'."
            )

        if not self._form_factor_params:
            self.backup_params(current_params)

        self._name = sf_name
        self._radius_effective_mode = re_mode

        new_params: dict[str, dict[str, Any]] = {}
        for param in kernel.info.parameters.kernel_parameters:
            if param.name in self._form_factor_params:
                new_params[param.name] = dict(self._form_factor_params[param.name])
            else:
                lo, hi = default_parameter_bounds(param.default, param.limits)
                new_params[param.name] = {
                    'value': param.default,
                    'min': lo,
                    'max': hi,
                    'vary': False,
                    'description': param.description,
                }

        if 'scale' not in new_params:
            if 'scale' in self._form_factor_params:
                new_params['scale'] = dict(self._form_factor_params['scale'])
            else:
                new_params['scale'] = {
                    'value': 1.0,
                    'min': 0.0,
                    'max': np.inf,
                    'vary': False,
                    'description': 'Scale factor for the model intensity',
                }

        if 'background' not in new_params:
            if 'background' in self._form_factor_params:
                new_params['background'] = dict(self._form_factor_params['background'])
            else:
                new_params['background'] = {
                    'value': 0.0,
                    'min': 0.0,
                    'max': np.inf,
                    'vary': False,
                    'description': 'Constant background level',
                }

        if re_mode == 'link_radius':
            if 'radius' in new_params and 'radius_effective' in new_params:
                new_params['radius_effective']['value'] = new_params['radius']['value']
                new_params['radius_effective']['vary'] = False
            else:
                warnings.warn(
                    'Cannot link radius_effective to radius: one or both parameters not found. Using unconstrained mode.',
                    stacklevel=3,
                )
                self._radius_effective_mode = 'unconstrained'

        return new_params

    def remove(self) -> tuple[str, dict[str, dict[str, Any]]]:
        if self._name is None:
            raise ValueError('No structure factor is currently set.')

        sf_name = self._name
        restored_params = self.restore_params()
        self._name = None
        self._radius_effective_mode = 'unconstrained'
        return sf_name, restored_params

    def clear(self) -> None:
        self._name = None
        self._radius_effective_mode = 'unconstrained'
        self._form_factor_params = {}

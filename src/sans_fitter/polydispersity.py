from typing import Any, Optional

PD_DEFAULTS = {
    'pd': 0.0,
    'pd_n': 35,
    'pd_nsigma': 3.0,
    'pd_type': 'gaussian',
    'vary': False,
}

PD_DISTRIBUTION_TYPES = ['gaussian', 'rectangle', 'lognormal', 'schulz', 'boltzmann']


class PolydispersityManager:
    """Manage polydispersity configuration and backup/restore state."""

    def __init__(self) -> None:
        self._param_names: list[str] = []
        self._params: dict[str, dict[str, Any]] = {}
        self._enabled = False
        self._backup: Optional[dict[str, Any]] = None

    @property
    def param_names(self) -> list[str]:
        return self._param_names

    @param_names.setter
    def param_names(self, value: list[str]) -> None:
        self._param_names = list(value)

    @property
    def params(self) -> dict[str, dict[str, Any]]:
        return self._params

    @params.setter
    def params(self, value: dict[str, dict[str, Any]]) -> None:
        self._params = {name: dict(info) for name, info in value.items()}

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def backup_state(self) -> Optional[dict[str, Any]]:
        return self._backup

    @backup_state.setter
    def backup_state(self, value: Optional[dict[str, Any]]) -> None:
        self._backup = value

    def initialize(self, param_names: list[str]) -> None:
        self._param_names = list(param_names)
        self._params = {
            param_name: {
                'pd': PD_DEFAULTS['pd'],
                'pd_n': PD_DEFAULTS['pd_n'],
                'pd_nsigma': PD_DEFAULTS['pd_nsigma'],
                'pd_type': PD_DEFAULTS['pd_type'],
                'vary': PD_DEFAULTS['vary'],
            }
            for param_name in self._param_names
        }
        self._enabled = False

    def get_parameters(self) -> list[str]:
        return list(self._param_names)

    def has_parameters(self) -> bool:
        return len(self._param_names) > 0

    def set_param(
        self,
        base_param: str,
        pd_width: Optional[float] = None,
        pd_n: Optional[int] = None,
        pd_nsigma: Optional[float] = None,
        pd_type: Optional[str] = None,
        vary: Optional[bool] = None,
    ) -> None:
        if base_param not in self._param_names:
            available = ', '.join(self._param_names)
            raise KeyError(
                f"Parameter '{base_param}' does not support polydispersity. "
                f'Available polydisperse parameters: {available}'
            )

        if pd_type is not None and pd_type not in PD_DISTRIBUTION_TYPES:
            raise ValueError(
                f"Invalid pd_type '{pd_type}'. Valid types: {', '.join(PD_DISTRIBUTION_TYPES)}"
            )

        if pd_width is not None and pd_width < 0:
            raise ValueError('pd_width must be non-negative')
        if pd_n is not None and pd_n <= 0:
            raise ValueError('pd_n must be positive')
        if pd_nsigma is not None and pd_nsigma <= 0:
            raise ValueError('pd_nsigma must be positive')

        if pd_width is not None:
            self._params[base_param]['pd'] = pd_width
        if pd_n is not None:
            self._params[base_param]['pd_n'] = pd_n
        if pd_nsigma is not None:
            self._params[base_param]['pd_nsigma'] = pd_nsigma
        if pd_type is not None:
            self._params[base_param]['pd_type'] = pd_type
        if vary is not None:
            self._params[base_param]['vary'] = vary

    def get_param(self, base_param: str) -> dict[str, Any]:
        if base_param not in self._param_names:
            available = ', '.join(self._param_names)
            raise KeyError(
                f"Parameter '{base_param}' does not support polydispersity. "
                f'Available polydisperse parameters: {available}'
            )

        pd_config = self._params[base_param].copy()
        pd_config['active'] = pd_config['pd'] > 0
        return pd_config

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def is_enabled(self) -> bool:
        return self._enabled

    def get_fitting_params(self) -> dict[str, Any]:
        if not self._enabled:
            return {}

        pd_params = {}
        for param_name in self._param_names:
            pd_config = self._params[param_name]
            pd_params[f'{param_name}_pd'] = pd_config['pd']
            pd_params[f'{param_name}_pd_n'] = pd_config['pd_n']
            pd_params[f'{param_name}_pd_nsigma'] = pd_config['pd_nsigma']
            pd_params[f'{param_name}_pd_type'] = pd_config['pd_type']
        return pd_params

    def get_varying_params(self) -> list[str]:
        if not self._enabled:
            return []

        return [
            param_name
            for param_name, pd_config in self._params.items()
            if pd_config.get('vary', False)
        ]

    def display(self) -> None:
        if not self._param_names:
            print('No polydisperse parameters available for this model.')
            return

        status = 'ENABLED' if self._enabled else 'DISABLED'
        print(f'\n{"=" * 90}')
        print(f'Polydispersity Status: {status}')
        print(f'{"=" * 90}')
        print(
            f'{"Parameter":<15} {"Width":<10} {"N Points":<10} {"N Sigma":<10} {"Type":<12} {"Vary":<8}'
        )
        print(f'{"-" * 90}')

        for param_name in self._param_names:
            pd_config = self._params[param_name]
            vary_str = '✓' if pd_config.get('vary', False) else '✗'
            print(
                f'{param_name:<15} {pd_config["pd"]:<10.4g} {pd_config["pd_n"]:<10} '
                f'{pd_config["pd_nsigma"]:<10.4g} {pd_config["pd_type"]:<12} {vary_str:<8}'
            )
        print(f'{"=" * 90}\n')

    def backup(self) -> None:
        self._backup = {
            'polydisperse_param_names': list(self._param_names),
            'polydisperse_params': {k: dict(v) for k, v in self._params.items()},
            'pd_enabled': self._enabled,
        }

    def restore(self) -> None:
        if self._backup:
            self._param_names = self._backup['polydisperse_param_names']
            self._params = {k: dict(v) for k, v in self._backup['polydisperse_params'].items()}
            self._enabled = self._backup['pd_enabled']
            self._backup = None

    def has_backup(self) -> bool:
        return self._backup is not None

    def clear(self) -> None:
        self._param_names = []
        self._params = {}
        self._enabled = False
        self._backup = None

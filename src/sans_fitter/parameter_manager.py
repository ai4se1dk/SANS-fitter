"""
Parameter Manager - Handles model parameter management for SANS fitting.

This module encapsulates all parameter-related operations including initialization,
validation, bounds management, structure factor parameter linking, and polydispersity.
"""

from typing import Any

import numpy as np

from .contracts import ParameterStateSnapshot
from .polydispersity import PolydispersityManager
from .structure_factor import StructureFactorManager


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

    def initialize_from_kernel(self, kernel: Any, model_name: str) -> None:
        """
        Initialize parameters from a SasModels kernel.

        Args:
            kernel: SasModels kernel object
            model_name: Name of the model

        Raises:
            ValueError: If kernel is invalid
        """
        if kernel is None:
            raise ValueError('Kernel cannot be None')

        # Clear all state first to ensure clean initialization
        self.clear()

        self.model_name = model_name

        # Extract parameters from kernel
        for param in kernel.info.parameters.kernel_parameters:
            self.params[param.name] = {
                'value': param.default,
                'min': param.limits[0] if param.limits[0] > -np.inf else 0,
                'max': param.limits[1] if param.limits[1] < np.inf else param.default * 10,
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

    def snapshot_fit_state(self) -> ParameterStateSnapshot:
        """Capture a stable snapshot of parameter state for fitting engines."""
        return ParameterStateSnapshot(
            params={name: dict(info) for name, info in self.params.items()},
            polydisperse_param_names=self._pd_manager.get_parameters(),
            polydisperse_params={
                name: dict(info) for name, info in self._pd_manager.params.items()
            },
            pd_enabled=self._pd_manager.is_enabled(),
            radius_effective_mode=self._radius_effective_mode,
            structure_factor_name=self._structure_factor_name,
            varying_params=self.get_varying_params(),
            varying_pd_params=self.get_varying_pd_params(),
        )

    def apply_fitted_values(self, fitted_values: dict[str, float]) -> None:
        """Apply fitted values back into regular and PD parameter state."""
        for name, value in fitted_values.items():
            if name in self.params:
                self.set_param(name, value=value)
            elif name.endswith('_pd'):
                base_param = name[:-3]
                if base_param in self._pd_manager.get_parameters():
                    self.set_pd_param(base_param, pd_width=value)

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
            name: Parameter name
            value: Initial value (optional)
            min: Minimum bound (optional)
            max: Maximum bound (optional)
            vary: Whether to vary during fit (optional)

        Raises:
            KeyError: If parameter name doesn't exist
        """
        if name not in self.params:
            available = ', '.join(self.params.keys())
            raise KeyError(f"Parameter '{name}' not found. Available: {available}")

        if value is not None:
            self.params[name]['value'] = value
            # Sync radius_effective when radius is updated in link_radius mode
            if (
                name == 'radius'
                and self._radius_effective_mode == 'link_radius'
                and 'radius_effective' in self.params
            ):
                self.params['radius_effective']['value'] = value
        if min is not None:
            self.params[name]['min'] = min
        if max is not None:
            self.params[name]['max'] = max
        if vary is not None:
            self.params[name]['vary'] = vary

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
        """Display current parameter values and settings in a readable format."""
        if not self.params:
            print('No parameters available.')
            return

        print(f'\n{"=" * 80}')
        print(f'Model: {self.model_name}')
        if self._structure_factor_name:
            print(f'Structure Factor: {self._structure_factor_name}')
            print(f'Radius Effective Mode: {self._radius_effective_mode}')
        print(f'{"=" * 80}')
        print(f'{"Parameter":<20} {"Value":<12} {"Min":<12} {"Max":<12} {"Vary":<8}')
        print(f'{"-" * 80}')

        for name, info in self.params.items():
            vary_str = '✓' if info['vary'] else '✗'
            # Show linked indicator for radius_effective in link_radius mode
            if name == 'radius_effective' and self._radius_effective_mode == 'link_radius':
                vary_str = '→radius'
            print(
                f'{name:<20} {info["value"]:<12.4g} {info["min"]:<12.4g} '
                f'{info["max"]:<12.4g} {vary_str:<8}'
            )
        print(f'{"=" * 80}\n')

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
        return sf_name

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
        """Display polydispersity parameter values and settings."""
        self._pd_manager.display()

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

        # Reset polydispersity state
        self._pd_manager.clear()

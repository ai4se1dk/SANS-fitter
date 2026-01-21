"""
Parameter Manager - Handles model parameter management for SANS fitting.

This module encapsulates all parameter-related operations including initialization,
validation, bounds management, structure factor parameter linking, and polydispersity.
"""

from typing import Any, Optional

import numpy as np

# Default values for polydispersity parameters
PD_DEFAULTS = {
    'pd': 0.0,  # No polydispersity by default
    'pd_n': 35,  # Number of Gaussian quadrature points
    'pd_nsigma': 3.0,  # Number of sigmas to include
    'pd_type': 'gaussian',  # Distribution type
}

# Available polydispersity distribution types
PD_DISTRIBUTION_TYPES = ['gaussian', 'rectangle', 'lognormal', 'schulz', 'boltzmann']


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
        self.model_name: Optional[str] = None
        self._structure_factor_name: Optional[str] = None
        self._radius_effective_mode: str = 'unconstrained'
        self._form_factor_params: dict[str, dict[str, Any]] = {}

        # Polydispersity support
        self._polydisperse_param_names: list[str] = []  # List of params that support PD
        self.polydisperse_params: dict[str, dict[str, Any]] = {}  # PD parameter values
        self._pd_enabled: bool = False  # Global PD visibility toggle
        self._pd_param_states: dict[str, bool] = {}  # Individual param vary states

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

        self.model_name = model_name
        self.params = {}
        self._polydisperse_param_names = []

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

        # Initialize polydispersity parameters
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

    def set_param(
        self,
        name: str,
        value: Optional[float] = None,
        min: Optional[float] = None,
        max: Optional[float] = None,
        vary: Optional[bool] = None,
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
        self._form_factor_params = {k: dict(v) for k, v in self.params.items()}

    def restore_params(self) -> None:
        """Restore backed up parameters (used when removing structure factor)."""
        if self._form_factor_params:
            self.params = {k: dict(v) for k, v in self._form_factor_params.items()}
            self._form_factor_params = {}

    def has_backed_up_params(self) -> bool:
        """
        Check if there are backed up parameters.

        Returns:
            True if parameters have been backed up, False otherwise
        """
        return bool(self._form_factor_params)

    def get_backed_up_params(self) -> dict[str, dict[str, Any]]:
        """
        Get the backed up form factor parameters.

        Returns:
            Dictionary of backed up parameters
        """
        return self._form_factor_params

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
        if radius_effective_mode not in ['unconstrained', 'link_radius']:
            raise ValueError(
                f"Invalid radius_effective_mode '{radius_effective_mode}'. "
                "Use 'unconstrained' or 'link_radius'."
            )

        # Backup form factor parameters if not already done
        if not self._form_factor_params:
            self.backup_params()

        self._structure_factor_name = structure_factor_name
        self._radius_effective_mode = radius_effective_mode

        # Rebuild parameters from product model
        new_params = {}
        for param in kernel.info.parameters.kernel_parameters:
            # Preserve existing values if parameter already exists
            if param.name in self._form_factor_params:
                new_params[param.name] = dict(self._form_factor_params[param.name])
            else:
                new_params[param.name] = {
                    'value': param.default,
                    'min': param.limits[0] if param.limits[0] > -np.inf else 0,
                    'max': param.limits[1] if param.limits[1] < np.inf else param.default * 10,
                    'vary': False,
                    'description': param.description,
                }

        # Ensure scale and background are present
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

        self.params = new_params

        # Handle radius_effective linking
        if radius_effective_mode == 'link_radius':
            if 'radius' in self.params and 'radius_effective' in self.params:
                # Link radius_effective to radius
                self.params['radius_effective']['value'] = self.params['radius']['value']
                self.params['radius_effective']['vary'] = False
            else:
                import warnings

                warnings.warn(
                    'Cannot link radius_effective to radius: one or both parameters not found. '
                    'Using unconstrained mode.',
                    stacklevel=3,
                )
                self._radius_effective_mode = 'unconstrained'

    def remove_structure_factor(self) -> str:
        """
        Remove structure factor and restore form factor parameters.

        Returns:
            Name of the removed structure factor

        Raises:
            ValueError: If no structure factor is currently set
        """
        if self._structure_factor_name is None:
            raise ValueError('No structure factor is currently set.')

        sf_name = self._structure_factor_name
        self.restore_params()
        self._structure_factor_name = None
        self._radius_effective_mode = 'unconstrained'

        return sf_name

    def get_structure_factor(self) -> Optional[str]:
        """
        Get the name of the currently applied structure factor.

        Returns:
            Name of the structure factor, or None if no structure factor is set
        """
        return self._structure_factor_name

    def get_radius_effective_mode(self) -> str:
        """
        Get the current radius_effective mode.

        Returns:
            Current radius_effective mode ('unconstrained' or 'link_radius')
        """
        return self._radius_effective_mode

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
        self.polydisperse_params = {}
        self._pd_param_states = {}

        for param_name in self._polydisperse_param_names:
            self.polydisperse_params[param_name] = {
                'pd': PD_DEFAULTS['pd'],
                'pd_n': PD_DEFAULTS['pd_n'],
                'pd_nsigma': PD_DEFAULTS['pd_nsigma'],
                'pd_type': PD_DEFAULTS['pd_type'],
            }
            # By default, PD parameters are not varied
            self._pd_param_states[param_name] = False

    def get_polydisperse_parameters(self) -> list[str]:
        """
        Return list of parameter names that support polydispersity.

        Returns:
            List of parameter names that can have polydispersity applied
        """
        return list(self._polydisperse_param_names)

    def has_polydisperse_parameters(self) -> bool:
        """
        Check if the current model has any polydisperse parameters.

        Returns:
            True if model has polydisperse parameters, False otherwise
        """
        return len(self._polydisperse_param_names) > 0

    def set_pd_param(
        self,
        base_param: str,
        pd_width: Optional[float] = None,
        pd_n: Optional[int] = None,
        pd_nsigma: Optional[float] = None,
        pd_type: Optional[str] = None,
        vary: Optional[bool] = None,
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
        if base_param not in self._polydisperse_param_names:
            available = ', '.join(self._polydisperse_param_names)
            raise KeyError(
                f"Parameter '{base_param}' does not support polydispersity. "
                f'Available polydisperse parameters: {available}'
            )

        if pd_type is not None and pd_type not in PD_DISTRIBUTION_TYPES:
            raise ValueError(
                f"Invalid pd_type '{pd_type}'. Valid types: {', '.join(PD_DISTRIBUTION_TYPES)}"
            )

        if pd_width is not None:
            self.polydisperse_params[base_param]['pd'] = pd_width
        if pd_n is not None:
            self.polydisperse_params[base_param]['pd_n'] = pd_n
        if pd_nsigma is not None:
            self.polydisperse_params[base_param]['pd_nsigma'] = pd_nsigma
        if pd_type is not None:
            self.polydisperse_params[base_param]['pd_type'] = pd_type
        if vary is not None:
            self._pd_param_states[base_param] = vary

    def get_pd_param(self, base_param: str) -> dict[str, Any]:
        """
        Get polydispersity configuration for a specific parameter.

        Args:
            base_param: Name of the base parameter (e.g., 'radius')

        Returns:
            Dictionary with pd, pd_n, pd_nsigma, pd_type, and vary values

        Raises:
            KeyError: If base_param is not a polydisperse parameter
        """
        if base_param not in self._polydisperse_param_names:
            available = ', '.join(self._polydisperse_param_names)
            raise KeyError(
                f"Parameter '{base_param}' does not support polydispersity. "
                f'Available polydisperse parameters: {available}'
            )

        return {
            **self.polydisperse_params[base_param],
            'vary': self._pd_param_states.get(base_param, False),
        }

    def toggle_pd_visibility(self, enabled: bool) -> None:
        """
        Enable/disable polydispersity globally.

        When disabled, polydispersity parameters are excluded from fitting
        but their values are preserved for when PD is re-enabled.

        Args:
            enabled: Whether polydispersity should be enabled
        """
        self._pd_enabled = enabled

    def is_pd_enabled(self) -> bool:
        """
        Check if polydispersity is globally enabled.

        Returns:
            True if polydispersity is enabled, False otherwise
        """
        return self._pd_enabled

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
        if not self._pd_enabled:
            return {}

        pd_params = {}
        for param_name in self._polydisperse_param_names:
            pd_config = self.polydisperse_params[param_name]
            pd_params[f'{param_name}_pd'] = pd_config['pd']
            pd_params[f'{param_name}_pd_n'] = pd_config['pd_n']
            pd_params[f'{param_name}_pd_nsigma'] = pd_config['pd_nsigma']
            pd_params[f'{param_name}_pd_type'] = pd_config['pd_type']

        return pd_params

    def get_varying_pd_params(self) -> list[str]:
        """
        Get list of polydispersity parameter names set to vary.

        Only returns parameters when pd_enabled is True.

        Returns:
            List of base parameter names whose PD width should vary
        """
        if not self._pd_enabled:
            return []

        return [param_name for param_name, vary in self._pd_param_states.items() if vary]

    def display_pd_params(self) -> None:
        """Display polydispersity parameter values and settings."""
        if not self._polydisperse_param_names:
            print('No polydisperse parameters available for this model.')
            return

        status = 'ENABLED' if self._pd_enabled else 'DISABLED'
        print(f'\n{"=" * 90}')
        print(f'Polydispersity Status: {status}')
        print(f'{"=" * 90}')
        print(
            f'{"Parameter":<15} {"Width":<10} {"N Points":<10} {"N Sigma":<10} {"Type":<12} {"Vary":<8}'
        )
        print(f'{"-" * 90}')

        for param_name in self._polydisperse_param_names:
            pd_config = self.polydisperse_params[param_name]
            vary_str = '✓' if self._pd_param_states.get(param_name, False) else '✗'
            print(
                f'{param_name:<15} {pd_config["pd"]:<10.4g} {pd_config["pd_n"]:<10} '
                f'{pd_config["pd_nsigma"]:<10.4g} {pd_config["pd_type"]:<12} {vary_str:<8}'
            )
        print(f'{"=" * 90}\n')

    def clear(self) -> None:
        """Clear all parameters and reset state."""
        self.params = {}
        self.model_name = None
        self._structure_factor_name = None
        self._radius_effective_mode = 'unconstrained'
        self._form_factor_params = {}

        # Reset polydispersity state
        self._polydisperse_param_names = []
        self.polydisperse_params = {}
        self._pd_enabled = False
        self._pd_param_states = {}

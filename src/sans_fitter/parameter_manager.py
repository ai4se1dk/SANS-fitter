"""
Parameter Manager - Handles model parameter management for SANS fitting.

This module encapsulates all parameter-related operations including initialization,
validation, bounds management, and structure factor parameter linking.
"""

from typing import Any, Optional

import numpy as np


class ParameterManager:
    """
    Manages model parameters for SANS fitting.

    Handles parameter initialization, validation, bounds management,
    and special logic for structure factor parameter linking.

    Attributes:
        params: Dictionary of parameter configurations
        model_name: Name of the current model
        structure_factor_name: Name of applied structure factor (if any)
        radius_effective_mode: Mode for handling radius_effective ('unconstrained' or 'link_radius')
    """

    def __init__(self):
        """Initialize the parameter manager."""
        self.params: dict[str, dict[str, Any]] = {}
        self.model_name: Optional[str] = None
        self._structure_factor_name: Optional[str] = None
        self._radius_effective_mode: str = 'unconstrained'
        self._form_factor_params: dict[str, dict[str, Any]] = {}

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

        # Extract parameters from kernel
        for param in kernel.info.parameters.kernel_parameters:
            self.params[param.name] = {
                'value': param.default,
                'min': param.limits[0] if param.limits[0] > -np.inf else 0,
                'max': param.limits[1] if param.limits[1] < np.inf else param.default * 10,
                'vary': False,  # By default, parameters are fixed
                'description': param.description,
            }

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

    def clear(self) -> None:
        """Clear all parameters and reset state."""
        self.params = {}
        self.model_name = None
        self._structure_factor_name = None
        self._radius_effective_mode = 'unconstrained'
        self._form_factor_params = {}

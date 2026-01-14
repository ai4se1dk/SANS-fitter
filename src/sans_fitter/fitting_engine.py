"""
Fitting Engine - Abstract base class for SANS fitting engines.

This module defines the interface that all fitting engines must implement,
enabling a strategy pattern for different optimization methods.
"""

from abc import ABC, abstractmethod
from typing import Any


class FittingEngine(ABC):
    """
    Abstract base class for fitting engines.

    This defines the interface that all concrete fitting engines must implement,
    allowing SANSFitter to work with different optimization backends using a
    strategy pattern.
    """

    @abstractmethod
    def fit(
        self,
        data: Any,
        kernel: Any,
        param_manager: Any,
        method: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Perform the fit using this engine's optimization method.

        Args:
            data: SANS data object from sasdata
            kernel: SasModels kernel
            param_manager: ParameterManager instance with parameter configurations
            method: Optimization method (engine-specific)
            **kwargs: Additional engine-specific arguments

        Returns:
            Dictionary containing:
                - engine: Name of the fitting engine
                - method: Optimization method used
                - chisq: Final chi-squared value
                - parameters: Dict of fitted parameter values with uncertainties
                - result: Raw result object from the fitting engine
                - problem/fitted_model: Engine-specific model object (optional)

        Raises:
            ValueError: If fitting fails or inputs are invalid
        """
        pass

    @abstractmethod
    def get_fitted_curve(self, fit_result: dict[str, Any], data: Any, kernel: Any) -> tuple:
        """
        Calculate the fitted curve from fit results.

        Args:
            fit_result: Dictionary returned from fit() method
            data: SANS data object
            kernel: SasModels kernel

        Returns:
            Tuple of (q_values, intensity_values) for the fitted model
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of this fitting engine.

        Returns:
            String identifier for the engine (e.g., 'bumps', 'scipy')
        """
        pass

    @property
    @abstractmethod
    def available_methods(self) -> list[str]:
        """
        Get list of available optimization methods for this engine.

        Returns:
            List of method names supported by this engine
        """
        pass

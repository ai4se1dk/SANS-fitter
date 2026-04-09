from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass(slots=True)
class FitArtifacts:
    """Engine-specific runtime data needed after fitting."""

    fitted_curve: Optional[np.ndarray] = None
    raw_result: Any = None
    runtime_handle: Any = None
    runtime_key: Optional[str] = None


@dataclass(slots=True)
class FitResultContract:
    """Stable internal fit-result contract used across post-fit operations."""

    engine: str
    method: str
    chisq: float
    parameters: dict[str, dict[str, Any]]
    artifacts: FitArtifacts = field(default_factory=FitArtifacts)

    def to_legacy_dict(self) -> dict[str, Any]:
        """Expose the historical dict-based result shape for public compatibility."""
        result = {
            'engine': self.engine,
            'method': self.method,
            'chisq': self.chisq,
            'parameters': {name: dict(info) for name, info in self.parameters.items()},
        }

        if self.artifacts.raw_result is not None:
            result['result'] = self.artifacts.raw_result

        if self.artifacts.runtime_key and self.artifacts.runtime_handle is not None:
            result[self.artifacts.runtime_key] = self.artifacts.runtime_handle

        return result

    def require_fitted_curve(self) -> np.ndarray:
        """Return the fitted curve or raise if the contract is incomplete."""
        if self.artifacts.fitted_curve is None:
            raise ValueError('Fit result does not include a fitted curve.')
        return self.artifacts.fitted_curve

    def save_csv(self, filename: str, model_name: str, data: Any) -> None:
        """Save fit results, fitted curve, and residuals to CSV."""
        fitted_curve = self.require_fitted_curve()
        residuals = (data.y - fitted_curve) / data.dy

        with open(filename, 'w') as f:
            f.write('# SANS Fit Results\n')
            f.write(f'# Model: {model_name}\n')
            f.write(f'# Engine: {self.engine}\n')
            f.write(f'# Method: {self.method}\n')
            f.write(f'# Chi-squared: {self.chisq:.6f}\n')
            f.write('#\n')
            f.write('# Fitted Parameters:\n')
            for name, info in self.parameters.items():
                f.write(f'# {name}: {info["formatted"]}\n')
            f.write('#\n')
            f.write('Q,I_exp,dI_exp,I_fit,Residuals\n')

            for q, i_exp, di_exp, i_fit, res in zip(
                data.x, data.y, data.dy, fitted_curve, residuals
            ):
                f.write(f'{q:.6e},{i_exp:.6e},{di_exp:.6e},{i_fit:.6e},{res:.6e}\n')


def save_fit_result(
    filename: str, model_name: str, data: Any, fit_result: FitResultContract
) -> None:
    """Compatibility wrapper for saving fit results from SANSFitter."""
    fit_result.save_csv(filename=filename, model_name=model_name, data=data)

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .data_loader import _has_real_data


def _validate_export_lengths(**arrays: Any) -> None:
    """Raise when export arrays do not all share the same length."""
    lengths = {name: len(values) for name, values in arrays.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) > 1:
        mismatch = ', '.join(f'{name}={length}' for name, length in lengths.items())
        raise ValueError(f'Cannot export fit results with mismatched array lengths: {mismatch}')


def resolve_fit_index(fit_index: Any, n_points: int) -> np.ndarray:
    """Normalize a stored fit index to a boolean array of length *n_points*.

    A missing index (None) means every point was fitted.
    """
    if fit_index is None:
        return np.ones(n_points, dtype=bool)
    index = np.asarray(fit_index, dtype=bool)
    if index.shape != (n_points,):
        raise ValueError(
            f'Fit index length ({index.size}) does not match the data length ({n_points}).'
        )
    return index


@dataclass(slots=True)
class FitArtifacts:
    """Engine-specific runtime data needed after fitting."""

    fitted_curve: Optional[np.ndarray] = None
    fit_index: Optional[np.ndarray] = None
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
        """Save fit results, fitted curve, and residuals to CSV.

        Only the points included in the fit (inside the Q range, unmasked)
        are exported, so every row carries a fitted intensity and residual.
        """
        fitted_curve = self.require_fitted_curve()
        has_dx = _has_real_data(data.dx)

        index = resolve_fit_index(self.artifacts.fit_index, len(data.x))
        x = np.asarray(data.x)[index]
        y = np.asarray(data.y)[index]
        dy = np.asarray(data.dy)[index]
        dx = np.asarray(data.dx)[index] if has_dx else None

        arrays_to_validate = {
            'x': x,
            'y': y,
            'dy': dy,
            'fitted_curve': fitted_curve,
        }
        if has_dx:
            arrays_to_validate['dx'] = dx
        _validate_export_lengths(**arrays_to_validate)

        residuals = (y - fitted_curve) / dy
        _validate_export_lengths(residuals=residuals, **arrays_to_validate)

        with open(filename, 'w') as f:
            f.write('# SANS Fit Results\n')
            f.write(f'# Model: {model_name}\n')
            f.write(f'# Engine: {self.engine}\n')
            f.write(f'# Method: {self.method}\n')
            f.write(f'# Chi-squared: {self.chisq:.6f}\n')
            f.write(f'# Q range: {x.min():.6g} to {x.max():.6g}\n')
            f.write(f'# Points fitted: {len(x)} of {len(index)}\n')
            f.write('#\n')
            f.write('# Fitted Parameters:\n')
            for name, info in self.parameters.items():
                f.write(f'# {name}: {info["formatted"]}\n')
            f.write('#\n')

            if has_dx:
                f.write('Q,dQ,I_exp,dI_exp,I_fit,Residuals\n')
                for q, dq, i_exp, di_exp, i_fit, res in zip(x, dx, y, dy, fitted_curve, residuals):
                    f.write(f'{q:.6e},{dq:.6e},{i_exp:.6e},{di_exp:.6e},{i_fit:.6e},{res:.6e}\n')
            else:
                f.write('Q,I_exp,dI_exp,I_fit,Residuals\n')
                for q, i_exp, di_exp, i_fit, res in zip(x, y, dy, fitted_curve, residuals):
                    f.write(f'{q:.6e},{i_exp:.6e},{di_exp:.6e},{i_fit:.6e},{res:.6e}\n')


def save_fit_result(
    filename: str, model_name: str, data: Any, fit_result: FitResultContract
) -> None:
    """Compatibility wrapper for saving fit results from SANSFitter."""
    fit_result.save_csv(filename=filename, model_name=model_name, data=data)

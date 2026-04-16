"""
Shared read-only capability metadata for SANS fitting.

This module provides importable helper functions for model discovery,
parameter inspection, structure-factor metadata, and polydispersity
options. Both SANS-pilot and SANS-webapp can use these instead of
re-encoding the same scientific metadata in their MCP tool layers.
"""

from __future__ import annotations

from typing import Any

from .polydispersity import PD_DEFAULTS, PD_DISTRIBUTION_TYPES
from .sans_fitter import SANSFitter, get_all_models

# Canonical structure-factor catalog.  Kept here so that every MCP
# surface returns the same descriptions without hardcoding.
SUPPORTED_STRUCTURE_FACTORS: dict[str, str] = {
    'hardsphere': (
        'Hard sphere structure factor (Percus-Yevick closure) - for non-interacting hard spheres'
    ),
    'hayter_msa': ('Hayter-Penfold rescaled MSA - for charged spheres with Coulombic interactions'),
    'squarewell': ('Square well potential - for particles with short-range attraction'),
    'stickyhardsphere': (
        'Sticky hard sphere (Baxter model) - for particles with very short-range attraction'
    ),
}


def get_available_models() -> list[str]:
    """Return a sorted list of every sasmodels model name."""
    return get_all_models()


def get_supported_structure_factors() -> dict[str, str]:
    """Return the canonical structure-factor catalog with descriptions."""
    return dict(SUPPORTED_STRUCTURE_FACTORS)


def get_model_parameter_specs(model_name: str) -> dict[str, Any]:
    """Inspect parameters for a plain form-factor model.

    Creates a temporary ``SANSFitter``, loads *model_name*, and returns
    the resulting parameter dictionary (name -> {value, min, max, vary, …}).
    """
    fitter = SANSFitter()
    fitter.set_model(model_name)
    return fitter.params


def get_product_model_parameters(
    form_factor: str,
    structure_factor: str,
) -> dict[str, Any]:
    """Inspect parameters for a form_factor@structure_factor product model.

    Creates a temporary ``SANSFitter``, loads the product model, and
    returns the combined parameter dictionary.
    """
    fitter = SANSFitter()
    fitter.set_model(form_factor)
    fitter.set_structure_factor(structure_factor)
    return fitter.params


def get_polydispersity_options() -> dict[str, Any]:
    """Return PD distribution types, defaults, and field descriptions."""
    return {
        'distribution_types': PD_DISTRIBUTION_TYPES,
        'defaults': PD_DEFAULTS,
        'description': {
            'pd_width': 'Relative width of distribution (0.1 = 10% polydispersity)',
            'pd_type': ('Distribution shape (gaussian, lognormal, schulz, rectangle, boltzmann)'),
            'pd_n': 'Number of quadrature points (higher = more accurate, slower)',
            'pd_nsigma': 'Number of standard deviations to include',
            'vary': 'Whether to fit the pd_width during optimization',
        },
    }


def get_model_polydispersity_support(model_name: str) -> dict[str, Any]:
    """Check which parameters of *model_name* support polydispersity.

    Returns a dict with ``supports_polydispersity`` (bool) and
    ``polydisperse_parameters`` (list[str]).
    """
    fitter = SANSFitter()
    fitter.set_model(model_name)
    return {
        'supports_polydispersity': fitter.supports_polydispersity(),
        'polydisperse_parameters': fitter.get_polydisperse_parameters(),
    }

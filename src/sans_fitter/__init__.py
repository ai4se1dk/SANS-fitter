"""
SANS Model Fitter - A flexible template for fitting SANS data with SasModels
"""

from .capabilities import (
    SUPPORTED_STRUCTURE_FACTORS,
    get_available_models,
    get_model_parameter_specs,
    get_model_polydispersity_support,
    get_polydispersity_options,
    get_product_model_parameters,
    get_supported_structure_factors,
)
from .parameter_manager import ParameterManager
from .polydispersity import PD_DEFAULTS, PD_DISTRIBUTION_TYPES
from .results import FitResultContract
from .sans_fitter import SANSFitter, get_all_models
from .sasview_params import SasViewParam, SasViewParamFile, parse_sasview_params

__version__ = '0.0.3'
__all__ = [
    'SANSFitter',
    'ParameterManager',
    'PD_DEFAULTS',
    'PD_DISTRIBUTION_TYPES',
    'get_all_models',
    'FitResultContract',
    'SUPPORTED_STRUCTURE_FACTORS',
    'get_available_models',
    'get_model_parameter_specs',
    'get_model_polydispersity_support',
    'get_polydispersity_options',
    'get_product_model_parameters',
    'get_supported_structure_factors',
    'SasViewParam',
    'SasViewParamFile',
    'parse_sasview_params',
]

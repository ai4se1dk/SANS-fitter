"""
SANS Model Fitter - A flexible template for fitting SANS data with SasModels
"""

from . import data_ops
from .parameter_manager import ParameterManager
from .polydispersity import PD_DEFAULTS, PD_DISTRIBUTION_TYPES
from .results import FitResultContract, PosteriorSummary
from .sans_fitter import SANSFitter, get_all_models

__version__ = '0.2.2'
__all__ = [
    'SANSFitter',
    'ParameterManager',
    'PD_DEFAULTS',
    'PD_DISTRIBUTION_TYPES',
    'get_all_models',
    'FitResultContract',
    'data_ops',
    'examples',
    'PosteriorSummary',
]

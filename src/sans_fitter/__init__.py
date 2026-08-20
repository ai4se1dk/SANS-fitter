"""
SANS Model Fitter - A flexible template for fitting SANS data with SasModels
"""

from . import data_ops, examples, pr_inversion
from .parameter_manager import ParameterManager
from .polydispersity import PD_DEFAULTS, PD_DISTRIBUTION_TYPES
from .pr_inversion import InsufficientDataError, PrEstimationError, PrResult
from .results import FitResultContract, PosteriorSummary
from .sans_fitter import SANSFitter, get_all_models

__version__ = '0.3.0'
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
    'pr_inversion',
    'PrResult',
    'InsufficientDataError',
    'PrEstimationError',
]

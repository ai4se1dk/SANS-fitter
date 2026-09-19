"""
SANS Model Fitter - A flexible template for fitting SANS data with SasModels
"""

__version__ = '0.4.0'

from . import examples
from . import inversion as pr_inversion
from .console import set_verbosity
from .data import ops as data_ops
from .fitter import SANSFitter, get_all_models, get_structure_factors
from .inversion import InsufficientDataError, PrEstimationError, PrResult
from .modeling.constraints import ConstraintError, ExpressionError
from .modeling.parameters import ParameterManager
from .modeling.polydispersity import PD_DEFAULTS, PD_DISTRIBUTION_TYPES
from .multi_results import MultiFitReport, MultiFitResult
from .multifit import DatasetHandle, MultiFitter
from .results import FitResultContract, PosteriorSummary

__all__ = [
    'SANSFitter',
    'MultiFitter',
    'DatasetHandle',
    'MultiFitResult',
    'MultiFitReport',
    'ConstraintError',
    'ExpressionError',
    'ParameterManager',
    'PD_DEFAULTS',
    'PD_DISTRIBUTION_TYPES',
    'get_all_models',
    'set_verbosity',
    'get_structure_factors',
    'FitResultContract',
    'data_ops',
    'examples',
    'PosteriorSummary',
    'pr_inversion',
    'PrResult',
    'InsufficientDataError',
    'PrEstimationError',
]

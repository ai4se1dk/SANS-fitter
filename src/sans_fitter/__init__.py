"""
SANS Model Fitter - A flexible template for fitting SANS data with SasModels
"""

from .bumps_engine import BumpsFittingEngine
from .fitting_engine import FittingEngine
from .parameter_manager import ParameterManager
from .sans_fitter import SANSFitter

try:
    from .scipy_engine import ScipyFittingEngine  # noqa: F401

    __all__ = [
        'SANSFitter',
        'ParameterManager',
        'FittingEngine',
        'BumpsFittingEngine',
        'ScipyFittingEngine',
    ]
except ImportError:
    __all__ = ['SANSFitter', 'ParameterManager', 'FittingEngine', 'BumpsFittingEngine']

__version__ = '0.1.0'

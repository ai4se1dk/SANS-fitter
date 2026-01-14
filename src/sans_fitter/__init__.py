"""
SANS Model Fitter - A flexible template for fitting SANS data with SasModels
"""

from .parameter_manager import ParameterManager
from .sans_fitter import SANSFitter

__version__ = '0.1.0'
__all__ = ['SANSFitter', 'ParameterManager']

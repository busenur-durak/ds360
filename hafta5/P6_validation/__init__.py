"""
P6: Validation Module

Bu modül time series model validation ve cross-validation işlemlerini içerir.

Modüller:
- time_series_cv.py: Comprehensive cross-validation
- time_series_cv_simple.py: Simplified cross-validation

Usage:
    from P6_validation import time_series_cv
    from P6_validation import time_series_cv_simple
"""

__version__ = "1.0.0"
__author__ = "M5 Forecasting Team"

try:
    from .time_series_cv import run_time_series_cv as run_cv_comprehensive
    from .time_series_cv_simple import run_time_series_cv as run_cv_simple
    
    __all__ = ['run_cv_comprehensive', 'run_cv_simple']
except ImportError:
    __all__ = []
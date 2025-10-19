"""
P3: Traditional Models Module

Bu modül geleneksel istatistiksel time series modellerini içerir.

Modüller:
- arima_single_item.py: ARIMA model implementation

Usage:
    from P3_traditional_models import arima_single_item
"""

__version__ = "1.0.0"
__author__ = "M5 Forecasting Team"

try:
    from .arima_single_item import main as run_arima
    
    __all__ = ['run_arima']
except ImportError:
    __all__ = []
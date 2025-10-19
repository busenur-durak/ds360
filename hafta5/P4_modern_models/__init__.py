"""
P4: Modern Models Module

Bu modül modern time series forecasting modellerini içerir.

Modüller:
- prophet_single_item.py: Facebook Prophet implementation

Usage:
    from P4_modern_models import prophet_single_item
"""

__version__ = "1.0.0"
__author__ = "M5 Forecasting Team"

try:
    from .prophet_single_item import main as run_prophet
    
    __all__ = ['run_prophet']
except ImportError:
    __all__ = []
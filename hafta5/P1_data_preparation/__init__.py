"""
P1: Data Preparation Module

Bu modül M5 Competition verisinin hazırlanması ve preprocessing işlemlerini içerir.

Modüller:
- create_m5_subset.py: M5 verisinden subset oluşturma
- create_sample_data.py: Sentetik veri oluşturma

Usage:
    from P1_data_preparation import create_m5_subset
    from P1_data_preparation import create_sample_data
"""

__version__ = "1.0.0"
__author__ = "M5 Forecasting Team"

# Import main functions for easy access
try:
    from .create_m5_subset import main as create_subset
    from .create_sample_data import main as create_sample
    
    __all__ = ['create_subset', 'create_sample']
except ImportError:
    # Graceful degradation if modules not available
    __all__ = []
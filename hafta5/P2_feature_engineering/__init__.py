"""
P2: Feature Engineering Module

Bu modül time series için feature engineering işlemlerini içerir.

Modüller:
- feature_engineering.py: Lag, rolling ve seasonal features

Usage:
    from P2_feature_engineering import feature_engineering
"""

__version__ = "1.0.0"
__author__ = "M5 Forecasting Team"

try:
    from .feature_engineering import main as create_features
    
    __all__ = ['create_features']
except ImportError:
    __all__ = []
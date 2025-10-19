"""
P7: Automation Module

Bu modül pipeline automation ve workflow orchestration işlemlerini içerir.

Modüller:
- prefect_demand_forecast.py: Prefect-based automated forecasting pipeline

Usage:
    from P7_automation import prefect_demand_forecast
"""

__version__ = "1.0.0"
__author__ = "M5 Forecasting Team"

try:
    from .prefect_demand_forecast import demand_forecast_flow
    
    __all__ = ['demand_forecast_flow']
except ImportError:
    __all__ = []
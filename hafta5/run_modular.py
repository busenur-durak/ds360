#!/usr/bin/env python3
"""
M5 Forecasting - Modular Pipeline Runner

Bu script yeni modüler yapıyı kullanarak adım adım pipeline çalıştırır.
Her modülü ayrı ayrı import edip çalıştırabilirsiniz.

Usage:
    python run_modular.py [--module MODULE_NAME]
    
Examples:
    python run_modular.py                    # Full pipeline
    python run_modular.py --module P1        # Only data preparation
    python run_modular.py --module P3        # Only ARIMA
"""

import sys
import os
import argparse
from datetime import datetime

def run_p1_data_preparation():
    """P1: Data Preparation Module"""
    print("📊 P1: Data Preparation başlatılıyor...")
    try:
        from P1_data_preparation.create_m5_subset import main as create_subset
        print("   🔄 M5 subset oluşturuluyor...")
        create_subset()
        print("   ✅ P1 tamamlandı")
        return True
    except Exception as e:
        print(f"   ❌ P1 hatası: {e}")
        return False

def run_p2_feature_engineering():
    """P2: Feature Engineering Module"""
    print("⚙️ P2: Feature Engineering başlatılıyor...")
    try:
        from P2_feature_engineering.feature_engineering import main as create_features
        print("   🔄 Features oluşturuluyor...")
        create_features()
        print("   ✅ P2 tamamlandı")
        return True
    except Exception as e:
        print(f"   ❌ P2 hatası: {e}")
        return False

def run_p3_traditional_models():
    """P3: Traditional Models Module"""
    print("📈 P3: ARIMA Model başlatılıyor...")
    try:
        from P3_traditional_models.arima_single_item import main as run_arima
        print("   🔄 ARIMA çalıştırılıyor...")
        run_arima()
        print("   ✅ P3 tamamlandı")
        return True
    except Exception as e:
        print(f"   ❌ P3 hatası: {e}")
        return False

def run_p4_modern_models():
    """P4: Modern Models Module"""
    print("🚀 P4: Prophet Model başlatılıyor...")
    try:
        from P4_modern_models.prophet_single_item import main as run_prophet
        print("   🔄 Prophet çalıştırılıyor...")
        run_prophet()
        print("   ✅ P4 tamamlandı")
        return True
    except Exception as e:
        print(f"   ❌ P4 hatası: {e}")
        return False

def run_p5_ml_models():
    """P5: ML Models Module"""
    print("🤖 P5: LightGBM Model başlatılıyor...")
    try:
        from P5_ml_models.lightgbm_multi_item import main as run_lightgbm
        print("   🔄 LightGBM çalıştırılıyor...")
        run_lightgbm()
        print("   ✅ P5 tamamlandı")
        return True
    except Exception as e:
        print(f"   ❌ P5 hatası: {e}")
        return False

def run_p6_validation():
    """P6: Validation Module"""
    print("✅ P6: Cross-Validation başlatılıyor...")
    try:
        from P6_validation.time_series_cv_simple import run_time_series_cv
        print("   🔄 Time Series CV çalıştırılıyor...")
        run_time_series_cv()
        print("   ✅ P6 tamamlandı")
        return True
    except Exception as e:
        print(f"   ❌ P6 hatası: {e}")
        return False

def run_p7_automation():
    """P7: Automation Module"""
    print("🔄 P7: Prefect Pipeline başlatılıyor...")
    try:
        from P7_automation.prefect_demand_forecast import demand_forecast_flow
        print("   🔄 Automated forecasting çalıştırılıyor...")
        result = demand_forecast_flow(
            run_date=datetime.now().strftime('%Y-%m-%d'),
            forecast_days=7
        )
        print("   ✅ P7 tamamlandı")
        return True
    except Exception as e:
        print(f"   ❌ P7 hatası: {e}")
        return False

def main():
    """Ana çalıştırıcı fonksiyon"""
    parser = argparse.ArgumentParser(description='M5 Forecasting Modular Pipeline')
    parser.add_argument('--module', type=str, choices=['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'], 
                       help='Specific module to run')
    args = parser.parse_args()
    
    print("🏪 M5 FORECASTING - MODULAR PIPELINE")
    print("=" * 50)
    print(f"📅 Başlangıç zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"📁 Çalışma dizini: {os.getcwd()}")
    
    module_functions = {
        'P1': run_p1_data_preparation,
        'P2': run_p2_feature_engineering,
        'P3': run_p3_traditional_models,
        'P4': run_p4_modern_models,
        'P5': run_p5_ml_models,
        'P6': run_p6_validation,
        'P7': run_p7_automation
    }
    
    success_count = 0
    total_count = 0
    
    if args.module:
        # Specific module run
        print(f"\n🎯 Sadece {args.module} modülü çalıştırılıyor...")
        func = module_functions[args.module]
        success = func()
        success_count = 1 if success else 0
        total_count = 1
    else:
        # Full pipeline run
        print("\n🔄 Full pipeline çalıştırılıyor...")
        print("-" * 50)
        
        for module_name, func in module_functions.items():
            print(f"\n▶️  {module_name} Modülü:")
            success = func()
            if success:
                success_count += 1
            total_count += 1
            print("")
    
    print("-" * 50)
    print(f"🎉 Pipeline tamamlandı!")
    print(f"✅ Başarılı modüller: {success_count}/{total_count}")
    print(f"⏰ Bitiş zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success_count == total_count:
        print("🎊 Tüm modüller başarıyla çalıştı!")
        return 0
    else:
        print("⚠️ Bazı modüllerde hata oluştu.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
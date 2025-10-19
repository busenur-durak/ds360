#!/usr/bin/env python3
"""
M5 Forecasting Pipeline - Docker Runner

Bu script Docker container içinde Prefect flow'unu tek seferlik çalıştırır.
Schedule olmadan, sadece bir kez tahmin üretir ve çıkar.

Usage:
    python run.py
    
Docker Usage:
    docker build -t m5-forecast:dev .
    docker run --rm -v $(pwd)/artifacts:/app/artifacts m5-forecast:dev
"""

import sys
import os
from datetime import datetime

# Prefect flow'unu modüler yapıdan import et
try:
    from P7_automation.prefect_demand_forecast import demand_forecast_flow
    print("✅ Prefect demand forecast flow başarıyla import edildi (P7 modülünden)")
except ImportError as e:
    print(f"❌ Import hatası: {e}")
    print("🔄 Fallback olarak ana klasörden deneniyor...")
    try:
        from prefect_demand_forecast import demand_forecast_flow
        print("✅ Prefect demand forecast flow başarıyla import edildi (ana klasörden)")
    except ImportError as e2:
        print(f"❌ Fallback import hatası: {e2}")
        sys.exit(1)

def main():
    """Ana çalıştırıcı fonksiyon"""
    print("🐳 M5 FORECASTING - DOCKER PIPELINE")
    print("=" * 50)
    print(f"📅 Başlangıç zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"📁 Çalışma dizini: {os.getcwd()}")
    print(f"📂 Artifacts: ./artifacts")
    
    # Gerekli klasörlerin var olduğunu kontrol et
    required_paths = [
        "./data",
        "./artifacts/datasets", 
        "./artifacts/models",
        "./artifacts/figures",
        "./artifacts/preds",
        "./artifacts/reports"
    ]
    
    print("\n🔍 Klasör kontrolü:")
    for path in required_paths:
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"   {status} {path}")
        if not exists and path.startswith("./artifacts"):
            os.makedirs(path, exist_ok=True)
            print(f"      📁 Oluşturuldu: {path}")
    
    # Data kontrolü
    data_files = [
        "./data/sales_train_CA.csv",
        "./data/calendar.csv", 
        "./data/sell_prices.csv"
    ]
    
    print("\n📊 Veri dosyası kontrolü:")
    missing_data = []
    for file in data_files:
        exists = os.path.exists(file)
        status = "✅" if exists else "❌"
        print(f"   {status} {file}")
        if not exists:
            missing_data.append(file)
    
    if missing_data:
        print("\n⚠️  Eksik veri dosyaları var!")
        print("   Docker çalıştırırken data klasörünü mount edin:")
        print("   docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts m5-forecast:dev")
        print("\n   Veya veri dosyalarını Docker build öncesi hazırlayın.")
    
    try:
        print("\n🚀 Pipeline başlatılıyor...")
        print("-" * 50)
        
        # Flow'u çalıştır
        result = demand_forecast_flow(
            run_date=datetime.now().strftime('%Y-%m-%d'),
            forecast_days=7,
            artifacts_path="./artifacts"
        )
        
        print("-" * 50)
        print("🎉 Pipeline başarıyla tamamlandı!")
        print(f"📊 Status: {result.get('status', 'UNKNOWN')}")
        print(f"📈 Tahmin sayısı: {result.get('prediction_count', 'N/A')}")
        print(f"🏪 Ürün sayısı: {result.get('item_count', 'N/A')}")
        print(f"📁 CSV dosyası: {result.get('csv_path', 'N/A')}")
        print(f"📊 PNG dosyası: {result.get('png_path', 'N/A')}")
        
        # Output dosyalarını kontrol et
        if 'csv_path' in result and os.path.exists(result['csv_path']):
            print(f"✅ CSV çıktısı oluşturuldu: {os.path.getsize(result['csv_path'])} bytes")
        if 'png_path' in result and os.path.exists(result['png_path']):
            print(f"✅ PNG çıktısı oluşturuldu: {os.path.getsize(result['png_path'])} bytes")
        
        print(f"\n⏰ Bitiş zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Kullanıcı tarafından durduruldu")
        return 1
    except Exception as e:
        print(f"\n❌ Pipeline hatası: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
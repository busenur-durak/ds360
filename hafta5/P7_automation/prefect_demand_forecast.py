#!/usr/bin/env python3
"""
Prefect Demand Forecasting Pipeline

Bu script her sabah 09:00 Europe/Istanbul saatinde çalışacak otomatik talep tahmin pipeline'ı:
1. Veri yükleme
2. Feature engineering
3. Model eğitimi/yükleme
4. Tahmin üretme
5. Sonuçları kaydetme

Prefect Schedule:
- Cron: "0 9 * * *" Europe/Istanbul
- Her gün sabah 9'da çalışır
- Production-ready workflow orchestration

Kullanım:
python prefect_demand_forecast.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Prefect
try:
    from prefect import task, flow
    PREFECT_AVAILABLE = True
except ImportError:
    print("⚠️  Prefect bulunamadı. Normal fonksiyonlar kullanılacak.")
    PREFECT_AVAILABLE = False
    
    # Mock decorators for non-Prefect environments
    def task(func):
        return func
    def flow(func):
        return func

# LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("❌ LightGBM kütüphanesi bulunamadı.")
    LIGHTGBM_AVAILABLE = False

# Preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

# ================================
# PREFECT TASKS
# ================================

@task
def load_data_task(artifacts_path: str = "/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta5/artifacts") -> pd.DataFrame:
    """
    Veri yükleme görevi (P1'in basitleştirilmiş hali)
    
    Gerçek production'da:
    - Database connection
    - API calls
    - Data validation
    """
    print("📁 Veri yükleme başlıyor...")
    
    try:
        # Mevcut feature engineered veriyi yükle
        train_path = f'{artifacts_path}/datasets/fe_train.parquet'
        valid_path = f'{artifacts_path}/datasets/fe_valid.parquet'
        
        if os.path.exists(train_path) and os.path.exists(valid_path):
            train_df = pd.read_parquet(train_path)
            valid_df = pd.read_parquet(valid_path)
            full_df = pd.concat([train_df, valid_df]).sort_index()
            
            print(f"   ✓ Veri yüklendi: {full_df.shape}")
            print(f"   • Tarih aralığı: {full_df.index.min()} - {full_df.index.max()}")
            print(f"   • Ürün sayısı: {full_df['item_id'].nunique()}")
            
            return full_df
        else:
            raise FileNotFoundError("Feature engineered veriler bulunamadı. Önce önceki scriptleri çalıştırın.")
            
    except Exception as e:
        print(f"   ❌ Veri yükleme hatası: {e}")
        raise

@task
def feature_engineer_task(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering görevi (P2'den gerekli kısım)
    
    Not: Bu örnekte veri zaten feature engineered, 
    ama production'da fresh data için gerekli
    """
    print("⚙️ Feature engineering başlıyor...")
    
    try:
        # Veri zaten FE edilmiş ama son kontroller
        processed_df = data_df.copy()
        
        # Eksik değer kontrolü
        missing_counts = processed_df.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"   ⚠️  {missing_counts.sum()} eksik değer bulundu, doldurulacak")
            # Lag ve rolling features için forward fill
            processed_df = processed_df.fillna(method='ffill').fillna(0)
        
        # Feature validation
        required_features = ['lag_1', 'lag_7', 'lag_28', 'roll_mean_7', 'roll_mean_28', 
                           'dow', 'dom', 'weekofyear', 'month', 'item_id', 'store_id', 'sales']
        
        missing_features = [f for f in required_features if f not in processed_df.columns]
        if missing_features:
            raise ValueError(f"Eksik özellikler: {missing_features}")
        
        print(f"   ✓ Feature engineering tamamlandı: {processed_df.shape}")
        print(f"   • Özellik sayısı: {len(processed_df.columns)}")
        
        return processed_df
        
    except Exception as e:
        print(f"   ❌ Feature engineering hatası: {e}")
        raise

@task
def train_or_load_model_task(data_df: pd.DataFrame, 
                            artifacts_path: str = "./artifacts") -> Dict:
    """
    Model eğitimi veya yükleme görevi
    
    Eğer model dosyası varsa yükle, yoksa eğit
    """
    print("🤖 Model eğitimi/yükleme başlıyor...")
    
    model_path = f'{artifacts_path}/models/lgbm.pkl'
    
    try:
        # Model dosyası var mı kontrol et
        if os.path.exists(model_path):
            print("   📂 Mevcut model yükleniyor...")
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            model = model_data['model']
            feature_cols = model_data['feature_cols']
            label_encoders = model_data['label_encoders']
            
            print(f"   ✓ Model yüklendi: LightGBM")
            print(f"   • Özellik sayısı: {len(feature_cols)}")
            print(f"   • Model iterasyonu: {model.best_iteration}")
            
            return {
                'model': model,
                'feature_cols': feature_cols,
                'label_encoders': label_encoders,
                'is_new_model': False
            }
            
        else:
            print("   🎯 Yeni model eğitiliyor...")
            
            # Kategorik encoding
            label_encoders = {}
            data_encoded = data_df.copy()
            
            for col in ['item_id', 'store_id']:
                if col in data_encoded.columns:
                    le = LabelEncoder()
                    data_encoded[f'{col}_encoded'] = le.fit_transform(data_encoded[col])
                    label_encoders[col] = le
            
            # Feature/target ayırma
            feature_cols = [col for col in data_encoded.columns if col in [
                'lag_1', 'lag_7', 'lag_28', 'roll_mean_7', 'roll_mean_28',
                'dow', 'dom', 'weekofyear', 'month', 'item_id_encoded', 'store_id_encoded'
            ]]
            
            X = data_encoded[feature_cols].fillna(0)
            y = data_encoded['sales']
            
            # Train/validation split (son %10 validation)
            split_idx = int(len(X) * 0.9)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # LightGBM parametreleri
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'verbose': -1,
                'random_state': 42
            }
            
            # Model eğitimi
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=500,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # Model kaydet
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model_data = {
                'model': model,
                'feature_cols': feature_cols,
                'label_encoders': label_encoders,
                'training_date': datetime.now().isoformat()
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"   ✓ Model eğitimi tamamlandı ve kaydedildi")
            print(f"   • Best iteration: {model.best_iteration}")
            
            return {
                'model': model,
                'feature_cols': feature_cols,
                'label_encoders': label_encoders,
                'is_new_model': True
            }
            
    except Exception as e:
        print(f"   ❌ Model eğitimi/yükleme hatası: {e}")
        raise

@task
def predict_task(data_df: pd.DataFrame, 
                model_info: Dict, 
                forecast_days: int = 7) -> pd.DataFrame:
    """
    Tahmin üretme görevi
    
    Son güne kadar güncelle ve +7 gün (veya +28) tahmin üret
    """
    print(f"🔮 {forecast_days} günlük tahmin üretiliyor...")
    
    try:
        model = model_info['model']
        feature_cols = model_info['feature_cols']
        label_encoders = model_info['label_encoders']
        
        # Son tarihi bul
        last_date = data_df.index.max()
        forecast_start = last_date + timedelta(days=1)
        
        print(f"   • Son veri tarihi: {last_date}")
        print(f"   • Tahmin başlangıcı: {forecast_start}")
        
        # Her ürün için tahmin
        all_predictions = []
        unique_items = data_df['item_id'].unique()
        
        for item_id in unique_items:
            # Ürün verisini al
            item_data = data_df[data_df['item_id'] == item_id].copy()
            
            if len(item_data) == 0:
                continue
            
            # Son satırı template olarak kullan
            last_row = item_data.iloc[-1].copy()
            
            # Kategorik encoding
            for col, le in label_encoders.items():
                if col in last_row:
                    try:
                        last_row[f'{col}_encoded'] = le.transform([last_row[col]])[0]
                    except ValueError:
                        last_row[f'{col}_encoded'] = 0  # Unseen value
            
            # Her gün için tahmin
            for day in range(forecast_days):
                forecast_date = forecast_start + timedelta(days=day)
                
                # Tarih özelliklerini güncelle
                current_features = last_row.copy()
                current_features['dow'] = forecast_date.weekday()
                current_features['dom'] = forecast_date.day
                current_features['weekofyear'] = forecast_date.isocalendar()[1]
                current_features['month'] = forecast_date.month
                
                # Tahmin yap
                X_pred = current_features[feature_cols].values.reshape(1, -1)
                y_pred = model.predict(X_pred, num_iteration=model.best_iteration)[0]
                y_pred = max(0, y_pred)  # Negatif değerleri 0 yap
                
                # Sonucu kaydet
                all_predictions.append({
                    'date': forecast_date,
                    'item_id': item_id,
                    'store_id': last_row['store_id'],
                    'predicted_sales': y_pred,
                    'model_type': 'LightGBM',
                    'forecast_horizon': day + 1
                })
        
        # DataFrame'e çevir
        predictions_df = pd.DataFrame(all_predictions)
        
        print(f"   ✓ {len(predictions_df)} tahmin üretildi")
        print(f"   • Ürün sayısı: {predictions_df['item_id'].nunique()}")
        print(f"   • Ortalama tahmin: {predictions_df['predicted_sales'].mean():.2f}")
        
        return predictions_df
        
    except Exception as e:
        print(f"   ❌ Tahmin hatası: {e}")
        raise

@task
def save_outputs_task(predictions_df: pd.DataFrame, 
                     run_date: str,
                     artifacts_path: str = "./artifacts") -> Dict[str, str]:
    """
    Sonuçları kaydetme görevi
    
    CSV ve PNG dosyalarını ./artifacts/preds/ klasörüne kaydet
    """
    print("💾 Sonuçlar kaydediliyor...")
    
    try:
        # Çıktı klasörü
        output_dir = f'{artifacts_path}/preds'
        os.makedirs(output_dir, exist_ok=True)
        
        # Dosya adları
        run_date_str = datetime.strptime(run_date, '%Y-%m-%d').strftime('%Y%m%d')
        csv_path = f'{output_dir}/run_{run_date_str}.csv'
        png_path = f'{output_dir}/run_{run_date_str}_summary.png'
        
        # 1. CSV kaydet
        predictions_df.to_csv(csv_path, index=False)
        print(f"   ✓ CSV kaydedildi: {csv_path}")
        
        # 2. Özet grafik oluştur
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Günlük toplam tahmin
        daily_total = predictions_df.groupby('date')['predicted_sales'].sum()
        axes[0, 0].plot(daily_total.index, daily_total.values, marker='o', linewidth=2)
        axes[0, 0].set_title('Günlük Toplam Tahmin')
        axes[0, 0].set_ylabel('Toplam Satış')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Ürün bazında toplam
        item_totals = predictions_df.groupby('item_id')['predicted_sales'].sum().sort_values(ascending=False)
        axes[0, 1].bar(range(len(item_totals)), item_totals.values, alpha=0.7)
        axes[0, 1].set_title('Ürün Bazında Toplam Tahmin')
        axes[0, 1].set_ylabel('Toplam Satış')
        axes[0, 1].set_xticks(range(len(item_totals)))
        axes[0, 1].set_xticklabels(item_totals.index, rotation=45)
        
        # Forecast horizon analizi
        horizon_avg = predictions_df.groupby('forecast_horizon')['predicted_sales'].mean()
        axes[1, 0].plot(horizon_avg.index, horizon_avg.values, marker='s', linewidth=2)
        axes[1, 0].set_title('Tahmin Ufku Analizi')
        axes[1, 0].set_xlabel('Gün')
        axes[1, 0].set_ylabel('Ortalama Satış')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Özet istatistikler
        axes[1, 1].axis('off')
        stats_text = f"""
Tahmin Özeti ({run_date})

• Toplam tahmin: {len(predictions_df):,}
• Ürün sayısı: {predictions_df['item_id'].nunique()}
• Forecast horizon: {predictions_df['forecast_horizon'].max()} gün
• Ortalama günlük satış: {predictions_df['predicted_sales'].mean():.1f}
• Toplam beklenen satış: {predictions_df['predicted_sales'].sum():.0f}
• Min tahmin: {predictions_df['predicted_sales'].min():.1f}
• Max tahmin: {predictions_df['predicted_sales'].max():.1f}

Model: LightGBM
Pipeline: Prefect Automated
Çalışma zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=11, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle(f'Demand Forecast Summary - {run_date}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # PNG kaydet
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ PNG kaydedildi: {png_path}")
        
        # Sonuç özeti
        summary = {
            'csv_path': csv_path,
            'png_path': png_path,
            'total_predictions': len(predictions_df),
            'n_items': int(predictions_df['item_id'].nunique()),
            'forecast_days': int(predictions_df['forecast_horizon'].max()),
            'total_expected_sales': float(predictions_df['predicted_sales'].sum()),
            'avg_daily_sales': float(predictions_df['predicted_sales'].mean())
        }
        
        print(f"   ✓ Çıktılar başarıyla kaydedildi")
        
        return summary
        
    except Exception as e:
        print(f"   ❌ Çıktı kaydetme hatası: {e}")
        raise

# ================================
# PREFECT FLOW
# ================================

@flow
def demand_forecast_flow(run_date: Optional[str] = None, 
                        forecast_days: int = 7,
                        artifacts_path: str = "./artifacts") -> Dict:
    """
    Ana demand forecasting flow
    
    Parameters:
    - run_date: Çalışma tarihi (YYYY-MM-DD format, default: bugün)
    - forecast_days: Kaç gün tahmin (default: 7)
    - artifacts_path: Artifacts klasör yolu
    
    Schedule: Cron "0 9 * * *" Europe/Istanbul
    - Her gün sabah 09:00'da çalışır
    - İstanbul saati ile (Türkiye timezone)
    - Production ortamında Prefect server/cloud ile yönetilir
    """
    
    # Default run date
    if run_date is None:
        run_date = datetime.now().strftime('%Y-%m-%d')
    
    print("🏪 DEMAND FORECAST PIPELINE BAŞLIYOR")
    print("=" * 50)
    print(f"📅 Çalışma tarihi: {run_date}")
    print(f"🔮 Forecast horizon: {forecast_days} gün")
    print(f"📁 Artifacts path: {artifacts_path}")
    print(f"⏰ Başlangıç zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if PREFECT_AVAILABLE:
        print("✅ Prefect modu: Task orchestration aktif")
    else:
        print("⚠️  Normal mod: Sequential execution")
    
    print("-" * 50)
    
    try:
        # 1. Veri yükleme
        data_df = load_data_task(artifacts_path)
        
        # 2. Feature engineering
        processed_df = feature_engineer_task(data_df)
        
        # 3. Model eğitimi/yükleme
        model_info = train_or_load_model_task(processed_df, artifacts_path)
        
        # 4. Tahmin üretme
        predictions_df = predict_task(processed_df, model_info, forecast_days)
        
        # 5. Sonuçları kaydetme
        output_summary = save_outputs_task(predictions_df, run_date, artifacts_path)
        
        # Pipeline özeti
        pipeline_summary = {
            'run_date': run_date,
            'forecast_days': forecast_days,
            'execution_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_shape': list(data_df.shape),
            'model_info': {
                'type': 'LightGBM',
                'is_new_model': model_info['is_new_model'],
                'feature_count': len(model_info['feature_cols'])
            },
            'output_summary': output_summary,
            'status': 'SUCCESS'
        }
        
        print("-" * 50)
        print("🎉 PIPELINE BAŞARIYLA TAMAMLANDI!")
        print(f"📊 Toplam tahmin: {output_summary['total_predictions']}")
        print(f"🎯 Ürün sayısı: {output_summary['n_items']}")
        print(f"💰 Beklenen toplam satış: {output_summary['total_expected_sales']:.0f}")
        print(f"📁 CSV: {output_summary['csv_path']}")
        print(f"📈 PNG: {output_summary['png_path']}")
        print(f"⏰ Bitiş zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return pipeline_summary
        
    except Exception as e:
        print("-" * 50)
        print(f"❌ PIPELINE HATASI: {e}")
        
        error_summary = {
            'run_date': run_date,
            'execution_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'error': str(e),
            'status': 'FAILED'
        }
        
        raise

# ================================
# SCHEDULING & MAIN
# ================================

def setup_schedule():
    """
    Prefect schedule setup (eğitim amaçlı gösterim)
    
    Gerçek production için:
    1. prefect deployment build prefect_demand_forecast.py:demand_forecast_flow -n "daily-forecast"
    2. prefect deployment apply demand_forecast_flow-deployment.yaml
    3. prefect agent start -q default
    """
    
    if not PREFECT_AVAILABLE:
        print("⚠️  Prefect mevcut değil, schedule setup atlanıyor")
        return
    
    try:
        from prefect.deployments import Deployment
        from prefect.server.schemas.schedules import CronSchedule
        
        # Cron schedule: Her gün 09:00 Europe/Istanbul
        schedule = CronSchedule(
            cron="0 9 * * *",  # Dakika Saat Gün Ay Haftanın-günü
            timezone="Europe/Istanbul"
        )
        
        deployment = Deployment.build_from_flow(
            flow=demand_forecast_flow,
            name="daily-demand-forecast",
            description="Her sabah 09:00'da çalışan otomatik talep tahmin pipeline'ı",
            schedule=schedule,
            parameters={
                "forecast_days": 7,
                "artifacts_path": "./artifacts"
            },
            tags=["production", "forecasting", "daily"]
        )
        
        print("📅 Prefect Deployment hazırlandı:")
        print(f"   • Schedule: 0 9 * * * (her gün 09:00)")
        print(f"   • Timezone: Europe/Istanbul")
        print(f"   • Name: daily-demand-forecast")
        
        # Deployment'ı apply etmek için:
        # deployment.apply()
        
        return deployment
        
    except Exception as e:
        print(f"⚠️  Schedule setup hatası: {e}")
        return None

def main():
    """Ana çalıştırma fonksiyonu"""
    
    print("🚀 PREFECT DEMAND FORECASTING PIPELINE")
    print("=" * 60)
    
    if not LIGHTGBM_AVAILABLE:
        print("❌ LightGBM gerekli")
        return
    
    print("📋 Pipeline Bilgileri:")
    print("   • Schedule: Her gün 09:00 Europe/Istanbul")
    print("   • Tasks: Load → FE → Model → Predict → Save")
    print("   • Output: CSV + PNG reports")
    print("   • Orchestration: Prefect Tasks & Flows")
    
    if PREFECT_AVAILABLE:
        print("   • Prefect: ✅ Aktif")
    else:
        print("   • Prefect: ⚠️  Mock mode")
    
    print("\n💡 Production Deployment:")
    print("   1. prefect deployment build prefect_demand_forecast.py:demand_forecast_flow -n daily-forecast")
    print("   2. prefect deployment apply demand_forecast_flow-deployment.yaml") 
    print("   3. prefect agent start -q default")
    print("   4. Prefect UI: http://localhost:4200")
    
    print("\n" + "=" * 60)
    
    try:
        # Schedule setup (eğitim amaçlı)
        deployment = setup_schedule()
        
        # Test çalışması
        print("\n🧪 Test çalışması başlatılıyor...")
        result = demand_forecast_flow(
            run_date=datetime.now().strftime('%Y-%m-%d'),
            forecast_days=7
        )
        
        print(f"\n✅ Test başarılı!")
        print(f"📊 Status: {result['status']}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Kullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
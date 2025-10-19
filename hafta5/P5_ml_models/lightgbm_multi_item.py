#!/usr/bin/env python3
"""
LightGBM Çok Ürünlü Satış Tahmini

Bu script çok ürünlü (multi-item) forecasting için LightGBM kullanır.
Lag ve rolling özelliklerle 28 günlük iteratif tahmin yapar.

Yaklaşım:
- Tüm ürünler tek modelde (cross-product learning)
- LabelEncoder ile kategorik encoding
- Iteratif forecasting (basit yaklaşım, eğitim amaçlı)

Production Notları:
- Bu basit iteratif yaklaşım eğitim amaçlıdır
- Daha gelişmiş için: backtesting, cross-validation, ensemble
- Büyük ölçekte: distributed training, feature store

Kullanım: python lightgbm_multi_item.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import pickle
from datetime import datetime, timedelta

# LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("❌ LightGBM kütüphanesi bulunamadı. 'pip install lightgbm' ile kurun.")
    LIGHTGBM_AVAILABLE = False

# Preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

class LightGBMMultiItemForecaster:
    """
    LightGBM ile çok ürünlü forecasting sınıfı
    
    Özellikler:
    - Tüm ürünler tek modelde (shared learning)
    - Lag ve rolling feature'lar
    - Kategorik encoding (LabelEncoder)
    - Iteratif 28 günlük tahmin
    """
    
    def __init__(self, artifacts_path='/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta5/artifacts'):
        self.artifacts_path = artifacts_path
        self.train_df = None
        self.valid_df = None
        self.model = None
        self.label_encoders = {}
        self.feature_cols = []
        self.metrics = {}
        self.feature_importance = None
        
        # Çıktı klasörlerini oluştur
        os.makedirs(f'{artifacts_path}/models', exist_ok=True)
        os.makedirs(f'{artifacts_path}/preds', exist_ok=True)
        os.makedirs(f'{artifacts_path}/figures', exist_ok=True)
        
        print("🚀 LightGBM Çok Ürünlü Forecasting")
        print("💡 Avantajlar: Hızlı, cross-product learning, feature importance")
        print("⚠️  Not: Basit iteratif yaklaşım - prod için daha gelişmiş backtesting gerekir")
        print("=" * 75)
    
    def load_feature_data(self):
        """Feature engineered verileri yükle"""
        
        print("\n📁 1. Feature engineered veriler yükleniyor...")
        
        try:
            # Parquet dosyalarını yükle
            train_path = f'{self.artifacts_path}/datasets/fe_train.parquet'
            valid_path = f'{self.artifacts_path}/datasets/fe_valid.parquet'
            
            self.train_df = pd.read_parquet(train_path)
            self.valid_df = pd.read_parquet(valid_path)
            
            print(f"   ✓ Train data: {self.train_df.shape}")
            print(f"   ✓ Valid data: {self.valid_df.shape}")
            
            # Sütun bilgileri
            print(f"   • Sütunlar: {list(self.train_df.columns)}")
            
            # Ürün sayıları
            train_items = self.train_df['item_id'].nunique()
            valid_items = self.valid_df['item_id'].nunique()
            print(f"   • Train ürün sayısı: {train_items}")
            print(f"   • Valid ürün sayısı: {valid_items}")
            
            # Tarih aralıkları
            print(f"   • Train tarih: {self.train_df.index.min()} - {self.train_df.index.max()}")
            print(f"   • Valid tarih: {self.valid_df.index.min()} - {self.valid_df.index.max()}")
            
        except FileNotFoundError as e:
            print(f"   ❌ Dosya bulunamadı: {e}")
            print("   💡 Önce feature_engineering.py çalıştırın")
            raise
        except Exception as e:
            print(f"   ❌ Veri yükleme hatası: {e}")
            raise
    
    def encode_categorical_features(self):
        """Kategorik özellikleri encode et"""
        
        print("\n🏷️  2. Kategorik özellikler encode ediliyor...")
        
        categorical_cols = ['item_id', 'store_id']
        
        for col in categorical_cols:
            if col in self.train_df.columns:
                print(f"   • {col} encode ediliyor...")
                
                # LabelEncoder oluştur ve train ile fit et
                le = LabelEncoder()
                
                # Train verisi ile fit
                self.train_df[f'{col}_encoded'] = le.fit_transform(self.train_df[col])
                
                # Valid verisi transform (unseen values için handling)
                try:
                    self.valid_df[f'{col}_encoded'] = le.transform(self.valid_df[col])
                except ValueError as e:
                    print(f"     ⚠️  {col} için unseen values var, 0 ile doldurulacak")
                    # Unseen values'ları handle et
                    valid_encoded = []
                    for val in self.valid_df[col]:
                        if val in le.classes_:
                            valid_encoded.append(le.transform([val])[0])
                        else:
                            valid_encoded.append(0)  # Unseen = 0
                    self.valid_df[f'{col}_encoded'] = valid_encoded
                
                # Encoder'ı sakla
                self.label_encoders[col] = le
                
                print(f"     - Train unique: {self.train_df[f'{col}_encoded'].nunique()}")
                print(f"     - Valid unique: {self.valid_df[f'{col}_encoded'].nunique()}")
        
        print(f"   ✓ {len(categorical_cols)} kategorik özellik encode edildi")
    
    def prepare_features_target(self):
        """Özellik ve hedef değişkenleri hazırla"""
        
        print("\n🎯 3. Özellik ve hedef değişkenler hazırlanıyor...")
        
        # Hedef değişken
        target_col = 'sales'
        
        # Özellik sütunları
        # Lag ve rolling feature'lar
        lag_cols = [col for col in self.train_df.columns if 'lag_' in col]
        roll_cols = [col for col in self.train_df.columns if 'roll_' in col]
        
        # Tarih özellikleri
        date_cols = ['dow', 'dom', 'weekofyear', 'month']
        
        # Encoded kategorik özellikler
        encoded_cols = [col for col in self.train_df.columns if '_encoded' in col]
        
        # Tüm feature sütunları
        self.feature_cols = lag_cols + roll_cols + date_cols + encoded_cols
        
        print(f"   📊 Özellik Grupları:")
        print(f"   • Lag özellikleri: {len(lag_cols)} -> {lag_cols}")
        print(f"   • Rolling özellikleri: {len(roll_cols)} -> {roll_cols}")
        print(f"   • Tarih özellikleri: {len(date_cols)} -> {date_cols}")
        print(f"   • Kategorik özellikleri: {len(encoded_cols)} -> {encoded_cols}")
        print(f"   • Toplam özellik sayısı: {len(self.feature_cols)}")
        
        # Train/Valid ayırma
        X_train = self.train_df[self.feature_cols].copy()
        y_train = self.train_df[target_col].copy()
        
        X_valid = self.valid_df[self.feature_cols].copy()
        y_valid = self.valid_df[target_col].copy()
        
        print(f"   ✓ X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"   ✓ X_valid: {X_valid.shape}, y_valid: {y_valid.shape}")
        
        # NaN kontrol
        train_nans = X_train.isnull().sum().sum()
        valid_nans = X_valid.isnull().sum().sum()
        
        if train_nans > 0 or valid_nans > 0:
            print(f"   ⚠️  NaN değerler: Train={train_nans}, Valid={valid_nans}")
            X_train = X_train.fillna(0)
            X_valid = X_valid.fillna(0)
            print(f"   • NaN değerler 0 ile dolduruldu")
        
        return X_train, y_train, X_valid, y_valid
    
    def train_lightgbm_model(self, X_train, y_train, X_valid, y_valid):
        """LightGBM modelini eğit"""
        
        print("\n🌟 4. LightGBM modeli eğitiliyor...")
        
        try:
            # LightGBM parametreleri
            lgb_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
            
            print(f"   📋 Model Parametreleri:")
            for key, value in lgb_params.items():
                print(f"   • {key}: {value}")
            
            # Dataset'leri oluştur
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
            
            print(f"   • LightGBM eğitimi başlıyor...")
            
            # Modeli eğit
            self.model = lgb.train(
                lgb_params,
                train_data,
                valid_sets=[train_data, valid_data],
                valid_names=['train', 'valid'],
                num_boost_round=500,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=100)
                ]
            )
            
            print(f"   ✓ Model eğitimi tamamlandı")
            print(f"   • En iyi iterasyon: {self.model.best_iteration}")
            print(f"   • Train RMSE: {self.model.best_score['train']['rmse']:.4f}")
            print(f"   • Valid RMSE: {self.model.best_score['valid']['rmse']:.4f}")
            
        except Exception as e:
            print(f"   ❌ LightGBM eğitimi hatası: {e}")
            raise
    
    def calculate_validation_metrics(self, X_valid, y_valid):
        """Validation metrikleri hesapla"""
        
        print("\n📊 5. Validation metrikleri hesaplanıyor...")
        
        # Tahmin yap
        y_pred = self.model.predict(X_valid, num_iteration=self.model.best_iteration)
        
        # Negatif değerleri 0 yap
        y_pred = np.maximum(y_pred, 0)
        
        # Metrikler
        mae = mean_absolute_error(y_valid, y_pred)
        rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
        
        # MAPE
        mask = y_valid != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_valid[mask] - y_pred[mask]) / y_valid[mask])) * 100
        else:
            mape = float('inf')
        
        # sMAPE
        denominator = (np.abs(y_valid) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if mask.sum() > 0:
            smape = np.mean(np.abs(y_valid[mask] - y_pred[mask]) / denominator[mask]) * 100
        else:
            smape = float('inf')
        
        self.metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'sMAPE': smape,
            'R2': 1 - (np.sum((y_valid - y_pred) ** 2) / np.sum((y_valid - np.mean(y_valid)) ** 2))
        }
        
        print(f"   📈 LightGBM Validation Performansı:")
        print(f"   • MAE:   {mae:.2f}")
        print(f"   • RMSE:  {rmse:.2f}")
        print(f"   • MAPE:  {mape:.2f}%")
        print(f"   • sMAPE: {smape:.2f}%")
        print(f"   • R²:    {self.metrics['R2']:.4f}")
        
        return y_pred
    
    def create_feature_importance_plot(self):
        """Feature importance grafiği oluştur"""
        
        print("\n📊 6. Feature importance grafiği oluşturuluyor...")
        
        # Feature importance al (gain)
        importance_gain = self.model.feature_importance(importance_type='gain')
        feature_names = self.feature_cols
        
        # DataFrame oluştur
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_gain
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance_df
        
        # Grafik
        plt.figure(figsize=(12, 8))
        
        # Top 15 feature göster
        top_features = importance_df.head(15)
        
        bars = plt.barh(top_features['feature'], top_features['importance'], 
                       color='lightblue', alpha=0.8)
        
        # Değerleri bara yazdır
        for bar, value in zip(bars, top_features['importance']):
            plt.text(bar.get_width() + max(top_features['importance']) * 0.01, 
                    bar.get_y() + bar.get_height()/2,
                    f'{value:.0f}', ha='left', va='center', fontweight='bold')
        
        plt.title('LightGBM Feature Importance (Gain)', fontweight='bold', fontsize=16)
        plt.xlabel('Importance (Gain)', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Kaydet
        importance_path = f'{self.artifacts_path}/figures/lgbm_feature_importance.png'
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Feature importance grafiği: {importance_path}")
        plt.close()
        
        # Top 10 yazdır
        print(f"   🏆 En Önemli 10 Feature:")
        for i, (_, row) in enumerate(top_features.head(10).iterrows(), 1):
            print(f"     {i:2d}. {row['feature']:<20}: {row['importance']:6.0f}")
    
    def iterative_forecast(self, forecast_steps=28):
        """Iteratif 28 günlük tahmin"""
        
        print(f"\n🔮 7. {forecast_steps} günlük iteratif tahmin yapılıyor...")
        print(f"   ⚠️  Not: Bu basit iteratif yaklaşımdır - prod için daha gelişmiş backtesting gerekir")
        
        try:
            # Son tarih ve başlangıç
            last_date = self.valid_df.index.max()
            forecast_start = last_date + timedelta(days=1)
            
            print(f"   • Son veri tarihi: {last_date}")
            print(f"   • Tahmin başlangıcı: {forecast_start}")
            
            # Tüm ürünler için tahmin
            all_forecasts = []
            
            # Her ürün için ayrı ayrı tahmin (iteratif update için)
            unique_items = self.valid_df['item_id'].unique()
            
            for item_id in unique_items:
                print(f"   • {item_id} için tahmin yapılıyor...")
                
                # Son durumu al (valid son satır)
                item_valid = self.valid_df[self.valid_df['item_id'] == item_id].copy()
                
                if len(item_valid) == 0:
                    print(f"     ⚠️  {item_id} için valid veri bulunamadı")
                    continue
                
                # Son satırı al (feature template olarak)
                last_row = item_valid.iloc[-1].copy()
                
                # Bu ürün için iteratif tahmin
                item_forecasts = []
                current_features = last_row.copy()
                
                for step in range(forecast_steps):
                    forecast_date = forecast_start + timedelta(days=step)
                    
                    # Tarih özelliklerini güncelle
                    current_features['dow'] = forecast_date.weekday()
                    current_features['dom'] = forecast_date.day
                    current_features['weekofyear'] = forecast_date.isocalendar()[1]
                    current_features['month'] = forecast_date.month
                    
                    # Tahmin yap
                    X_pred = current_features[self.feature_cols].values.reshape(1, -1)
                    y_pred = self.model.predict(X_pred, num_iteration=self.model.best_iteration)[0]
                    y_pred = max(0, y_pred)  # Negatif değerleri 0 yap
                    
                    # Sonucu kaydet
                    item_forecasts.append({
                        'date': forecast_date,
                        'item_id': item_id,
                        'store_id': last_row['store_id'],
                        'y_pred': y_pred
                    })
                    
                    # Lag feature'ları güncelle (basit yaklaşım)
                    # Bu gerçek prod için çok basit - daha sofistike lag update gerekir
                    if 'lag_1' in current_features:
                        # Shift lag values
                        if 'lag_28' in current_features:
                            # Bu çok basit bir yaklaşım - real-world'de daha karmaşık olmalı
                            pass  # Şimdilik lag'leri güncelleme
                
                all_forecasts.extend(item_forecasts)
            
            # DataFrame'e çevir
            forecast_df = pd.DataFrame(all_forecasts)
            
            print(f"   ✓ {len(forecast_df)} tahmin üretildi")
            print(f"   • Ürün sayısı: {forecast_df['item_id'].nunique()}")
            print(f"   • Tarih aralığı: {forecast_df['date'].min()} - {forecast_df['date'].max()}")
            print(f"   • Ortalama tahmin: {forecast_df['y_pred'].mean():.2f}")
            
            return forecast_df
            
        except Exception as e:
            print(f"   ❌ Iteratif tahmin hatası: {e}")
            raise
    
    def save_results(self, forecast_df):
        """Sonuçları kaydet"""
        
        print("\n💾 8. Sonuçlar kaydediliyor...")
        
        try:
            # 1. Model kaydet
            model_path = f'{self.artifacts_path}/models/lgbm.pkl'
            
            model_data = {
                'model': self.model,
                'feature_cols': self.feature_cols,
                'label_encoders': self.label_encoders,
                'metrics': self.metrics,
                'feature_importance': self.feature_importance,
                'model_params': {
                    'best_iteration': self.model.best_iteration,
                    'best_score': self.model.best_score
                }
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"   ✓ Model: {model_path}")
            
            # 2. Tahminleri kaydet
            pred_path = f'{self.artifacts_path}/preds/lgbm_forecast_all.csv'
            forecast_df.to_csv(pred_path, index=False)
            print(f"   ✓ Tahminler: {pred_path}")
            
            # 3. Özet rapor
            import json
            
            report = {
                'model_type': 'LightGBM',
                'training_date': datetime.now().isoformat(),
                'data_info': {
                    'train_shape': list(self.train_df.shape),
                    'valid_shape': list(self.valid_df.shape),
                    'n_items': self.train_df['item_id'].nunique(),
                    'feature_count': len(self.feature_cols)
                },
                'model_performance': self.metrics,
                'top_features': self.feature_importance.head(10).to_dict('records') if self.feature_importance is not None else [],
                'forecast_info': {
                    'forecast_steps': 28,
                    'forecast_items': int(forecast_df['item_id'].nunique()),
                    'total_predictions': len(forecast_df)
                }
            }
            
            report_path = f'{self.artifacts_path}/preds/lgbm_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"   ✓ Rapor: {report_path}")
            
        except Exception as e:
            print(f"   ❌ Sonuç kaydetme hatası: {e}")
            raise
    
    def run_full_pipeline(self):
        """Tam LightGBM pipeline'ını çalıştır"""
        
        try:
            # 1. Veri yükleme
            self.load_feature_data()
            
            # 2. Kategorik encoding
            self.encode_categorical_features()
            
            # 3. Feature/target hazırlama
            X_train, y_train, X_valid, y_valid = self.prepare_features_target()
            
            # 4. Model eğitimi
            self.train_lightgbm_model(X_train, y_train, X_valid, y_valid)
            
            # 5. Validation metrikleri
            y_pred_valid = self.calculate_validation_metrics(X_valid, y_valid)
            
            # 6. Feature importance
            self.create_feature_importance_plot()
            
            # 7. İteratif tahmin
            forecast_df = self.iterative_forecast()
            
            # 8. Sonuçları kaydet
            self.save_results(forecast_df)
            
            print(f"\n🎉 LightGBM Multi-Item Forecasting tamamlandı!")
            print(f"🚀 Model: LightGBM Regressor")
            print(f"📊 Valid sMAPE: {self.metrics['sMAPE']:.2f}%")
            print(f"📈 R²: {self.metrics['R2']:.4f}")
            print(f"🔮 Tahmin: {len(forecast_df)} adet (28 gün x {forecast_df['item_id'].nunique()} ürün)")
            print(f"📁 Çıktılar: {self.artifacts_path}/")
            
            return self.model, forecast_df, self.metrics
            
        except Exception as e:
            print(f"\n❌ LightGBM Pipeline hatası: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Ana çalıştırma fonksiyonu"""
    
    if not LIGHTGBM_AVAILABLE:
        print("❌ LightGBM kütüphanesi gerekli. 'pip install lightgbm' ile kurun.")
        return
    
    print("=" * 75)
    print("LIGHTGBM ÇOK ÜRÜNLÜ SATIŞ TAHMİNİ")
    print("🚀 Avantajlar: Hızlı, feature importance, cross-product learning")
    print("⚠️  Bu basit iteratif yaklaşım - prod için daha gelişmiş backtesting gerekir")
    print("=" * 75)
    
    try:
        # LightGBM forecaster'ı başlat
        forecaster = LightGBMMultiItemForecaster()
        
        # Tam pipeline'ı çalıştır
        model, forecast, metrics = forecaster.run_full_pipeline()
        
        print(f"\n✅ İşlem başarıyla tamamlandı!")
        print(f"\n💡 LightGBM Avantajları:")
        print(f"   • Hızlı eğitim ve tahmin")
        print(f"   • Otomatik feature importance")
        print(f"   • Kategorik feature desteği")
        print(f"   • Cross-product learning (ürünler arası pattern)")
        print(f"\n⚠️  Production İyileştirmeleri:")
        print(f"   • Daha sofistike iteratif lag update")
        print(f"   • Cross-validation ve backtesting")
        print(f"   • Hyperparameter tuning")
        print(f"   • Ensemble models")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Kullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"\n💥 Beklenmeyen hata: {e}")

if __name__ == "__main__":
    main()
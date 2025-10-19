#!/usr/bin/env python3
"""
Prophet Tek Ürün Satış Tahmini ve ARIMA Karşılaştırması

Prophet Avantajları:
- Kolay kullanım: Minimal kod ile güçlü sonuçlar
- Otomatik sezonluk yakalama: Günlük, haftalık, yıllık pattern'ler
- Tatil desteği: Özel günlerin etkisini modelleme
- Missing data toleransı: Eksik verileri iyi handle eder
- Trend değişim noktaları: Otomatik trend kırılma tespiti
- Uncertainty intervals: Tahmin güven aralıkları

ARIMA vs Prophet:
- ARIMA: Geleneksel, matematiksel güçlü, stationary gerektirir
- Prophet: Modern, pratik, robust, business-friendly

Bu script:
1. Aynı ürün için Prophet modeli eğitir
2. ARIMA ile performans karşılaştırması yapar
3. Her iki modelin avantaj/dezavantajlarını gösterir

Kullanım: python prophet_single_item.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import json
import pickle
from datetime import datetime, timedelta

# Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("❌ Prophet kütüphanesi bulunamadı. 'pip install prophet' ile kurun.")
    PROPHET_AVAILABLE = False

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

class ProphetSingleItemForecaster:
    """
    Prophet ile tek ürün forecasting ve ARIMA karşılaştırması
    
    Prophet'in güçlü yanları:
    - Kolay parametre ayarı, otomatik sezonluk
    - Tatil efektleri ekleyebilme
    - Missing data'ya dayanıklılık
    - Business-friendly interpretability
    """
    
    def __init__(self, item_id=None, artifacts_path='/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta5/artifacts'):
        self.artifacts_path = artifacts_path
        self.item_id = item_id
        self.train_series = None
        self.valid_series = None
        self.prophet_model = None
        self.forecast_df = None
        self.metrics = {}
        self.arima_metrics = {}
        
        # Çıktı klasörlerini oluştur
        os.makedirs(f'{artifacts_path}/models', exist_ok=True)
        os.makedirs(f'{artifacts_path}/preds', exist_ok=True)
        os.makedirs(f'{artifacts_path}/figures', exist_ok=True)
        
        print("🔮 Prophet Tek Ürün Forecasting ve ARIMA Karşılaştırması")
        print("💡 Prophet Avantajları: Kolay kullanım, otomatik sezonluk, tatil desteği")
        print("=" * 70)
    
    def load_arima_results(self):
        """ARIMA sonuçlarını yükle (karşılaştırma için)"""
        
        print("\n📊 1. ARIMA sonuçları yükleniyor...")
        
        try:
            # ARIMA modelinden item_id'yi al
            arima_files = [f for f in os.listdir(f'{self.artifacts_path}/models') if f.startswith('arima_')]
            
            if not arima_files:
                print("   ⚠️  ARIMA modeli bulunamadı. Önce arima_single_item.py çalıştırın.")
                return None
            
            # İlk ARIMA modelini yükle
            arima_file = arima_files[0]
            self.item_id = arima_file.replace('arima_', '').replace('.pkl', '')
            
            print(f"   • ARIMA modelinden item_id alınıyor: {self.item_id}")
            
            # ARIMA raporunu yükle
            arima_report_path = f'{self.artifacts_path}/preds/arima_report_{self.item_id}.json'
            with open(arima_report_path, 'r') as f:
                arima_report = json.load(f)
            
            self.arima_metrics = arima_report['metrics']
            
            print(f"   ✓ ARIMA metrikleri yüklendi:")
            print(f"     - MAE: {self.arima_metrics['MAE']:.2f}")
            print(f"     - sMAPE: {self.arima_metrics['sMAPE']:.2f}%")
            
            return arima_report
            
        except Exception as e:
            print(f"   ❌ ARIMA sonuçları yüklenemedi: {e}")
            return None
    
    def load_time_series(self):
        """Aynı ürün için zaman serisi yükle"""
        
        print(f"\n📈 2. {self.item_id} için zaman serisi yükleniyor...")
        
        try:
            # Train ve valid verilerini yükle
            train_df = pd.read_csv('/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta5/artifacts/datasets/train.csv', parse_dates=['date'], index_col='date')
            valid_df = pd.read_csv('/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta5/artifacts/datasets/valid.csv', parse_dates=['date'], index_col='date')
            
            # Seçilen ürün için filtrele
            item_train = train_df[train_df['item_id'] == self.item_id]['sales'].copy()
            item_valid = valid_df[valid_df['item_id'] == self.item_id]['sales'].copy()
            
            self.train_series = item_train
            self.valid_series = item_valid
            
            print(f"   ✓ Train: {len(self.train_series)} gün")
            print(f"   ✓ Valid: {len(self.valid_series)} gün")
            print(f"   • Train ortalama: {self.train_series.mean():.2f}")
            
        except Exception as e:
            print(f"   ❌ Zaman serisi yükleme hatası: {e}")
            raise
    
    def prepare_prophet_data(self):
        """Prophet formatına çevir (ds, y)"""
        
        print(f"\n🔄 3. Prophet formatına çeviriliyor...")
        
        # Prophet için ds (date), y (value) formatı gerekli
        prophet_train = pd.DataFrame({
            'ds': self.train_series.index,
            'y': self.train_series.values
        })
        
        prophet_valid = pd.DataFrame({
            'ds': self.valid_series.index,
            'y': self.valid_series.values
        })
        
        print(f"   ✓ Prophet train data: {len(prophet_train)} satır")
        print(f"   ✓ Prophet valid data: {len(prophet_valid)} satır")
        print(f"   • Tarih aralığı: {prophet_train['ds'].min()} - {prophet_train['ds'].max()}")
        
        return prophet_train, prophet_valid
    
    def train_prophet_model(self, prophet_train_df):
        """Prophet modelini eğit"""
        
        print(f"\n🎯 4. Prophet modeli eğitiliyor...")
        
        try:
            # Prophet model parametreleri
            # daily_seasonality=True: Günlük sezonluk (7 günlük pattern)
            # weekly_seasonality=True: Haftalık sezonluk (hafta içi/sonu)
            # yearly_seasonality=False: Yıllık sezonluk kapalı (çok kısa veri için)
            
            self.prophet_model = Prophet(
                daily_seasonality=True,     # Günlük pattern'ler
                weekly_seasonality=True,    # Haftalık pattern'ler
                yearly_seasonality=False,   # Yıllık pattern kapalı (kısa veri)
                changepoint_prior_scale=0.05,  # Trend değişim hassasiyeti
                seasonality_prior_scale=10.0,  # Sezonluk hassasiyeti
                interval_width=0.95,        # %95 güven aralığı
                n_changepoints=25           # Trend değişim noktası sayısı
            )
            
            print(f"   • Günlük sezonluk: Açık")
            print(f"   • Haftalık sezonluk: Açık") 
            print(f"   • Yıllık sezonluk: Kapalı (kısa veri)")
            print(f"   • Changepoint prior: 0.05 (konservatif)")
            
            # Modeli eğit
            print(f"   • Model eğitiliyor...")
            self.prophet_model.fit(prophet_train_df)
            
            print(f"   ✓ Prophet modeli başarıyla eğitildi")
            
            # Model bileşenlerini yazdır
            if hasattr(self.prophet_model, 'seasonalities'):
                print(f"   📊 Aktif sezonluklar:")
                for name, seasonality in self.prophet_model.seasonalities.items():
                    print(f"     - {name}: period={seasonality['period']}, order={seasonality['fourier_order']}")
            
        except Exception as e:
            print(f"   ❌ Prophet model eğitimi hatası: {e}")
            raise
    
    def make_prophet_forecast(self, forecast_steps=28):
        """Prophet ile 28 gün tahmin yap"""
        
        print(f"\n🔮 5. Prophet ile {forecast_steps} gün tahmin yapılıyor...")
        
        try:
            # Future dataframe oluştur
            future_df = self.prophet_model.make_future_dataframe(periods=forecast_steps, freq='D')
            
            print(f"   • Future dataframe: {len(future_df)} satır")
            print(f"   • Tahmin aralığı: {future_df['ds'].iloc[-forecast_steps]} - {future_df['ds'].iloc[-1]}")
            
            # Tahmin yap
            print(f"   • Tahmin hesaplanıyor...")
            self.forecast_df = self.prophet_model.predict(future_df)
            
            # Son 28 günü al (validation period)
            forecast_period = self.forecast_df.tail(forecast_steps).copy()
            
            # Negatif değerleri 0 yap
            forecast_period['yhat'] = forecast_period['yhat'].clip(lower=0)
            
            print(f"   ✓ {forecast_steps} günlük tahmin tamamlandı")
            print(f"   • Ortalama tahmin: {forecast_period['yhat'].mean():.2f}")
            print(f"   • Min tahmin: {forecast_period['yhat'].min():.2f}")
            print(f"   • Max tahmin: {forecast_period['yhat'].max():.2f}")
            
            # Güven aralığı bilgileri
            ci_width = forecast_period['yhat_upper'].mean() - forecast_period['yhat_lower'].mean()
            print(f"   • Ortalama güven aralığı genişliği: {ci_width:.2f}")
            
            return forecast_period
            
        except Exception as e:
            print(f"   ❌ Prophet tahmin hatası: {e}")
            raise
    
    def calculate_metrics_and_compare(self, forecast_period):
        """Metrikleri hesapla ve ARIMA ile karşılaştır"""
        
        print(f"\n📊 6. Metrikler hesaplanıyor ve ARIMA ile karşılaştırılıyor...")
        
        # Prophet tahminleri
        y_true = self.valid_series.values
        y_pred = forecast_period['yhat'].values[:len(y_true)]
        
        # Metrikler
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAPE
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = float('inf')
        
        # sMAPE
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if mask.sum() > 0:
            smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
        else:
            smape = float('inf')
        
        self.metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'sMAPE': smape,
            'forecast_mean': y_pred.mean(),
            'actual_mean': y_true.mean()
        }
        
        print(f"   📈 Prophet Performansı:")
        print(f"   • MAE:   {mae:.2f}")
        print(f"   • RMSE:  {rmse:.2f}")
        print(f"   • MAPE:  {mape:.2f}%")
        print(f"   • sMAPE: {smape:.2f}%")
        
        # ARIMA ile karşılaştırma
        if self.arima_metrics:
            print(f"\n   🔍 ARIMA vs Prophet Karşılaştırması:")
            print(f"   {'Metrik':<8} {'ARIMA':<10} {'Prophet':<10} {'Kazanan':<10}")
            print(f"   {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
            
            metrics_to_compare = ['MAE', 'RMSE', 'sMAPE']
            for metric in metrics_to_compare:
                arima_val = self.arima_metrics[metric]
                prophet_val = self.metrics[metric]
                
                if metric == 'sMAPE' and (arima_val == float('inf') or prophet_val == float('inf')):
                    winner = "N/A"
                    arima_str = "∞" if arima_val == float('inf') else f"{arima_val:.2f}"
                    prophet_str = "∞" if prophet_val == float('inf') else f"{prophet_val:.2f}"
                else:
                    winner = "Prophet" if prophet_val < arima_val else "ARIMA"
                    arima_str = f"{arima_val:.2f}"
                    prophet_str = f"{prophet_val:.2f}"
                
                print(f"   {metric:<8} {arima_str:<10} {prophet_str:<10} {winner:<10}")
        
        return self.metrics
    
    def create_visualizations(self, forecast_period):
        """Prophet görselleştirmeleri oluştur"""
        
        print(f"\n📊 7. Görselleştirmeler oluşturuluyor...")
        
        # 1. Prophet'in kendi forecast plot'u
        fig1 = self.prophet_model.plot(self.forecast_df, figsize=(15, 8))
        
        # Validation period'u vurgula
        ax = fig1.gca()
        valid_start = self.valid_series.index[0]
        ax.axvline(x=valid_start, color='red', linestyle='--', alpha=0.7, 
                  label='Validation Start')
        ax.legend()
        ax.set_title(f'{self.item_id} - Prophet Forecast with Components', 
                    fontweight='bold', fontsize=16)
        
        # Prophet plot kaydet
        prophet_plot_path = f'{self.artifacts_path}/figures/prophet_{self.item_id}_components.png'
        fig1.savefig(prophet_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"   ✓ Prophet components plot: {prophet_plot_path}")
        
        # 2. Custom karşılaştırma grafiği
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Ana tahmin grafiği
        ax1 = axes[0, 0]
        
        # Son 100 günü göster
        train_plot = self.train_series.tail(100)
        
        ax1.plot(train_plot.index, train_plot.values, 
                label='Train (Gerçek)', color='blue', linewidth=2)
        ax1.plot(self.valid_series.index, self.valid_series.values, 
                label='Valid (Gerçek)', color='green', linewidth=2)
        ax1.plot(forecast_period['ds'], forecast_period['yhat'], 
                label='Prophet Tahmin', color='red', linewidth=2, linestyle='--')
        
        # Güven aralığı
        ax1.fill_between(forecast_period['ds'], 
                        forecast_period['yhat_lower'], 
                        forecast_period['yhat_upper'],
                        color='red', alpha=0.2, label='%95 Güven Aralığı')
        
        ax1.axvline(x=self.train_series.index[-1], color='gray', linestyle=':', alpha=0.7, 
                   label='Train/Valid Split')
        
        ax1.set_title(f'{self.item_id} - Prophet Forecast', fontweight='bold')
        ax1.set_ylabel('Satış')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Model karşılaştırması
        ax2 = axes[0, 1]
        
        if self.arima_metrics:
            models = ['ARIMA', 'Prophet']
            mae_values = [self.arima_metrics['MAE'], self.metrics['MAE']]
            smape_values = [self.arima_metrics['sMAPE'], self.metrics['sMAPE']]
            
            x = np.arange(len(models))
            width = 0.35
            
            bars1 = ax2.bar(x - width/2, mae_values, width, label='MAE', alpha=0.8)
            bars2 = ax2.bar(x + width/2, smape_values, width, label='sMAPE (%)', alpha=0.8)
            
            ax2.set_title('Model Karşılaştırması', fontweight='bold')
            ax2.set_ylabel('Metrik Değeri')
            ax2.set_xticks(x)
            ax2.set_xticklabels(models)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Değerleri bara yazdır
            for bar, value in zip(bars1, mae_values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.1f}', ha='center', va='bottom')
            for bar, value in zip(bars2, smape_values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{value:.1f}', ha='center', va='bottom')
        
        # Prophet bileşenleri (trend)
        ax3 = axes[1, 0]
        ax3.plot(self.forecast_df['ds'], self.forecast_df['trend'], 
                color='orange', linewidth=2)
        ax3.set_title('Prophet Trend Component', fontweight='bold')
        ax3.set_ylabel('Trend')
        ax3.grid(True, alpha=0.3)
        
        # Haftalık sezonluk
        ax4 = axes[1, 1]
        if 'weekly' in self.forecast_df.columns:
            ax4.plot(self.forecast_df['ds'], self.forecast_df['weekly'], 
                    color='purple', linewidth=1, alpha=0.7)
            ax4.set_title('Prophet Weekly Seasonality', fontweight='bold')
            ax4.set_ylabel('Weekly Effect')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Weekly seasonality\ncomponent not available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Weekly Seasonality (N/A)', fontweight='bold')
        
        plt.suptitle(f'Prophet Analysis - {self.item_id}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Kaydet
        main_plot_path = f'{self.artifacts_path}/figures/prophet_{self.item_id}_forecast.png'
        plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Ana karşılaştırma grafiği: {main_plot_path}")
        plt.close()
    
    def save_results(self, forecast_period):
        """Prophet sonuçlarını kaydet"""
        
        print(f"\n💾 8. Prophet sonuçları kaydediliyor...")
        
        # 1. Model bileşenleri ve parametreler (JSON)
        model_info = {
            'item_id': self.item_id,
            'model_type': 'Prophet',
            'parameters': {
                'daily_seasonality': True,
                'weekly_seasonality': True,
                'yearly_seasonality': False,
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'interval_width': 0.95,
                'n_changepoints': 25
            },
            'train_period': f"{self.train_series.index.min()} to {self.train_series.index.max()}",
            'valid_period': f"{self.valid_series.index.min()} to {self.valid_series.index.max()}",
            'forecast_steps': len(forecast_period),
            'metrics': self.metrics,
            'changepoints': self.prophet_model.changepoints.tolist() if hasattr(self.prophet_model, 'changepoints') else [],
            'seasonalities': list(self.prophet_model.seasonalities.keys()) if hasattr(self.prophet_model, 'seasonalities') else []
        }
        
        model_path = f'{self.artifacts_path}/models/prophet_{self.item_id}.json'
        with open(model_path, 'w') as f:
            json.dump(model_info, f, indent=2, default=str)
        print(f"   ✓ Model bilgileri: {model_path}")
        
        # 2. Tahminleri kaydet
        forecast_save_df = pd.DataFrame({
            'date': forecast_period['ds'],
            'item_id': self.item_id,
            'forecast': forecast_period['yhat'],
            'forecast_lower': forecast_period['yhat_lower'],
            'forecast_upper': forecast_period['yhat_upper'],
            'actual': self.valid_series.values[:len(forecast_period)]
        })
        
        pred_path = f'{self.artifacts_path}/preds/prophet_forecast_{self.item_id}.csv'
        forecast_save_df.to_csv(pred_path, index=False)
        print(f"   ✓ Tahminler: {pred_path}")
        
        # 3. Karşılaştırma raporu
        if self.arima_metrics:
            comparison = {
                'item_id': self.item_id,
                'comparison_date': datetime.now().isoformat(),
                'arima_metrics': self.arima_metrics,
                'prophet_metrics': self.metrics,
                'winner_by_metric': {},
                'summary': {
                    'arima_advantages': [
                        "Matematiksel olarak sağlam",
                        "Stationary serilerde güçlü",
                        "Parametre kontrolü yüksek"
                    ],
                    'prophet_advantages': [
                        "Kolay kullanım",
                        "Otomatik sezonluk yakalama",
                        "Tatil efektleri eklenebilir",
                        "Missing data'ya dayanıklı",
                        "Business-friendly yorumlama"
                    ]
                }
            }
            
            # Metrik bazında kazanan
            for metric in ['MAE', 'RMSE', 'sMAPE']:
                if metric in self.arima_metrics and metric in self.metrics:
                    arima_val = self.arima_metrics[metric]
                    prophet_val = self.metrics[metric]
                    
                    if arima_val == float('inf') or prophet_val == float('inf'):
                        comparison['winner_by_metric'][metric] = 'N/A'
                    else:
                        comparison['winner_by_metric'][metric] = 'Prophet' if prophet_val < arima_val else 'ARIMA'
            
            comparison_path = f'{self.artifacts_path}/preds/arima_vs_prophet_{self.item_id}.json'
            with open(comparison_path, 'w') as f:
                json.dump(comparison, f, indent=2, default=str)
            print(f"   ✓ Karşılaştırma raporu: {comparison_path}")
    
    def run_full_pipeline(self):
        """Tam Prophet pipeline'ını çalıştır"""
        
        try:
            # 1. ARIMA sonuçlarını yükle
            arima_report = self.load_arima_results()
            
            if not self.item_id:
                print("❌ Item ID bulunamadı. ARIMA modeli önce çalıştırılmalı.")
                return
            
            # 2. Aynı ürün için zaman serisi yükle
            self.load_time_series()
            
            # 3. Prophet formatına çevir
            prophet_train_df, prophet_valid_df = self.prepare_prophet_data()
            
            # 4. Prophet modelini eğit
            self.train_prophet_model(prophet_train_df)
            
            # 5. Tahmin yap
            forecast_period = self.make_prophet_forecast()
            
            # 6. Metrikleri hesapla ve karşılaştır
            self.calculate_metrics_and_compare(forecast_period)
            
            # 7. Görselleştirmeler
            self.create_visualizations(forecast_period)
            
            # 8. Sonuçları kaydet
            self.save_results(forecast_period)
            
            print(f"\n🎉 Prophet Forecasting ve ARIMA karşılaştırması tamamlandı!")
            print(f"🔮 Prophet Model: {self.item_id}")
            print(f"📈 Prophet sMAPE: {self.metrics['sMAPE']:.2f}%")
            if self.arima_metrics:
                print(f"⚖️  ARIMA sMAPE: {self.arima_metrics['sMAPE']:.2f}%")
                better = "Prophet" if self.metrics['sMAPE'] < self.arima_metrics['sMAPE'] else "ARIMA"
                print(f"🏆 Kazanan (sMAPE): {better}")
            print(f"📁 Çıktılar: {self.artifacts_path}/")
            
            return self.prophet_model, forecast_period, self.metrics
            
        except Exception as e:
            print(f"\n❌ Prophet Pipeline hatası: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Ana çalıştırma fonksiyonu"""
    
    if not PROPHET_AVAILABLE:
        print("❌ Prophet kütüphanesi gerekli. 'pip install prophet' ile kurun.")
        return
    
    print("=" * 70)
    print("PROPHET TEK ÜRÜN SATIŞ TAHMİNİ ve ARIMA KARŞILAŞTIRMASI")
    print("💡 Prophet: Kolay kullanım + Otomatik sezonluk + Tatil desteği")
    print("=" * 70)
    
    try:
        # Prophet forecaster'ı başlat
        forecaster = ProphetSingleItemForecaster()
        
        # Tam pipeline'ı çalıştır
        model, forecast, metrics = forecaster.run_full_pipeline()
        
        print(f"\n✅ İşlem başarıyla tamamlandı!")
        print(f"\n💡 Prophet vs ARIMA Özet:")
        print(f"   • ARIMA: Geleneksel, matematiksel güçlü")
        print(f"   • Prophet: Modern, pratik, kolay kullanım")
        print(f"   • Prophet Avantajları: Sezonluk, tatil, missing data tolerance")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Kullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"\n💥 Beklenmeyen hata: {e}")

if __name__ == "__main__":
    main()
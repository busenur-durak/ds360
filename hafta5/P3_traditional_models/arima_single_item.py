#!/usr/bin/env python3
"""
ARIMA Tek Ürün Satış Tahmini

ÖNEMLİ NOT: ARIMA tek seri için tasarlanmıştır!
- Çok ürünlü forecasting için yavaş ve scale etmesi zor
- Paralel işleme gerektirir veya farklı yaklaşımlar kullanılmalı
- Bu örnek eğitim amaçlıdır, production'da dikkatli kullanın

Bu script:
1. En yüksek satışlı tek ürünü seçer
2. ARIMA(p,d,q) parametrelerini grid search ile bulur
3. Stationarity test'leri uygular (ADF test)
4. 28 gün tahmin yapar
5. Sonuçları görselleştirir ve kaydeder

Kullanım: python arima_single_item.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import pickle
from itertools import product
from datetime import datetime, timedelta

# Statsmodels
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

class ARIMASingleItemForecaster:
    """
    Tek ürün için ARIMA forecasting sınıfı
    
    Bu sınıf tek bir zaman serisi için ARIMA modeli eğitir.
    Çok ürünlü forecasting için uygun değildir (yavaş, memory issues).
    """
    
    def __init__(self, artifacts_path='./artifacts'):
        self.artifacts_path = artifacts_path
        self.item_id = None
        self.ts_data = None
        self.train_series = None
        self.valid_series = None
        self.model = None
        self.forecast = None
        self.best_params = None
        self.metrics = {}
        
        # Çıktı klasörlerini oluştur
        os.makedirs(f'{artifacts_path}/models', exist_ok=True)
        os.makedirs(f'{artifacts_path}/preds', exist_ok=True)
        os.makedirs(f'{artifacts_path}/figures', exist_ok=True)
        
        print("🔮 ARIMA Tek Ürün Forecasting")
        print("⚠️  NOT: ARIMA tek seri içindir; çok ürünlü için yavaş/scale zor")
        print("=" * 65)
    
    def select_top_item(self):
        """En yüksek satışlı ürünü seç"""
        
        print("\n📊 1. En yüksek satışlı ürün seçiliyor...")
        
        try:
            # Train verisi yükle
            train_df = pd.read_csv('/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta5/artifacts/datasets/train.csv', parse_dates=['date'], index_col='date')
            
            # Ürün bazında toplam satış hesapla
            item_totals = train_df.groupby('item_id')['sales'].sum().sort_values(ascending=False)
            
            print("   • Ürün bazında toplam satışlar:")
            for i, (item, total) in enumerate(item_totals.items(), 1):
                print(f"     {i}. {item}: {total:,} satış")
            
            # En yüksek satışlı ürünü seç
            self.item_id = item_totals.index[0]
            top_sales = item_totals.iloc[0]
            
            print(f"\n   ✓ Seçilen ürün: {self.item_id}")
            print(f"   ✓ Toplam satış: {top_sales:,}")
            
        except FileNotFoundError:
            print("   ❌ Train verisi bulunamadı. Önce create_m5_subset.py çalıştırın.")
            raise
        
        return self.item_id
    
    def load_time_series(self):
        """Seçilen ürün için zaman serisi yükle"""
        
        print(f"\n📈 2. {self.item_id} için zaman serisi yükleniyor...")
        
        try:
            # Train ve valid verilerini yükle
            train_df = pd.read_csv('/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta5/artifacts/datasets/train.csv', parse_dates=['date'], index_col='date')
            valid_df = pd.read_csv('/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta5/artifacts/datasets/valid.csv', parse_dates=['date'], index_col='date')
            
            # Seçilen ürün için filtrele
            item_train = train_df[train_df['item_id'] == self.item_id]['sales'].copy()
            item_valid = valid_df[valid_df['item_id'] == self.item_id]['sales'].copy()
            
            print(f"   • Train dönem: {item_train.index.min()} - {item_train.index.max()}")
            print(f"   • Valid dönem: {item_valid.index.min()} - {item_valid.index.max()}")
            print(f"   • Train gün sayısı: {len(item_train)}")
            print(f"   • Valid gün sayısı: {len(item_valid)}")
            
            # Serilerimizi sakla
            self.train_series = item_train
            self.valid_series = item_valid
            
            # Tam seriyi birleştir (görselleştirme için)
            self.ts_data = pd.concat([item_train, item_valid])
            
            print(f"   ✓ Toplam {len(self.ts_data)} günlük veri yüklendi")
            
            # Temel istatistikler
            print(f"\n   📈 Temel İstatistikler:")
            print(f"   • Ortalama: {self.train_series.mean():.2f}")
            print(f"   • Std: {self.train_series.std():.2f}")
            print(f"   • Min: {self.train_series.min()}")
            print(f"   • Max: {self.train_series.max()}")
            print(f"   • Sıfır gün sayısı: {(self.train_series == 0).sum()}")
            
        except Exception as e:
            print(f"   ❌ Zaman serisi yükleme hatası: {e}")
            raise
    
    def test_stationarity(self, series, title="Series"):
        """Stationarity test (ADF test)"""
        
        print(f"\n🔍 3. Stationarity testi yapılıyor ({title})...")
        
        # Augmented Dickey-Fuller test
        adf_result = adfuller(series.dropna())
        
        print(f"   • ADF Statistic: {adf_result[0]:.6f}")
        print(f"   • p-value: {adf_result[1]:.6f}")
        print(f"   • Critical Values:")
        for key, value in adf_result[4].items():
            print(f"     - {key}: {value:.3f}")
        
        # Sonuç yorumu
        if adf_result[1] <= 0.05:
            print(f"   ✓ Seri durağan (stationary) - p < 0.05")
            is_stationary = True
        else:
            print(f"   ⚠️  Seri durağan değil (non-stationary) - p > 0.05")
            is_stationary = False
        
        return is_stationary, adf_result[1]
    
    def determine_d_parameter(self):
        """Differencing order (d) parametresini belirle"""
        
        print(f"\n🔧 4. ARIMA differencing (d) parametresi belirleniyor...")
        
        series = self.train_series.copy()
        d = 0
        max_d = 2  # Maksimum 2 kez differencing
        
        # Original serinin stationarity'sini test et
        is_stationary, p_value = self.test_stationarity(series, f"Original (d={d})")
        
        if is_stationary:
            print(f"   ✓ d={d} (differencing gerekmiyor)")
            return d
        
        # Differencing dene
        for d in range(1, max_d + 1):
            # Differencing uygula
            diff_series = series.diff(d).dropna()
            
            if len(diff_series) < 50:  # Çok az veri kaldıysa dur
                print(f"   ⚠️  d={d} için çok az veri kalıyor, d={d-1} kullanılacak")
                return d - 1
            
            is_stationary, p_value = self.test_stationarity(diff_series, f"Differenced (d={d})")
            
            if is_stationary:
                print(f"   ✓ d={d} seçildi")
                return d
        
        # Hiçbiri stationary değilse, d=1 kullan
        print(f"   ⚠️  Optimal d bulunamadı, d=1 kullanılacak")
        return 1
    
    def grid_search_arima(self, max_p=2, max_q=2):
        """ARIMA(p,d,q) parametreleri için grid search"""
        
        print(f"\n🔍 5. ARIMA grid search yapılıyor...")
        
        # d parametresini belirle
        d = self.determine_d_parameter()
        
        # Grid search parametreleri
        p_values = range(0, max_p + 1)
        q_values = range(0, max_q + 1)
        
        print(f"   • p değerleri: {list(p_values)}")
        print(f"   • d değeri: {d}")
        print(f"   • q değerleri: {list(q_values)}")
        
        best_aic = float('inf')
        best_params = None
        results = []
        
        total_combinations = len(p_values) * len(q_values)
        current_combination = 0
        
        for p in p_values:
            for q in q_values:
                current_combination += 1
                
                try:
                    print(f"   • ARIMA({p},{d},{q}) deneniyor... ({current_combination}/{total_combinations})")
                    
                    # ARIMA modelini fit et
                    model = ARIMA(self.train_series, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    aic = fitted_model.aic
                    bic = fitted_model.bic
                    
                    results.append({
                        'p': p, 'd': d, 'q': q,
                        'AIC': aic, 'BIC': bic,
                        'converged': fitted_model.mle_retvals['converged'] if hasattr(fitted_model, 'mle_retvals') else True
                    })
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)
                        print(f"     → Yeni en iyi: AIC={aic:.2f}")
                    else:
                        print(f"     → AIC={aic:.2f}")
                    
                except Exception as e:
                    print(f"     → Hata: {str(e)[:50]}...")
                    results.append({
                        'p': p, 'd': d, 'q': q,
                        'AIC': np.inf, 'BIC': np.inf,
                        'converged': False
                    })
        
        # Sonuçları dataframe'e çevir
        results_df = pd.DataFrame(results)
        
        print(f"\n   📊 Grid Search Sonuçları:")
        print(f"   • En iyi parametreler: ARIMA{best_params}")
        print(f"   • En iyi AIC: {best_aic:.2f}")
        
        # Top 3 sonucu göster
        top_results = results_df[results_df['converged']].nsmallest(3, 'AIC')
        print(f"\n   🏆 En İyi 3 Model:")
        for i, (_, row) in enumerate(top_results.iterrows(), 1):
            print(f"     {i}. ARIMA({int(row['p'])},{int(row['d'])},{int(row['q'])}) - AIC: {row['AIC']:.2f}")
        
        self.best_params = best_params
        return best_params, results_df
    
    def train_arima_model(self):
        """En iyi parametrelerle ARIMA modelini eğit"""
        
        print(f"\n🎯 6. ARIMA{self.best_params} modelini eğitiliyor...")
        
        try:
            # Modeli eğit
            model = ARIMA(self.train_series, order=self.best_params)
            self.model = model.fit()
            
            print(f"   ✓ Model başarıyla eğitildi")
            print(f"   • AIC: {self.model.aic:.2f}")
            print(f"   • BIC: {self.model.bic:.2f}")
            print(f"   • Log Likelihood: {self.model.llf:.2f}")
            
            # Model özeti (kısaca)
            print(f"\n   📋 Model Parametreleri:")
            if hasattr(self.model, 'params'):
                for param_name, param_value in self.model.params.items():
                    print(f"   • {param_name}: {param_value:.4f}")
            
        except Exception as e:
            print(f"   ❌ Model eğitimi hatası: {e}")
            raise
    
    def make_forecast(self, forecast_steps=28):
        """28 gün tahmin yap"""
        
        print(f"\n🔮 7. {forecast_steps} gün tahmin yapılıyor...")
        
        try:
            # Tahmin yap
            forecast_result = self.model.forecast(steps=forecast_steps)
            
            # Confidence interval'ı da al (eğer varsa)
            try:
                forecast_ci = self.model.get_forecast(steps=forecast_steps).conf_int()
                forecast_lower = forecast_ci.iloc[:, 0]
                forecast_upper = forecast_ci.iloc[:, 1]
            except:
                forecast_lower = None
                forecast_upper = None
            
            # Tahmin tarihlerini oluştur
            last_date = self.train_series.index[-1]
            forecast_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                         periods=forecast_steps, freq='D')
            
            # Tahmin serisini oluştur
            self.forecast = pd.Series(forecast_result, index=forecast_dates)
            
            # Negatif değerleri 0 yap (satış negatif olamaz)
            self.forecast = self.forecast.clip(lower=0)
            
            print(f"   ✓ {forecast_steps} günlük tahmin tamamlandı")
            print(f"   • Tahmin aralığı: {self.forecast.index.min()} - {self.forecast.index.max()}")
            print(f"   • Ortalama tahmin: {self.forecast.mean():.2f}")
            print(f"   • Min tahmin: {self.forecast.min():.2f}")
            print(f"   • Max tahmin: {self.forecast.max():.2f}")
            
            return self.forecast, forecast_lower, forecast_upper
            
        except Exception as e:
            print(f"   ❌ Tahmin hatası: {e}")
            raise
    
    def calculate_metrics(self):
        """Tahmin metriklerini hesapla"""
        
        print(f"\n📊 8. Tahmin metrikleri hesaplanıyor...")
        
        # Gerçek ve tahmin değerleri
        y_true = self.valid_series.values
        y_pred = self.forecast.values
        
        # Uzunlukları eşitle
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Metrikler
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAPE (dikkatli - sıfır değerler için)
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
            'forecast_mean': self.forecast.mean(),
            'actual_mean': self.valid_series.mean()
        }
        
        print(f"   📈 Tahmin Performansı:")
        print(f"   • MAE:   {mae:.2f}")
        print(f"   • RMSE:  {rmse:.2f}")
        print(f"   • MAPE:  {mape:.2f}%")
        print(f"   • sMAPE: {smape:.2f}%")
        
        return self.metrics
    
    def create_visualizations(self):
        """Görselleştirmeler oluştur"""
        
        print(f"\n📊 9. Görselleştirmeler oluşturuluyor...")
        
        # Ana forecast grafiği
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Ana tahmin grafiği
        ax1 = axes[0, 0]
        
        # Son 100 günü göster (daha temiz görünüm)
        train_plot = self.train_series.tail(100)
        
        ax1.plot(train_plot.index, train_plot.values, 
                label='Train (Gerçek)', color='blue', linewidth=2)
        ax1.plot(self.valid_series.index, self.valid_series.values, 
                label='Valid (Gerçek)', color='green', linewidth=2)
        ax1.plot(self.forecast.index, self.forecast.values, 
                label='ARIMA Tahmin', color='red', linewidth=2, linestyle='--')
        
        ax1.axvline(x=self.train_series.index[-1], color='gray', linestyle=':', alpha=0.7, 
                   label='Train/Valid Split')
        
        ax1.set_title(f'{self.item_id} - ARIMA{self.best_params} Tahmin', fontweight='bold')
        ax1.set_ylabel('Satış')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals (artıklar)
        ax2 = axes[0, 1]
        residuals = self.model.resid
        ax2.plot(residuals.index, residuals.values, color='purple', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8)
        ax2.set_title('Residuals (Artıklar)', fontweight='bold')
        ax2.set_ylabel('Residual')
        ax2.grid(True, alpha=0.3)
        
        # 3. ACF of residuals
        ax3 = axes[1, 0]
        try:
            plot_acf(residuals.dropna(), ax=ax3, lags=20, alpha=0.05)
            ax3.set_title('ACF of Residuals', fontweight='bold')
        except:
            ax3.text(0.5, 0.5, 'ACF hesaplanamadı', ha='center', va='center', 
                    transform=ax3.transAxes)
            ax3.set_title('ACF of Residuals (Error)', fontweight='bold')
        
        # 4. PACF of residuals
        ax4 = axes[1, 1]
        try:
            plot_pacf(residuals.dropna(), ax=ax4, lags=20, alpha=0.05)
            ax4.set_title('PACF of Residuals', fontweight='bold')
        except:
            ax4.text(0.5, 0.5, 'PACF hesaplanamadı', ha='center', va='center', 
                    transform=ax4.transAxes)
            ax4.set_title('PACF of Residuals (Error)', fontweight='bold')
        
        plt.tight_layout()
        
        # Kaydet
        figure_path = f'{self.artifacts_path}/figures/arima_{self.item_id}_forecast.png'
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Tahmin grafiği: {figure_path}")
        plt.close()
        
        # Basit metrik görselleştirmesi
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        metrics_names = ['MAE', 'RMSE', 'sMAPE (%)']
        metrics_values = [self.metrics['MAE'], self.metrics['RMSE'], self.metrics['sMAPE']]
        
        bars = ax.bar(metrics_names, metrics_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        
        # Değerleri bara yazdır
        for bar, value in zip(bars, metrics_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'{self.item_id} - ARIMA{self.best_params} Performance Metrics', 
                    fontweight='bold', fontsize=14)
        ax.set_ylabel('Metric Value')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Kaydet
        metrics_path = f'{self.artifacts_path}/figures/arima_{self.item_id}_metrics.png'
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Metrik grafiği: {metrics_path}")
        plt.close()
    
    def save_results(self):
        """Sonuçları kaydet"""
        
        print(f"\n💾 10. Sonuçlar kaydediliyor...")
        
        # 1. Model kaydet
        model_path = f'{self.artifacts_path}/models/arima_{self.item_id}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'item_id': self.item_id,
                'best_params': self.best_params,
                'train_series': self.train_series,
                'metrics': self.metrics
            }, f)
        print(f"   ✓ Model: {model_path}")
        
        # 2. Tahminleri kaydet
        forecast_df = pd.DataFrame({
            'date': self.forecast.index,
            'item_id': self.item_id,
            'forecast': self.forecast.values,
            'actual': self.valid_series.values[:len(self.forecast)]
        })
        
        pred_path = f'{self.artifacts_path}/preds/arima_forecast_{self.item_id}.csv'
        forecast_df.to_csv(pred_path, index=False)
        print(f"   ✓ Tahminler: {pred_path}")
        
        # 3. Özet rapor
        report = {
            'item_id': self.item_id,
            'model_type': 'ARIMA',
            'parameters': self.best_params,
            'train_period': f"{self.train_series.index.min()} to {self.train_series.index.max()}",
            'valid_period': f"{self.valid_series.index.min()} to {self.valid_series.index.max()}",
            'forecast_steps': len(self.forecast),
            'metrics': self.metrics,
            'model_aic': self.model.aic,
            'model_bic': self.model.bic
        }
        
        import json
        report_path = f'{self.artifacts_path}/preds/arima_report_{self.item_id}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"   ✓ Rapor: {report_path}")
    
    def run_full_pipeline(self):
        """Tam ARIMA pipeline'ını çalıştır"""
        
        try:
            # 1. En yüksek satışlı ürünü seç
            self.select_top_item()
            
            # 2. Zaman serisi yükle
            self.load_time_series()
            
            # 3. Grid search ile en iyi parametreleri bul
            best_params, grid_results = self.grid_search_arima()
            
            # 4. Modeli eğit
            self.train_arima_model()
            
            # 5. Tahmin yap
            self.make_forecast()
            
            # 6. Metrikleri hesapla
            self.calculate_metrics()
            
            # 7. Görselleştirmeler
            self.create_visualizations()
            
            # 8. Sonuçları kaydet
            self.save_results()
            
            print(f"\n🎉 ARIMA Forecasting tamamlandı!")
            print(f"📊 Model: ARIMA{self.best_params}")
            print(f"🎯 Ürün: {self.item_id}")
            print(f"📈 sMAPE: {self.metrics['sMAPE']:.2f}%")
            print(f"📁 Çıktılar: {self.artifacts_path}/")
            
            return self.model, self.forecast, self.metrics
            
        except Exception as e:
            print(f"\n❌ ARIMA Pipeline hatası: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Ana çalıştırma fonksiyonu"""
    
    print("=" * 65)
    print("ARIMA TEK ÜRÜN SATIŞ TAHMİNİ")
    print("⚠️  NOT: ARIMA tek seri içindir; çok ürünlü için yavaş/scale zor")
    print("=" * 65)
    
    try:
        # ARIMA forecaster'ı başlat
        forecaster = ARIMASingleItemForecaster()
        
        # Tam pipeline'ı çalıştır
        model, forecast, metrics = forecaster.run_full_pipeline()
        
        print(f"\n✅ İşlem başarıyla tamamlandı!")
        print(f"🔮 ARIMA model ile {len(forecast)} günlük tahmin üretildi")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Kullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"\n💥 Beklenmeyen hata: {e}")

if __name__ == "__main__":
    main()
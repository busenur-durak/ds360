#!/usr/bin/env python3
"""
M5 Forecasting - Zaman Serisi Talep Tahmini
Eğitim Amaçlı Kod - Basit ve Anlaşılır Implementasyon

Bu script M5 Competition verisi ile talep tahmini yapar.
Küçük bir alt-küme ile çalışarak hızlı sonuç alınmasını sağlar.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

# Scikit-learn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Zaman serisi kütüphaneleri
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Prophet için try/except
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("Prophet kütüphanesi bulunamadı. Prophet modelleri atlanacak.")
    PROPHET_AVAILABLE = False

# LightGBM için try/except
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("LightGBM kütüphanesi bulunamadı. LightGBM modelleri atlanacak.")
    LIGHTGBM_AVAILABLE = False

# Prefect için try/except (isteğe bağlı)
try:
    from prefect import task, flow
    PREFECT_AVAILABLE = True
except ImportError:
    print("Prefect kütüphanesi bulunamadı. Normal fonksiyonlar kullanılacak.")
    PREFECT_AVAILABLE = False

warnings.filterwarnings('ignore')

# Konfigürasyon
CONFIG = {
    'data_path': './data/',
    'artifacts_path': './artifacts/',
    'state_id': 'CA',  # Sadece California
    'store_id': 'CA_1',  # Tek mağaza
    'n_items': 10,  # Sadece 10 ürün
    'train_days': 365 * 2,  # Son 2 yıl
    'forecast_horizon': 28,  # 28 gün tahmin
    'random_seed': 42
}

class M5Forecaster:
    """
    M5 Forecasting ana sınıfı
    
    Bu sınıf veri yükleme, preprocessing, modelleme ve değerlendirme
    işlemlerini organize eder.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data = {}
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        
        # Çıktı klasörlerini oluştur
        os.makedirs(config['artifacts_path'], exist_ok=True)
        
        print(f"M5 Forecaster başlatıldı")
        print(f"Hedef: {config['state_id']} eyaleti, {config['store_id']} mağazası")
        print(f"Ürün sayısı: {config['n_items']}")
    
    def load_data(self) -> None:
        """
        M5 verilerini yükle ve temizle
        
        M5 Competition verisi 3 ana dosyadan oluşur:
        - sales_train_validation.csv: Satış verileri
        - calendar.csv: Tarih bilgileri
        - sell_prices.csv: Fiyat bilgileri (opsiyonel)
        """
        try:
            print("\n=== VERİ YÜKLEME ===")
            
            # 1. Satış verilerini yükle
            print("Satış verileri yükleniyor...")
            sales_path = os.path.join(self.config['data_path'], 'sales_train_validation.csv')
            
            if not os.path.exists(sales_path):
                raise FileNotFoundError(f"Satış verisi bulunamadı: {sales_path}")
            
            # Sadece belirtilen eyalet ve mağazayı filtrele
            sales_df = pd.read_csv(sales_path)
            print(f"Toplam veri boyutu: {sales_df.shape}")
            
            # Filtreleme
            mask = (sales_df['state_id'] == self.config['state_id']) & \
                   (sales_df['store_id'] == self.config['store_id'])
            
            sales_filtered = sales_df[mask].head(self.config['n_items'])
            print(f"Filtrelenmiş veri boyutu: {sales_filtered.shape}")
            
            # 2. Takvim verilerini yükle
            print("Takvim verileri yükleniyor...")
            calendar_path = os.path.join(self.config['data_path'], 'calendar.csv')
            
            if not os.path.exists(calendar_path):
                raise FileNotFoundError(f"Takvim verisi bulunamadı: {calendar_path}")
            
            calendar_df = pd.read_csv(calendar_path)
            calendar_df['date'] = pd.to_datetime(calendar_df['date'])
            
            # 3. Verileri birleştir
            print("Veriler birleştiriliyor...")
            long_data = self._reshape_to_long_format(sales_filtered, calendar_df)
            
            self.data['sales'] = sales_filtered
            self.data['calendar'] = calendar_df
            self.data['long'] = long_data
            
            print(f"✓ Veri yükleme tamamlandı. Toplam satır: {len(long_data)}")
            
        except Exception as e:
            print(f"❌ Veri yükleme hatası: {e}")
            raise
    
    def _reshape_to_long_format(self, sales_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
        """
        Satış verisini geniş formattan uzun formata çevir
        
        M5 verisi geniş formatta (her gün bir sütun). Zaman serisi analizi
        için uzun format gerekli (her satır bir gün-ürün kombinasyonu).
        """
        print("Veri formatı uzun formata çevriliyor...")
        
        # ID sütunlarını al
        id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        
        # Satış sütunlarını al (d_1, d_2, ... formatında)
        sales_cols = [col for col in sales_df.columns if col.startswith('d_')]
        
        # Uzun formata çevir
        long_df = pd.melt(
            sales_df[id_cols + sales_cols],
            id_vars=id_cols,
            value_vars=sales_cols,
            var_name='d',
            value_name='sales'
        )
        
        # Takvim verisi ile birleştir
        long_df = long_df.merge(calendar_df[['d', 'date', 'wm_yr_wk', 'weekday', 'month', 'year']], on='d')
        long_df = long_df.sort_values(['id', 'date']).reset_index(drop=True)
        
        # NaN değerleri 0 ile doldur
        long_df['sales'] = long_df['sales'].fillna(0)
        
        print(f"✓ Format dönüşümü tamamlandı. Şekil: {long_df.shape}")
        return long_df
    
    def prepare_time_series(self) -> None:
        """
        Zaman serisi verilerini hazırla ve böl
        
        Burada time-based split yapıyoruz. Shuffle yapmıyoruz çünkü
        zaman serisi verilerinde sıra önemli.
        """
        print("\n=== ZAMAN SERİSİ HAZIRLIĞI ===")
        
        # Son N günü al (daha hızlı çalışması için)
        long_df = self.data['long'].copy()
        
        # Tarih sıralaması
        long_df = long_df.sort_values('date')
        unique_dates = sorted(long_df['date'].unique())
        
        # Son train_days gün al
        if len(unique_dates) > self.config['train_days']:
            start_date = unique_dates[-self.config['train_days']]
            long_df = long_df[long_df['date'] >= start_date]
            print(f"Son {self.config['train_days']} gün kullanılıyor: {start_date} - {unique_dates[-1]}")
        
        # Time-based split
        unique_dates = sorted(long_df['date'].unique())
        split_idx = len(unique_dates) - self.config['forecast_horizon']
        
        if split_idx <= 0:
            raise ValueError("Yeterli veri yok. Daha az forecast_horizon veya daha fazla veri gerekli.")
        
        train_end_date = unique_dates[split_idx - 1]
        test_start_date = unique_dates[split_idx]
        
        # Train/test ayırma
        train_df = long_df[long_df['date'] <= train_end_date]
        test_df = long_df[long_df['date'] >= test_start_date]
        
        print(f"Train dönemi: {train_df['date'].min()} - {train_df['date'].max()}")
        print(f"Test dönemi: {test_df['date'].min()} - {test_df['date'].max()}")
        print(f"Train boyutu: {len(train_df)}, Test boyutu: {len(test_df)}")
        
        self.data['train'] = train_df
        self.data['test'] = test_df
        
        # Her ürün için ayrı zaman serisi oluştur
        self.data['item_series'] = {}
        for item_id in train_df['item_id'].unique():
            item_train = train_df[train_df['item_id'] == item_id].copy()
            item_test = test_df[test_df['item_id'] == item_id].copy()
            
            # Eksik günleri doldur (bazı günler satış olmayabilir)
            date_range = pd.date_range(start=item_train['date'].min(), 
                                     end=item_test['date'].max(), 
                                     freq='D')
            
            full_series = pd.DataFrame({'date': date_range})
            
            # Train ve test verilerini birleştir
            all_item_data = pd.concat([item_train, item_test])
            full_series = full_series.merge(all_item_data, on='date', how='left')
            full_series['sales'] = full_series['sales'].fillna(0)
            full_series['item_id'] = full_series['item_id'].fillna(item_id)
            
            # Train/test ayrımını tekrar yap
            train_mask = full_series['date'] <= train_end_date
            
            self.data['item_series'][item_id] = {
                'train': full_series[train_mask].copy(),
                'test': full_series[~train_mask].copy(),
                'full': full_series.copy()
            }
        
        print(f"✓ {len(self.data['item_series'])} ürün için zaman serisi hazırlandı")
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Tahmin performans metriklerini hesapla
        
        MAE: Mean Absolute Error
        RMSE: Root Mean Square Error  
        MAPE: Mean Absolute Percentage Error
        sMAPE: Symmetric Mean Absolute Percentage Error
        """
        # Negatif değerleri 0 yap
        y_pred = np.maximum(y_pred, 0)
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAPE hesapla (sıfır değerler için korumalı)
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = float('inf')
        
        # sMAPE hesapla
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if mask.sum() > 0:
            smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
        else:
            smape = float('inf')
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'sMAPE': smape
        }
    
    def train_naive_models(self) -> None:
        """
        Basit (naive) modelleri eğit
        
        Bu modeller karmaşık değil ama genellikle iyi baseline sağlar:
        1. Naive: Son değeri tekrarla
        2. Seasonal Naive: Geçen haftanın aynı gününü tekrarla
        3. Moving Average: Hareketli ortalama
        """
        print("\n=== NAİVE MODELLER ===")
        
        for item_id in self.data['item_series'].keys():
            print(f"Ürün: {item_id}")
            
            train_series = self.data['item_series'][item_id]['train']
            test_length = len(self.data['item_series'][item_id]['test'])
            
            if len(train_series) == 0:
                print(f"⚠️  {item_id} için yeterli train verisi yok")
                continue
                
            sales_values = train_series['sales'].values
            
            # 1. Naive Forecast (son değeri tekrarla)
            naive_pred = np.full(test_length, sales_values[-1])
            
            # 2. Seasonal Naive (7 gün önceyi tekrarla)
            seasonal_naive_pred = []
            for i in range(test_length):
                lookback_idx = len(sales_values) - 7 + (i % 7)
                if lookback_idx >= 0:
                    seasonal_naive_pred.append(sales_values[lookback_idx])
                else:
                    seasonal_naive_pred.append(sales_values[-1])
            seasonal_naive_pred = np.array(seasonal_naive_pred)
            
            # 3. Moving Average (son 7 günün ortalaması)
            ma_window = min(7, len(sales_values))
            ma_value = np.mean(sales_values[-ma_window:])
            ma_pred = np.full(test_length, ma_value)
            
            # Tahminleri kaydet
            if item_id not in self.predictions:
                self.predictions[item_id] = {}
            
            self.predictions[item_id]['naive'] = naive_pred
            self.predictions[item_id]['seasonal_naive'] = seasonal_naive_pred
            self.predictions[item_id]['moving_average'] = ma_pred
        
        print("✓ Naive modeller tamamlandı")
    
    def train_exponential_smoothing(self) -> None:
        """
        Exponential Smoothing modelleri eğit
        
        Holt-Winters yöntemi trend ve mevsimselliği yakalayabilir.
        """
        print("\n=== EXPONENTIAL SMOOTHING ===")
        
        for item_id in self.data['item_series'].keys():
            print(f"Ürün: {item_id}")
            
            try:
                train_series = self.data['item_series'][item_id]['train']
                test_length = len(self.data['item_series'][item_id]['test'])
                
                if len(train_series) < 14:  # En az 2 hafta veri gerekli
                    print(f"⚠️  {item_id} için yeterli veri yok (en az 14 gün gerekli)")
                    continue
                
                sales_values = train_series['sales'].values
                
                # Holt-Winters modeli (trend + seasonality)
                # seasonal_periods=7 (haftalık mevsimsellik)
                model = ExponentialSmoothing(
                    sales_values,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=7
                )
                
                fitted_model = model.fit()
                pred = fitted_model.forecast(steps=test_length)
                
                # Negatif değerleri 0 yap
                pred = np.maximum(pred, 0)
                
                if item_id not in self.predictions:
                    self.predictions[item_id] = {}
                
                self.predictions[item_id]['exponential_smoothing'] = pred
                
                # Modeli kaydet
                if item_id not in self.models:
                    self.models[item_id] = {}
                self.models[item_id]['exponential_smoothing'] = fitted_model
                
            except Exception as e:
                print(f"⚠️  {item_id} için Exponential Smoothing hatası: {e}")
        
        print("✓ Exponential Smoothing tamamlandı")
    
    def train_prophet_models(self) -> None:
        """
        Facebook Prophet modelleri eğit
        
        Prophet özellikle trend ve mevsimsellik olan zaman serilerinde başarılı.
        """
        if not PROPHET_AVAILABLE:
            print("Prophet mevcut değil, atlanıyor...")
            return
            
        print("\n=== PROPHET MODELS ===")
        
        for item_id in self.data['item_series'].keys():
            print(f"Ürün: {item_id}")
            
            try:
                train_series = self.data['item_series'][item_id]['train']
                test_length = len(self.data['item_series'][item_id]['test'])
                
                if len(train_series) < 30:  # Prophet için daha fazla veri gerekli
                    print(f"⚠️  {item_id} için yeterli veri yok (en az 30 gün gerekli)")
                    continue
                
                # Prophet formatına çevir (ds: tarih, y: değer)
                prophet_df = pd.DataFrame({
                    'ds': train_series['date'],
                    'y': train_series['sales']
                })
                
                # Prophet modeli
                model = Prophet(
                    yearly_seasonality=False,  # Kısa veri için gerekli değil
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.1  # Daha yumuşak trend değişimleri
                )
                
                model.fit(prophet_df)
                
                # Gelecek tarihleri oluştur
                future = model.make_future_dataframe(periods=test_length)
                forecast = model.predict(future)
                
                # Sadece tahmin kısmını al
                pred = forecast['yhat'].iloc[-test_length:].values
                pred = np.maximum(pred, 0)  # Negatif değerleri 0 yap
                
                if item_id not in self.predictions:
                    self.predictions[item_id] = {}
                
                self.predictions[item_id]['prophet'] = pred
                
                # Modeli kaydet
                if item_id not in self.models:
                    self.models[item_id] = {}
                self.models[item_id]['prophet'] = model
                
            except Exception as e:
                print(f"⚠️  {item_id} için Prophet hatası: {e}")
        
        print("✓ Prophet modelleri tamamlandı")
    
    def train_lightgbm_models(self) -> None:
        """
        LightGBM ile zaman serisi modeli eğit
        
        Gradient boosting yöntemi, feature engineering ile güçlü sonuçlar verir.
        """
        if not LIGHTGBM_AVAILABLE:
            print("LightGBM mevcut değil, atlanıyor...")
            return
            
        print("\n=== LIGHTGBM MODELS ===")
        
        for item_id in self.data['item_series'].keys():
            print(f"Ürün: {item_id}")
            
            try:
                train_series = self.data['item_series'][item_id]['train']
                test_series = self.data['item_series'][item_id]['test']
                
                if len(train_series) < 28:  # En az 4 hafta
                    print(f"⚠️  {item_id} için yeterli veri yok (en az 28 gün gerekli)")
                    continue
                
                # Feature engineering
                train_features = self._create_features(train_series)
                test_features = self._create_features(test_series, is_test=True)
                
                if len(train_features) == 0:
                    print(f"⚠️  {item_id} için feature oluşturulamadı")
                    continue
                
                # LightGBM veri formatı
                feature_cols = [col for col in train_features.columns if col != 'sales']
                
                train_X = train_features[feature_cols]
                train_y = train_features['sales']
                test_X = test_features[feature_cols]
                
                # Model eğitimi
                model = lgb.LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=self.config['random_seed'],
                    verbosity=-1
                )
                
                model.fit(train_X, train_y)
                pred = model.predict(test_X)
                pred = np.maximum(pred, 0)  # Negatif değerleri 0 yap
                
                if item_id not in self.predictions:
                    self.predictions[item_id] = {}
                
                self.predictions[item_id]['lightgbm'] = pred
                
                # Modeli kaydet
                if item_id not in self.models:
                    self.models[item_id] = {}
                self.models[item_id]['lightgbm'] = model
                
            except Exception as e:
                print(f"⚠️  {item_id} için LightGBM hatası: {e}")
        
        print("✓ LightGBM modelleri tamamlandı")
    
    def _create_features(self, df: pd.DataFrame, is_test: bool = False) -> pd.DataFrame:
        """
        LightGBM için feature engineering
        
        Zaman serisi için tipik özellikler:
        - Lag features (geçmiş değerler)
        - Rolling statistics (hareketli istatistikler)
        - Time features (gün, hafta, ay)
        """
        features_df = df.copy()
        
        # Zaman özellikleri
        features_df['day_of_week'] = features_df['date'].dt.dayofweek
        features_df['day_of_month'] = features_df['date'].dt.day
        features_df['week_of_year'] = features_df['date'].dt.isocalendar().week
        features_df['month'] = features_df['date'].dt.month
        
        # Lag features (1, 7, 14 gün önceki değerler)
        for lag in [1, 7, 14]:
            features_df[f'sales_lag_{lag}'] = features_df['sales'].shift(lag)
        
        # Rolling features (son 7 günün ortalaması ve standart sapması)
        features_df['sales_roll_mean_7'] = features_df['sales'].rolling(window=7, min_periods=1).mean()
        features_df['sales_roll_std_7'] = features_df['sales'].rolling(window=7, min_periods=1).std()
        
        # Test verisi için lag'leri düzelt (train'den alınmalı)
        if is_test:
            # Test verisinde lag'ler NaN olacak, bunları train'in son değerleri ile doldur
            for col in features_df.columns:
                if 'lag_' in col or 'roll_' in col:
                    features_df[col] = features_df[col].fillna(method='ffill').fillna(0)
        
        # NaN değerleri doldur
        features_df = features_df.fillna(0)
        
        # Gereksiz sütunları çıkar
        drop_cols = ['date', 'item_id', 'd']
        for col in drop_cols:
            if col in features_df.columns:
                features_df = features_df.drop(col, axis=1)
        
        return features_df
    
    def evaluate_models(self) -> None:
        """
        Tüm modelleri değerlendir ve sonuçları karşılaştır
        """
        print("\n=== MODEL DEĞERLENDİRME ===")
        
        all_metrics = {}
        
        for item_id in self.data['item_series'].keys():
            print(f"\nÜrün: {item_id}")
            
            # Gerçek değerler
            actual = self.data['item_series'][item_id]['test']['sales'].values
            
            if len(actual) == 0:
                print(f"⚠️  {item_id} için test verisi yok")
                continue
            
            item_metrics = {}
            
            # Her model için metrik hesapla
            if item_id in self.predictions:
                for model_name, pred in self.predictions[item_id].items():
                    if len(pred) == len(actual):
                        metrics = self.calculate_metrics(actual, pred)
                        item_metrics[model_name] = metrics
                        
                        print(f"  {model_name:20} - MAE: {metrics['MAE']:.2f}, "
                              f"RMSE: {metrics['RMSE']:.2f}, sMAPE: {metrics['sMAPE']:.2f}%")
            
            all_metrics[item_id] = item_metrics
        
        # Ortalama performans
        print("\n=== ORTALAMA PERFORMANS ===")
        model_names = set()
        for item_metrics in all_metrics.values():
            model_names.update(item_metrics.keys())
        
        avg_metrics = {}
        for model_name in model_names:
            mae_values = []
            rmse_values = []
            smape_values = []
            
            for item_metrics in all_metrics.values():
                if model_name in item_metrics:
                    mae_values.append(item_metrics[model_name]['MAE'])
                    rmse_values.append(item_metrics[model_name]['RMSE'])
                    smape_values.append(item_metrics[model_name]['sMAPE'])
            
            if mae_values:
                avg_metrics[model_name] = {
                    'MAE': np.mean(mae_values),
                    'RMSE': np.mean(rmse_values),
                    'sMAPE': np.mean(smape_values)
                }
        
        # Sonuçları yazdır
        for model_name, metrics in avg_metrics.items():
            print(f"{model_name:20} - MAE: {metrics['MAE']:.2f}, "
                  f"RMSE: {metrics['RMSE']:.2f}, sMAPE: {metrics['sMAPE']:.2f}%")
        
        self.metrics = {
            'detailed': all_metrics,
            'average': avg_metrics
        }
        
        print("✓ Model değerlendirme tamamlandı")
    
    def create_visualizations(self) -> None:
        """
        Sonuçları görselleştir
        
        Her ürün için ayrı grafik + genel karşılaştırma
        """
        print("\n=== GÖRSELLEŞTİRME ===")
        
        # 1. Her ürün için ayrı grafik
        for item_id in list(self.data['item_series'].keys())[:3]:  # İlk 3 ürün
            self._plot_item_forecast(item_id)
        
        # 2. Model performans karşılaştırması
        self._plot_model_comparison()
        
        print("✓ Görselleştirme tamamlandı")
    
    def _plot_item_forecast(self, item_id: str) -> None:
        """Tek ürün için tahmin grafiği"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Veriyi al
        train_data = self.data['item_series'][item_id]['train']
        test_data = self.data['item_series'][item_id]['test']
        
        # Son 60 günü göster (daha temiz görünüm için)
        if len(train_data) > 60:
            train_data = train_data.tail(60)
        
        # Üst grafik: Zaman serisi + tahminler
        ax1.plot(train_data['date'], train_data['sales'], 
                label='Train (Gerçek)', color='blue', linewidth=2)
        ax1.plot(test_data['date'], test_data['sales'], 
                label='Test (Gerçek)', color='green', linewidth=2)
        
        # Tahminleri çiz
        colors = ['red', 'orange', 'purple', 'brown', 'pink']
        if item_id in self.predictions:
            for i, (model_name, pred) in enumerate(self.predictions[item_id].items()):
                color = colors[i % len(colors)]
                ax1.plot(test_data['date'], pred, 
                        label=f'{model_name} (Tahmin)', 
                        color=color, linestyle='--', alpha=0.8)
        
        ax1.set_title(f'Ürün: {item_id} - Satış Tahmini')
        ax1.set_ylabel('Satış Miktarı')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Alt grafik: Hata analizi
        if item_id in self.predictions:
            actual = test_data['sales'].values
            for i, (model_name, pred) in enumerate(self.predictions[item_id].items()):
                if len(pred) == len(actual):
                    error = actual - pred
                    color = colors[i % len(colors)]
                    ax2.plot(test_data['date'], error, 
                            label=f'{model_name} Hatası', 
                            color=color, alpha=0.7)
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Tahmin Hataları (Gerçek - Tahmin)')
        ax2.set_ylabel('Hata')
        ax2.set_xlabel('Tarih')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Kaydet
        filename = f"forecast_{item_id.replace('/', '_')}.png"
        filepath = os.path.join(self.config['artifacts_path'], filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Grafik kaydedildi: {filepath}")
        plt.close()
    
    def _plot_model_comparison(self) -> None:
        """Model performanslarını karşılaştır"""
        if not self.metrics or 'average' not in self.metrics:
            print("⚠️  Metrik verisi bulunamadı")
            return
        
        avg_metrics = self.metrics['average']
        
        if not avg_metrics:
            print("⚠️  Ortalama metrik verisi yok")
            return
        
        # Verileri hazırla
        models = list(avg_metrics.keys())
        mae_values = [avg_metrics[m]['MAE'] for m in models]
        rmse_values = [avg_metrics[m]['RMSE'] for m in models]
        smape_values = [avg_metrics[m]['sMAPE'] for m in models]
        
        # Grafik
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # MAE karşılaştırması
        bars1 = ax1.bar(models, mae_values, color='skyblue', alpha=0.7)
        ax1.set_title('Mean Absolute Error (MAE)')
        ax1.set_ylabel('MAE')
        ax1.tick_params(axis='x', rotation=45)
        
        # Değerleri bara yazdır
        for bar, value in zip(bars1, mae_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # RMSE karşılaştırması
        bars2 = ax2.bar(models, rmse_values, color='lightcoral', alpha=0.7)
        ax2.set_title('Root Mean Square Error (RMSE)')
        ax2.set_ylabel('RMSE')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, rmse_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # sMAPE karşılaştırması
        bars3 = ax3.bar(models, smape_values, color='lightgreen', alpha=0.7)
        ax3.set_title('Symmetric Mean Absolute Percentage Error (sMAPE)')
        ax3.set_ylabel('sMAPE (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars3, smape_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # En iyi model tablosu
        ax4.axis('off')
        
        # En iyi modeli bul (MAE'ye göre)
        best_model_idx = np.argmin(mae_values)
        best_model = models[best_model_idx]
        
        table_data = [
            ['Metrik', 'En İyi Model', 'Değer'],
            ['MAE', models[np.argmin(mae_values)], f"{min(mae_values):.2f}"],
            ['RMSE', models[np.argmin(rmse_values)], f"{min(rmse_values):.2f}"],
            ['sMAPE', models[np.argmin(smape_values)], f"{min(smape_values):.1f}%"]
        ]
        
        table = ax4.table(cellText=table_data, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # Header'ı vurgula
        for i in range(3):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax4.set_title(f'En İyi Performans Özeti\nGenel Kazanan: {best_model}', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Kaydet
        filepath = os.path.join(self.config['artifacts_path'], 'model_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Model karşılaştırma grafiği kaydedildi: {filepath}")
        plt.close()
    
    def save_results(self) -> None:
        """
        Sonuçları kaydet
        
        1. Tahminleri CSV olarak
        2. Modelleri pickle olarak
        3. Metrikleri JSON olarak
        """
        print("\n=== SONUÇLARI KAYDETME ===")
        
        # 1. Tahminleri CSV olarak kaydet
        predictions_list = []
        
        for item_id in self.predictions.keys():
            test_dates = self.data['item_series'][item_id]['test']['date'].values
            actual_values = self.data['item_series'][item_id]['test']['sales'].values
            
            for i, date in enumerate(test_dates):
                row = {
                    'item_id': item_id,
                    'date': date,
                    'actual': actual_values[i] if i < len(actual_values) else 0
                }
                
                # Her modelin tahminini ekle
                for model_name, pred in self.predictions[item_id].items():
                    if i < len(pred):
                        row[f'pred_{model_name}'] = pred[i]
                    else:
                        row[f'pred_{model_name}'] = 0
                
                predictions_list.append(row)
        
        if predictions_list:
            pred_df = pd.DataFrame(predictions_list)
            pred_path = os.path.join(self.config['artifacts_path'], 'predictions.csv')
            pred_df.to_csv(pred_path, index=False)
            print(f"✓ Tahminler kaydedildi: {pred_path}")
        
        # 2. Modelleri kaydet
        model_path = os.path.join(self.config['artifacts_path'], 'models.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.models, f)
        print(f"✓ Modeller kaydedildi: {model_path}")
        
        # 3. Metrikleri kaydet
        if self.metrics:
            import json
            
            # JSON serileştirme için numpy float'ları normal float'a çevir
            def convert_numpy(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(x) for x in obj]
                return obj
            
            metrics_clean = convert_numpy(self.metrics)
            
            metrics_path = os.path.join(self.config['artifacts_path'], 'metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics_clean, f, indent=2)
            print(f"✓ Metrikler kaydedildi: {metrics_path}")
        
        print("✓ Tüm sonuçlar kaydedildi")
    
    def run_full_pipeline(self) -> None:
        """
        Tam pipeline'ı çalıştır
        
        Bu ana fonksiyon tüm adımları sırayla çalıştırır.
        """
        try:
            print("🚀 M5 Forecasting Pipeline Başlatılıyor...")
            print(f"⚙️  Konfigürasyon: {self.config}")
            
            # 1. Veri yükleme
            self.load_data()
            
            # 2. Zaman serisi hazırlığı
            self.prepare_time_series()
            
            # 3. Model eğitimi
            self.train_naive_models()
            self.train_exponential_smoothing()
            self.train_prophet_models()
            self.train_lightgbm_models()
            
            # 4. Değerlendirme
            self.evaluate_models()
            
            # 5. Görselleştirme
            self.create_visualizations()
            
            # 6. Sonuçları kaydetme
            self.save_results()
            
            print("\n🎉 Pipeline başarıyla tamamlandı!")
            print(f"📁 Sonuçlar: {self.config['artifacts_path']}")
            
        except Exception as e:
            print(f"\n❌ Pipeline hatası: {e}")
            import traceback
            traceback.print_exc()
            raise

# Prefect ile çalışma (opsiyonel)
if PREFECT_AVAILABLE:
    @task
    def load_data_task(forecaster):
        forecaster.load_data()
        return forecaster
    
    @task
    def prepare_time_series_task(forecaster):
        forecaster.prepare_time_series()
        return forecaster
    
    @task
    def train_models_task(forecaster):
        forecaster.train_naive_models()
        forecaster.train_exponential_smoothing()
        forecaster.train_prophet_models()
        forecaster.train_lightgbm_models()
        return forecaster
    
    @task
    def evaluate_task(forecaster):
        forecaster.evaluate_models()
        return forecaster
    
    @task
    def visualize_task(forecaster):
        forecaster.create_visualizations()
        return forecaster
    
    @task
    def save_results_task(forecaster):
        forecaster.save_results()
        return forecaster
    
    @flow(name="M5 Forecasting Pipeline")
    def m5_forecasting_flow(config):
        """Prefect flow olarak pipeline çalıştır"""
        forecaster = M5Forecaster(config)
        
        forecaster = load_data_task(forecaster)
        forecaster = prepare_time_series_task(forecaster)
        forecaster = train_models_task(forecaster)
        forecaster = evaluate_task(forecaster)
        forecaster = visualize_task(forecaster)
        forecaster = save_results_task(forecaster)
        
        return forecaster

def main():
    """
    Ana çalıştırma fonksiyonu
    
    Bu fonksiyon script'i doğrudan çalıştırdığımızda çağrılır.
    """
    print("=" * 60)
    print("M5 FORECASTING - ZAMAN SERİSİ TALEP TAHMİNİ")
    print("Eğitim Amaçlı Uygulama")
    print("=" * 60)
    
    # Konfigürasyonu yazdır
    print("\n📋 KONFIGÜRASYON:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    try:
        if PREFECT_AVAILABLE:
            print("\n🔄 Prefect ile çalışılıyor...")
            forecaster = m5_forecasting_flow(CONFIG)
        else:
            print("\n🔄 Normal modda çalışılıyor...")
            forecaster = M5Forecaster(CONFIG)
            forecaster.run_full_pipeline()
        
        print("\n✅ İşlem başarıyla tamamlandı!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Kullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"\n💥 Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
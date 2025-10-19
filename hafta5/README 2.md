# Hafta 5 - M5 Forecasting: Zaman Serisi Talep Tahmini

Bu proje M5 Competition verisi ile zaman serisi talep tahmini yapan eğitim amaçlı bir uygulamadır. Basit, temiz ve anlaşılır kod ile çeşitli forecasting yöntemlerini karşılaştırır.

## 🎯 Hedef

- M5 Competition verisi ile talep tahmini yapmak
- Farklı zaman serisi modellerini karşılaştırmak
- Eğitim amaçlı basit ve anlaşılır kod sunmak
- Hızlı çalışması için küçük veri alt-kümesi kullanmak

## 📋 Özellikler

### Modeller
- **Naive Models**: Basit tahmin yöntemleri (son değer, seasonal naive, moving average)
- **Exponential Smoothing**: Holt-Winters yöntemi ile trend ve mevsimsellik
- **Prophet**: Facebook'un zaman serisi kütüphanesi
- **LightGBM**: Gradient boosting ile feature engineering

### Metrikler
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **MAPE** (Mean Absolute Percentage Error)
- **sMAPE** (Symmetric Mean Absolute Percentage Error)

### Çıktılar
- Model karşılaştırma grafikleri
- Tahmin sonuçları (CSV)
- Eğitilmiş modeller (pickle)
- Performance metrikleri (JSON)

## 🚀 Kurulum

### Gereksinimler

```bash
pip install -r requirements.txt
```

Gerekli kütüphaneler:
- pandas, numpy, matplotlib
- scikit-learn, statsmodels
- prophet, lightgbm
- prefect (opsiyonel)

### Veri Hazırlığı

**Seçenek 1: Gerçek M5 Verisi**
1. [M5 Competition](https://www.kaggle.com/c/m5-forecasting-accuracy) sayfasından veriyi indirin
2. `data/` klasörüne şu dosyaları yerleştirin:
   - `calendar.csv`
   - `sales_train_validation.csv`
   - `sell_prices.csv`

**Seçenek 2: Örnek Veri (Test için)**
```bash
python create_sample_data.py
```

## 💻 Kullanım

### Temel Çalıştırma

```bash
python m5_forecasting.py
```

### Konfigürasyon

Script başındaki `CONFIG` dictionary'si ile ayarları değiştirebilirsiniz:

```python
CONFIG = {
    'data_path': './data/',
    'artifacts_path': './artifacts/',
    'state_id': 'CA',          # Eyalet seçimi
    'store_id': 'CA_1',        # Mağaza seçimi
    'n_items': 10,             # Kaç ürün analiz edilecek
    'train_days': 365 * 2,     # Eğitim için kaç gün kullanılacak
    'forecast_horizon': 28,    # Kaç gün tahmin yapılacak
    'random_seed': 42
}
```

### Prefect ile Çalıştırma (Opsiyonel)

Eğer Prefect yüklüyse, otomatik olarak workflow olarak çalışır.

## 📊 Çıktılar

Tüm sonuçlar `artifacts/` klasörüne kaydedilir:

- `predictions.csv`: Tüm modellerin tahminleri
- `models.pkl`: Eğitilmiş modeller
- `metrics.json`: Performance metrikleri
- `model_comparison.png`: Model karşılaştırma grafiği
- `forecast_[item_id].png`: Her ürün için tahmin grafikleri

## 🔍 Kod Yapısı

### Ana Sınıf: M5Forecaster

```python
class M5Forecaster:
    def load_data()                    # Veri yükleme ve temizleme
    def prepare_time_series()          # Zaman serisi hazırlığı ve time-based split
    def train_naive_models()           # Basit modeller
    def train_exponential_smoothing()  # Holt-Winters
    def train_prophet_models()         # Prophet modelleri
    def train_lightgbm_models()        # LightGBM ile feature engineering
    def evaluate_models()              # Model değerlendirme
    def create_visualizations()        # Grafik oluşturma
    def save_results()                 # Sonuçları kaydetme
    def run_full_pipeline()            # Tam pipeline
```

### Önemli Özellikler

**Time-based Split**: Zaman serisi için doğru veri bölünmesi
```python
# Shuffle yapmıyoruz, sıralı olarak bölüyoruz
train_end_date = unique_dates[split_idx - 1]
test_start_date = unique_dates[split_idx]
```

**Feature Engineering**: LightGBM için zaman özellikleri
```python
# Lag features, rolling statistics, time features
features_df['sales_lag_7'] = features_df['sales'].shift(7)
features_df['sales_roll_mean_7'] = features_df['sales'].rolling(7).mean()
```

**Hata Yönetimi**: Try/except blokları ile güvenli çalışma
```python
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
```

## 📈 Beklenen Sonuçlar

Genellikle şu sıralamayı görürüz:
1. **LightGBM**: En iyi performance (feature engineering sayesinde)
2. **Prophet**: İyi trend ve mevsimsellik yakalama
3. **Exponential Smoothing**: Orta düzey performance
4. **Naive Models**: Baseline olarak kullanışlı

## 🎓 Eğitim Notları

### Neden Bu Yaklaşımı Seçtik?

1. **Küçük Veri Alt-kümesi**: Hızlı çalışma ve kolay anlama
2. **Time-based Split**: Zaman serisi için doğru yaklaşım
3. **Çoklu Model**: Farklı yöntemleri karşılaştırma imkanı
4. **Bol Yorum**: "Neden böyle yapıyoruz?" açıklamaları
5. **Hata Yönetimi**: Gerçek dünya senaryolarına hazırlık

### Öğrenciler İçin İpuçları

- Önce `create_sample_data.py` ile örnek veri oluşturun
- `CONFIG` ayarlarını değiştirerek deneyim kazanın
- Her modelin kodunu ayrı ayrı inceleyin
- Grafikleri analiz ederek pattern'leri anlamaya çalışın
- Kendi feature'larınızı eklemeyi deneyin

## ⚠️ Bilinen Sınırlamalar

- Sadece tek eyalet (CA) ve tek mağaza (CA_1) ile çalışır
- Kompleks feature engineering sınırlı
- Ensemble modeller yok
- Hyperparameter tuning minimal

## 🔧 Sorun Giderme

**Prophet Yüklenmiyor**:
```bash
pip install prophet
# veya
conda install -c conda-forge prophet
```

**LightGBM Problemi**:
```bash
pip install lightgbm
# Mac için
brew install libomp
```

**Veri Bulunamıyor**:
```bash
# Önce örnek veri oluşturun
python create_sample_data.py
```

## 📚 Referanslar

- [M5 Competition](https://www.kaggle.com/c/m5-forecasting-accuracy)
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Statsmodels Time Series](https://www.statsmodels.org/stable/tsa.html)

---

Bu proje eğitim amaçlıdır. Üretim ortamında kullanmadan önce daha kapsamlı validasyon ve optimizasyon yapılması önerilir.
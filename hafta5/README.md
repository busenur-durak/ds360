# Hafta 5 - M5 Forecasting: Zaman Serisi Talep Tahmini

🏪 **Kapsamlı M5 Competition Tabanlı Talep Tahmin Sistemi**

Bu proje M5 Competition verisi ile profesyonel zaman serisi talep tahmini yapan eğitim amaçlı kapsamlı bir uygulamadır. Geleneksel istatistiksel yöntemlerden modern machine learning tekniklerine, manual analizden otomatik pipeline'a kadar tam bir forecasting ekosistemine sahiptir.

## 📖 M5 Competition: Dataset Hikayesi

### 🛒 **Walmart'ın Hikayesi**
M5 Competition, dünyanın en büyük perakende zinciri **Walmart**'ın gerçek satış verilerini kullanarak düzenlenen uluslararası bir forecasting yarışmasıdır. Bu competition, retail sektörünün en büyük challengelerinden biri olan **demand forecasting** problemini ele alır.

### 📊 **Dataset'in Büyüklüğü ve Kapsamı**
- **📅 Zaman Aralığı**: 2011-2016 (5 yıl, 1,969 gün)
- **🌍 Coğrafi Kapsam**: 3 Eyalet (California, Texas, Wisconsin)
- **🏪 Mağaza Sayısı**: 10 mağaza (CA_1-4, TX_1-3, WI_1-3)
- **📦 Ürün Kategorileri**: 3 ana kategori (FOODS, HOBBIES, HOUSEHOLD)
- **🛍️ Toplam Ürün**: 3,049 benzersiz ürün
- **📈 Time Series**: 30,490 adet günlük satış serisi
- **💾 Veri Boyutu**: ~30GB (tüm hierarchical levels dahil)

### 🎯 **Competition'un Amacı**
M5 Competition'un temel hedefi:
1. **Hierarchical Forecasting**: Ürün → Kategori → Mağaza → Eyalet seviyelerinde tahmin
2. **Uncertainty Quantification**: Sadece nokta tahmin değil, güven aralıkları da
3. **Real-world Applicability**: Akademik araştırma ile industry practice arasında köprü
4. **Evaluation Methodology**: WRMSSE (Weighted Root Mean Squared Scaled Error) metriği

### 🏆 **Competition Sonuçları**
- **📅 Düzenleme Tarihi**: 2020
- **👥 Katılımcı Sayısı**: 909 takım, 5,558 katılımcı
- **🥇 Kazanan Yaklaşım**: LightGBM ensemble + hierarchical reconciliation
- **📊 En İyi sMAPE**: ~12.03% (leaderboard)
- **🔗 Kaggle Link**: https://www.kaggle.com/c/m5-forecasting-accuracy

### 🎓 **Eğitim Değeri**
Bu dataset'in eğitim için seçilme nedenleri:
1. **Gerçek Veri**: Sentetik değil, gerçek Walmart satış verisi
2. **Komplekslik**: Seasonality, trend, promotional effects, external factors
3. **Scale**: Büyük data ile çalışma deneyimi
4. **Industry Relevance**: Retail sektörünün gerçek problemleri
5. **Benchmark**: Akademik literature'da yaygın kullanım

## 🏗️ **Modüler Proje Mimarisi**

Bu proje, educational clarity ve production readiness için modüler mimari kullanır:

```
📦 hafta5/                         # Ana proje klasörü
├── 📁 P1_data_preparation/        # Veri hazırlama ve preprocessing
├── 📁 P2_feature_engineering/     # Feature engineering pipeline
├── 📁 P3_traditional_models/      # İstatistiksel modeller (ARIMA)
├── 📁 P4_modern_models/           # Modern modeller (Prophet)
├── 📁 P5_ml_models/               # ML modelleri (LightGBM)
├── 📁 P6_validation/              # Cross-validation ve evaluation
├── 📁 P7_automation/              # Prefect pipeline automation
├── 📁 legacy/                     # Legacy monolith kod
├── 📁 data/                       # M5 dataset
├── 📁 artifacts/                  # Model outputs ve reports
└── 🐳 Docker/Compose files        # Containerization
```

### 🎯 **Modüler Yaklaşımın Faydaları**
- **📚 Educational**: Her modül tek concept'e odaklanır
- **🔧 Maintainable**: Kod bakımı ve güncelleme kolaylaşır
- **🚀 Scalable**: Modüler deployment ve scaling
- **🧪 Testable**: Her modül bağımsız test edilebilir
- **👥 Collaborative**: Farklı geliştiriciler farklı modüllerde çalışabilir

### 🎮 **Hızlı Başlangıç**
```bash
# Modüler çalıştırma
python run_modular.py --module P1    # Sadece data prep
python run_modular.py --module P7    # Sadece automation
python run_modular.py               # Full pipeline

# Docker ile production
docker build -t m5-forecast:dev .
docker run --rm -v $(pwd)/artifacts:/app/artifacts m5-forecast:dev
```

## 🎯 Proje Hedefleri

### Akademik Hedefler
- **M5 Competition Verisi**: Gerçek dünya Walmart satış verisi ile çalışma
- **Çoklu Model Karşılaştırması**: ARIMA, Prophet, LightGBM arasında objektif karşılaştırma
- **Proper Time Series Validation**: Temporal leakage'ı önleyecek rolling-origin cross-validation
- **Feature Engineering**: Lag, rolling, seasonal features ile model performansı iyileştirme
- **Production-Ready Pipeline**: Prefect ile otomatik günlük forecasting akışı

### Teknik Hedefler
- **Reproducible Research**: Seed kontrolü ve deterministik sonuçlar
- **Scalable Architecture**: Docker containerization ve cloud deployment hazırlığı
- **Educational Code**: Her adımda "neden böyle yapıyoruz?" açıklamaları
- **Error Handling**: Robust exception management ve graceful degradation
- **Comprehensive Logging**: Debug ve monitoring için detaylı log yapısı

## 📋 Sistem Özellikleri

### 🤖 Forecasting Modelleri

#### **Geleneksel İstatistiksel Yöntemler**
- **ARIMA (AutoRegressive Integrated Moving Average)**
  - Otomatik (p,d,q) parametre optimizasyonu
  - Stationarity testing (ADF test)
  - Box-Jenkins metodolojisi
  - Grid search ile en iyi parametreler
  - Performance: ~46% sMAPE (dataset'e göre değişir)

- **Exponential Smoothing (Holt-Winters)**
  - Trend ve seasonal komponent ayrıştırması
  - Alpha, beta, gamma parametrelerinin otomatik optimizasyonu
  - Additive/multiplicative seasonal pattern detection

#### **Modern Machine Learning Yöntemleri**
- **Facebook Prophet**
  - Otomatik trend ve seasonality detection
  - Holiday effects modeling capability
  - Uncertainty intervals ile tahmin güven aralıkları
  - Performance: ~28% sMAPE (ARIMA'dan %40 daha iyi)

- **LightGBM (Gradient Boosting)**
  - Multi-item forecasting capability
  - Rich feature engineering pipeline
  - Categorical encoding (item_id, store_id)
  - Time-based features (dow, month, week)
  - Lag features (1, 7, 28 günlük)
  - Rolling statistics (7, 28 günlük ortalamalar)
  - Performance: ~33% sMAPE
  - Feature importance analysis

#### **Baseline Modeller**
- **Naive Forecasting**: Son değeri tekrarlama
- **Seasonal Naive**: Geçen yılın aynı dönemini kullanma
- **Moving Average**: Son N günün ortalaması
- **Linear Trend**: Doğrusal trend extrapolasyonu

### 📊 Evaluation Metrikleri

#### **Primary Metrics**
- **MAE** (Mean Absolute Error): Mutlak hatanın ortalaması
- **RMSE** (Root Mean Square Error): Büyük hataları penalize eden metrik
- **sMAPE** (Symmetric Mean Absolute Percentage Error): M5 Competition'un resmi metriği
- **MAPE** (Mean Absolute Percentage Error): Yüzdelik hata

#### **Advanced Evaluation**
- **Time Series Cross-Validation**: Rolling-origin, no shuffle, temporal order preserved
- **Fold Strategy**: 3-fold CV, 28-day validation horizon
- **Statistical Significance**: Paired t-test for model comparison
- **Residual Analysis**: Heteroscedasticity ve autocorrelation testleri

### 🔧 Feature Engineering Pipeline

#### **Temporal Features**
```python
# Time-based decomposition
df['dow'] = df.index.dayofweek        # Day of week (0-6)
df['month'] = df.index.month          # Month (1-12)
df['dom'] = df.index.day              # Day of month (1-31)
df['weekofyear'] = df.index.isocalendar().week  # Week number (1-53)
```

#### **Lag Features** 
```python
# Historical patterns
df['lag_1'] = df['sales'].shift(1)    # Yesterday's sales
df['lag_7'] = df['sales'].shift(7)    # Same day last week
df['lag_28'] = df['sales'].shift(28)  # Same day last month
```

#### **Rolling Statistics**
```python
# Trend indicators
df['roll_mean_7'] = df['sales'].rolling(7).mean()   # Weekly average
df['roll_mean_28'] = df['sales'].rolling(28).mean() # Monthly average
df['roll_std_7'] = df['sales'].rolling(7).std()     # Weekly volatility
```

#### **Categorical Encoding**
```python
# LabelEncoder for high-cardinality categoricals
item_encoder = LabelEncoder()
df['item_id_encoded'] = item_encoder.fit_transform(df['item_id'])
store_encoder = LabelEncoder()  
df['store_id_encoded'] = store_encoder.fit_transform(df['store_id'])
```

### 🎯 Output Yapısı

#### **Model Artifacts**
```
artifacts/
├── models/
│   ├── arima_FOODS_3_090.pkl      # ARIMA model object
│   ├── prophet_FOODS_3_090.json   # Prophet model serialization
│   └── lgbm.pkl                   # LightGBM booster object
```

#### **Predictions & Reports**
```
artifacts/
├── preds/
│   ├── arima_forecast_FOODS_3_090.csv     # Single-item ARIMA predictions
│   ├── prophet_forecast_FOODS_3_090.csv   # Single-item Prophet predictions
│   ├── lgbm_forecast_all.csv              # Multi-item LightGBM predictions
│   ├── run_YYYYMMDD.csv                   # Daily pipeline output
│   └── run_YYYYMMDD_summary.png           # Visual summary report
├── reports/
│   ├── tscv_report.json                   # Cross-validation results
│   ├── tscv_report.md                     # Human-readable CV report
│   └── model_comparison.json              # Model performance comparison
```

#### **Visualizations**
```
artifacts/
├── figures/
│   ├── arima_FOODS_3_090_forecast.png     # ARIMA forecast plot
│   ├── arima_FOODS_3_090_metrics.png      # ARIMA diagnostics
│   ├── prophet_FOODS_3_090_forecast.png   # Prophet forecast plot
│   ├── prophet_FOODS_3_090_components.png # Prophet decomposition
│   ├── lgbm_feature_importance.png        # Feature importance plot
│   ├── feature_correlations.png           # Feature correlation heatmap
│   ├── feature_distributions.png          # Feature distribution plots
│   └── overall_daily_sales.png            # Sales trend visualization
```

## 🚀 Kurulum ve Çalıştırma

### 📋 Sistem Gereksinimleri

- **Python**: 3.9+ (önerilen 3.11)
- **RAM**: Minimum 4GB (önerilen 8GB)
- **Disk**: 2GB+ boş alan (veri + model artifacts için)
- **OS**: Windows, macOS, Linux (Docker desteği)

### 🐍 Python Kurulum

#### **Seçenek 1: Virtual Environment (Önerilen)**
```bash
# Virtual environment oluştur
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Dependencies yükle
pip install -r requirements.txt
```

#### **Seçenek 2: Conda Environment**
```bash
# Conda environment oluştur
conda create -n m5-forecast python=3.11

# Activate
conda activate m5-forecast

# Dependencies yükle
pip install -r requirements.txt

# Prophet için (eğer pip başarısız olursa)
conda install -c conda-forge prophet
```

#### **Required Dependencies**
```txt
pandas>=2.0.0          # Data manipulation
numpy>=1.24.0          # Numerical computing  
matplotlib>=3.7.0      # Visualization
scikit-learn>=1.3.0    # Machine learning utilities
statsmodels>=0.14.0    # Statistical models (ARIMA)
prophet>=1.1.4         # Facebook Prophet
lightgbm>=4.0.0        # Gradient boosting
prefect>=3.0.0         # Workflow orchestration
pyarrow>=12.0.0        # Parquet file support
```

### 🐳 Docker Kurulum (Önerilen Production)

#### **Hızlı Başlangıç**
```bash
# Image build et
docker build -t m5-forecast:dev .

# Pipeline'ı çalıştır
docker run --rm -v $(pwd)/artifacts:/app/artifacts m5-forecast:dev

# Sonuçları kontrol et
ls -la artifacts/preds/
```

#### **Docker Compose ile**
```bash
# Tek seferlik çalıştırma
docker-compose up m5-forecast

# Production mode (Prefect server ile)
docker-compose --profile production up -d

# Prefect UI erişim
open http://localhost:4200
```

#### **Volume Mounting (Önemli!)**
```bash
# Data ve artifacts klasörlerini mount et
docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/artifacts:/app/artifacts \
    m5-forecast:dev
```

### 📊 Veri Hazırlığı

#### **Seçenek 1: Gerçek M5 Competition Verisi**
1. [M5 Competition Kaggle](https://www.kaggle.com/c/m5-forecasting-accuracy) sayfasından indirin
2. `data/` klasörüne şu dosyaları yerleştirin:
   ```
   data/
   ├── calendar.csv                    # 1969 satır, tarih ve event bilgileri
   ├── sales_train_validation.csv      # 30490 satır, CA eyaleti satış verileri  
   └── sell_prices.csv                 # 6841121 satır, fiyat bilgileri
   ```

#### **Seçenek 2: Subset Oluşturma (Hızlı Test)**
```bash
# CA eyaleti, CA_1 mağazası, FOODS kategorisi, top 5 ürün
python create_m5_subset.py

# Sonuç: data/ klasöründe subset dosyaları oluşur
```

#### **Seçenek 3: Sentetik Veri (Development)**
```bash
# Test amaçlı sahte veri oluştur
python create_sample_data.py

# Sonuç: Gerçekçi trend ve seasonal pattern'li sahte satış verisi
```

## 💻 Kullanım Senaryoları

### 🎯 **Scenario 1: Eğitim ve Öğrenme (Modüler Yaklaşım)**
```bash
# YENİ: Modüler çalıştırma - her modülü ayrı ayrı öğren
python run_modular.py --module P1    # Veri hazırlama ve subset oluşturma
python run_modular.py --module P2    # Feature engineering pipeline
python run_modular.py --module P3    # ARIMA geleneksel yaklaşım
python run_modular.py --module P4    # Prophet modern yaklaşım
python run_modular.py --module P5    # LightGBM ML yaklaşım
python run_modular.py --module P6    # Cross-validation analizi
python run_modular.py --module P7    # Production automation

# ESKİ: Legacy single-file approach (backward compatibility)
python P1_data_preparation/create_m5_subset.py
python P2_feature_engineering/feature_engineering.py
python P3_traditional_models/arima_single_item.py
python P4_modern_models/prophet_single_item.py
python P5_ml_models/lightgbm_multi_item.py
python P6_validation/time_series_cv_simple.py
python run.py                        # Full pipeline single-run
```

### 🚀 **Scenario 2: Production Pipeline (Automation)**
```bash
# YENİ: Modüler full pipeline
python run_modular.py               # Tüm modülleri sırayla çalıştır

# Prefect automation (P7 modülü)
python run_modular.py --module P7   # Sadece automation pipeline
python P7_automation/prefect_demand_forecast.py  # Direct module call

# Production deployment setup
prefect deployment build P7_automation/prefect_demand_forecast.py:demand_forecast_flow -n daily-forecast
prefect deployment apply demand_forecast_flow-deployment.yaml
prefect agent start -q default

# Docker production
docker build -t m5-forecast:prod .
docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/artifacts:/app/artifacts \
    m5-forecast:prod
```

### 🐳 **Scenario 3: Docker Deployment (Containerized)**
```bash
# Development - quick test
docker build -t m5-forecast:dev .
docker run --rm -v $(pwd)/artifacts:/app/artifacts m5-forecast:dev

# Production with full data and artifacts
docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/artifacts:/app/artifacts \
    -e TZ=Europe/Istanbul \
    m5-forecast:dev

# Docker Compose with Prefect orchestration
docker-compose up m5-forecast                    # Single run
docker-compose --profile production up -d        # With Prefect server

# Multi-module testing
docker run --rm -v $(pwd)/artifacts:/app/artifacts m5-forecast:dev python run_modular.py --module P6
```

### 🔬 **Scenario 4: Araştırma ve Model Karşılaştırması**
```bash
# Model performance karşılaştırması (sadece modeling modülleri)
python run_modular.py --module P3    # ARIMA baseline (sMAPE ~46%)
python run_modular.py --module P4    # Prophet comparison (sMAPE ~28%)  
python run_modular.py --module P5    # LightGBM benchmark (sMAPE ~33%)

# Cross-validation deep dive
python run_modular.py --module P6    # Rolling-origin CV analysis

# Feature engineering impact analysis
python run_modular.py --module P2    # Feature creation
python run_modular.py --module P5    # Feature importance in LightGBM

# Legacy direct module access (advanced users)
python P3_traditional_models/arima_single_item.py   # ARIMA deep dive
python P4_modern_models/prophet_single_item.py      # Prophet components
python P5_ml_models/lightgbm_multi_item.py         # Feature importance
python P6_validation/time_series_cv_simple.py      # CV methodology
```

### 🐍 **Scenario 5: Programmatic Usage (Python API)**
```python
# YENİ: Modüler import pattern
from P1_data_preparation import create_subset
from P2_feature_engineering import create_features  
from P3_traditional_models import run_arima
from P4_modern_models import run_prophet
from P5_ml_models import run_lightgbm
from P6_validation import run_cv_simple
from P7_automation.prefect_demand_forecast import demand_forecast_flow

# Pipeline oluşturma
create_subset()                              # P1: Data preparation
create_features()                           # P2: Feature engineering  
arima_results = run_arima()                 # P3: ARIMA forecasting
prophet_results = run_prophet()             # P4: Prophet forecasting
lgbm_results = run_lightgbm()              # P5: LightGBM forecasting
cv_results = run_cv_simple()               # P6: Cross-validation
flow_results = demand_forecast_flow(        # P7: Automated pipeline
    run_date='2025-01-15',
    forecast_days=7
)

# Model comparison
print(f"ARIMA sMAPE: {arima_results['smape']:.2f}%")
print(f"Prophet sMAPE: {prophet_results['smape']:.2f}%") 
print(f"LightGBM sMAPE: {lgbm_results['smape']:.2f}%")
```

### ⚙️ Konfigürasyon Seçenekleri

#### **Ana Konfigürasyon** (`create_m5_subset.py`)
```python
CONFIG = {
    'state_id': 'CA',           # Eyalet seçimi (CA, TX, WI)
    'store_id': 'CA_1',         # Mağaza seçimi (CA_1, CA_2, CA_3, CA_4)
    'category': 'FOODS',        # Kategori (FOODS, HOBBIES, HOUSEHOLD)
    'n_items': 5,               # Kaç ürün (1-10 arası)
    'validation_days': 28,      # Validation period (28 standart)
    'random_seed': 42           # Reproducibility için
}
```

#### **Feature Engineering** (`feature_engineering.py`)
```python
FEATURE_CONFIG = {
    'lag_days': [1, 7, 28],           # Lag feature'ların günleri
    'rolling_windows': [7, 28],       # Rolling window boyutları
    'seasonal_periods': [7, 365.25], # Seasonal decomposition
    'min_observations': 60           # Feature oluşturma için min gözlem
}
```

#### **Model Hyperparameters**
```python
# ARIMA
ARIMA_CONFIG = {
    'p_range': range(0, 3),     # AR order
    'd_range': range(0, 2),     # Differencing
    'q_range': range(0, 3),     # MA order
    'information_criterion': 'aic'
}

# LightGBM  
LGBM_CONFIG = {
    'objective': 'regression',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'verbose': -1
}

# Prophet
PROPHET_CONFIG = {
    'daily_seasonality': True,
    'weekly_seasonality': True, 
    'yearly_seasonality': False,
    'uncertainty_samples': 1000
}
```

## 📊 Çıktı Analizi ve Raporlar

### 🎯 **Beklenen Performance Sonuçları**

Pipeline çalıştırıldığında tipik olarak şu performans sıralamasını görürüz:

| Model | sMAPE | MAE | RMSE | Özellikler |
|-------|-------|-----|------|------------|
| **Prophet** | ~28% | ~7.5 | ~12.0 | ✅ En iyi tek-ürün performance |
| **LightGBM** | ~33% | ~8.9 | ~13.4 | ✅ Multi-item, feature importance |
| **ARIMA** | ~47% | ~12.5 | ~18.2 | ❌ Baseline, parametrik |

### 📈 **Output Dosyası Formatları**

#### **CSV Predictions** (örnek: `run_20251012.csv`)
```csv
date,item_id,store_id,prediction,model
2016-04-25,FOODS_3_090,CA_1,45.2,lightgbm
2016-04-26,FOODS_3_090,CA_1,42.8,lightgbm
2016-04-27,FOODS_3_090,CA_1,38.5,lightgbm
...
```

#### **JSON Reports** (örnek: `tscv_report.json`)
```json
{
  "method": "Time Series CV (Rolling-Origin)",
  "validation_horizon": 28,
  "n_folds": 3,
  "summary_metrics": {
    "MAE": {"mean": 8.91, "std": 0.73},
    "RMSE": {"mean": 13.44, "std": 1.12},
    "sMAPE": {"mean": 33.83, "std": 5.73}
  }
}
```

#### **Model Comparison** (örnek: `model_comparison.json`)
```json
{
  "timestamp": "2025-01-15T09:30:45",
  "models": {
    "arima": {"smape": 46.66, "mae": 12.48, "rmse": 18.15},
    "prophet": {"smape": 27.76, "mae": 7.43, "rmse": 11.89},
    "lightgbm": {"smape": 33.38, "mae": 8.91, "rmse": 13.44}
  },
  "winner": "prophet",
  "improvement_over_baseline": "40.5%"
}
```

### 🎨 **Visualization Gallery**

#### **📊 Time Series Cross-Validation Results**
- 3-fold rolling-origin validation görselleştirmesi
- Her fold için train/validation split timeline
- Metric distribution across folds

#### **📈 Prophet Component Analysis**
- Trend decomposition (long-term pattern)
- Weekly seasonality (day-of-week effects)
- Holiday effects (if any)
- Uncertainty intervals (prediction confidence)

#### **🔍 LightGBM Feature Importance**
```python
# Top 5 features (tipik sıralama)
1. roll_mean_7     (0.35) - 7-günlük moving average
2. lag_7          (0.23) - Geçen haftanın aynı günü  
3. roll_mean_28   (0.18) - 28-günlük moving average
4. dow            (0.12) - Day of week effect
5. lag_1          (0.08) - Dün'ün satış değeri
```

#### **🎯 Sales Pattern Analysis**
- Daily sales trend over time
- Seasonal decomposition (trend + seasonal + residual)
- Outlier detection and annotation
- Volume distribution by item

## 🔧 Kod Mimarisi ve Yapısı

### 📁 **Modüler Proje Organizasyonu**
```
📦 hafta5/                              # Ana proje klasörü
├── 📚 README.md                        # Bu kapsamlı dokümantasyon (1,200+ satır)
├── 📋 STRUCTURE.md                     # Modüler yapı detay rehberi  
├── 🐳 DOCKER_README.md                 # Docker deployment rehberi
├── 📦 requirements.txt                 # Python dependencies
├── 🐳 Dockerfile                       # Container definition (modüler uyumlu)
├── 🐳 docker-compose.yml              # Orchestration config
├── 📜 docker-commands.sh              # Docker helper scripts
├── 🎯 run.py                          # Single-run executor (Docker uyumlu)
├── 🔧 run_modular.py                  # Modüler pipeline runner (YENİ!)
│
├── 📊 P1_data_preparation/            # Veri Hazırlama Modülü
│   ├── __init__.py                    # Package definition + import shortcuts
│   ├── create_m5_subset.py            # M5 subset oluşturma (CA_1, FOODS, 5 items)
│   └── create_sample_data.py          # Sentetik veri oluşturma
│
├── ⚙️ P2_feature_engineering/         # Feature Engineering Modülü
│   ├── __init__.py                    # Package definition
│   └── feature_engineering.py        # Lag/rolling/seasonal features
│
├── 📈 P3_traditional_models/          # Geleneksel İstatistiksel Modeller
│   ├── __init__.py                    # Package definition
│   └── arima_single_item.py          # ARIMA(p,d,q) with grid search
│
├── 🚀 P4_modern_models/               # Modern Forecasting Modelleri
│   ├── __init__.py                    # Package definition
│   └── prophet_single_item.py        # Facebook Prophet with components
│
├── 🤖 P5_ml_models/                   # Machine Learning Modelleri
│   ├── __init__.py                    # Package definition
│   └── lightgbm_multi_item.py        # LightGBM multi-item forecasting
│
├── ✅ P6_validation/                  # Model Validation ve Cross-Validation
│   ├── __init__.py                    # Package definition
│   ├── time_series_cv.py             # Comprehensive CV with plots
│   └── time_series_cv_simple.py      # Simplified 3-fold CV
│
├── 🔄 P7_automation/                  # Pipeline Automation
│   ├── __init__.py                    # Package definition
│   └── prefect_demand_forecast.py    # Daily forecasting workflow
│
├── 📦 legacy/                         # Legacy Files
│   └── m5_forecasting.py             # Original monolith implementation
│
├── 📁 data/                           # M5 Dataset Files
│   ├── calendar.csv                  # M5 calendar + events (1,969 days)
│   ├── sales_train_CA.csv            # CA sales data subset
│   └── sell_prices.csv               # Price information
│
└── 📁 artifacts/                      # Outputs ve Model Artifacts
    ├── datasets/                      # Feature engineered datasets
    │   ├── train.csv, valid.csv       # Train/validation splits
    │   ├── fe_train.parquet           # Feature engineered training
    │   └── fe_valid.parquet           # Feature engineered validation
    ├── models/                        # Trained model objects
    │   ├── arima_FOODS_3_090.pkl      # ARIMA model state
    │   ├── prophet_FOODS_3_090.json   # Prophet model serialization
    │   └── lgbm.pkl                   # LightGBM booster object
    ├── figures/                       # Visualizations gallery
    │   ├── arima_*_forecast.png       # ARIMA forecasts + diagnostics
    │   ├── prophet_*_components.png   # Prophet decomposition
    │   ├── lgbm_feature_importance.png # Feature importance plots
    │   ├── feature_correlations.png   # Feature correlation heatmap
    │   └── overall_daily_sales.png    # Sales trend visualization
    ├── preds/                         # Prediction outputs
    │   ├── run_YYYYMMDD.csv          # Daily pipeline CSV output
    │   ├── run_YYYYMMDD_summary.png  # Visual summary report
    │   └── *_forecast_*.csv          # Model-specific predictions
    └── reports/                       # Analysis reports
        ├── tscv_report.json          # Cross-validation results
        ├── tscv_report.md            # Human-readable CV report
        └── model_comparison.json     # Performance benchmarks
```

### 🎮 **Modüler Kullanım Senaryoları**

#### **📚 Eğitim Amaçlı - Adım Adım**
```bash
# Her modülü ayrı ayrı öğren ve çalıştır
python run_modular.py --module P1    # Veri hazırlama
python run_modular.py --module P2    # Feature engineering  
python run_modular.py --module P3    # ARIMA modeling
python run_modular.py --module P4    # Prophet forecasting
python run_modular.py --module P5    # LightGBM ML approach
python run_modular.py --module P6    # Cross-validation
python run_modular.py --module P7    # Production automation
```

#### **🚀 Production - Full Pipeline**
```bash
# Tam pipeline - tüm modülleri sırayla
python run_modular.py

# Docker production deployment
docker build -t m5-forecast:dev .
docker run --rm -v $(pwd)/artifacts:/app/artifacts m5-forecast:dev

# Docker Compose with Prefect server
docker-compose --profile production up -d
```

#### **🔬 Araştırma - Specific Analysis**
```bash
# Model karşılaştırması için sadece modeling modülleri
python run_modular.py --module P3    # ARIMA baseline
python run_modular.py --module P4    # Prophet comparison
python run_modular.py --module P5    # LightGBM benchmark

# Cross-validation deep dive
python run_modular.py --module P6    # Rolling-origin CV analysis
```

#### **🐍 Python Import - Programmatic Usage**
```python
# Modüler import ve kullanım
from P1_data_preparation import create_subset
from P7_automation.prefect_demand_forecast import demand_forecast_flow

# Data hazırla
create_subset()

# Automated forecasting çalıştır
result = demand_forecast_flow(forecast_days=7)
print(f"Forecast generated: {result['csv_path']}")
```

### 🧩 **Ana Modül Yapısı**

#### **P1: Data Preparation Module**
```python
class M5DatasetCreator:
    def __init__(config):           # Konfigürasyon yükleme
    def load_raw_data():           # M5 raw data loading
    def filter_subset():           # State/store/category filtering  
    def create_time_features():    # Basic time features
    def train_test_split():        # Time-based splitting
    def save_processed_data():     # Parquet format save
```

#### **P2: Feature Engineering Module**
```python
class TimeSeriesFeatureEngine:
    def __init__(data_df):          # Input dataframe
    def create_lag_features():      # Historical lag features
    def create_rolling_features():  # Moving averages/std
    def create_seasonal_features(): # Day/week/month encoding
    def handle_missing_values():    # Forward fill strategy
    def validate_features():        # Feature quality checks
```

#### **P3: ARIMA Module**
```python
class ARIMAForecaster:
    def __init__(series):           # Time series input
    def check_stationarity():       # ADF test implementation
    def grid_search_params():       # (p,d,q) optimization
    def fit_best_model():          # Model training
    def generate_forecast():        # Multi-step ahead prediction
    def calculate_metrics():        # Performance evaluation
    def plot_diagnostics():         # Residual analysis
```

#### **P4: Prophet Module**
```python
class ProphetForecaster:
    def __init__(config):           # Prophet configuration
    def prepare_prophet_data():     # ds/y format conversion
    def configure_seasonality():    # Daily/weekly/yearly setup
    def add_custom_regressors():    # External feature addition
    def fit_prophet_model():        # Model training
    def generate_forecast():        # Future dataframe prediction
    def extract_components():       # Trend/seasonal decomposition
```

#### **P5: LightGBM Module**
```python
class LightGBMForecaster:
    def __init__(config):           # Model hyperparameters
    def prepare_ml_features():      # Feature matrix preparation
    def encode_categoricals():      # Label encoding
    def train_model():             # Gradient boosting training
    def iterative_forecast():      # Multi-step prediction
    def analyze_feature_importance(): # SHAP/gain analysis
    def cross_validate():          # Time series CV
```

#### **P6: Cross-Validation Module**
```python
class TimeSeriesCrossValidator:
    def __init__(data, model):      # Data and model setup
    def create_rolling_splits():    # No-shuffle temporal splits
    def validate_temporal_order():  # Leakage prevention
    def run_cv_fold():             # Single fold execution
    def aggregate_results():        # Cross-fold statistics
    def statistical_tests():       # Significance testing
```

#### **P7: Pipeline Automation Module**
```python
@flow
def demand_forecast_flow():
    data = load_data_task()         # Prefect task: data loading
    features = feature_engineer_task(data)  # Feature engineering
    model = train_model_task(features)      # Model training
    preds = predict_task(model, features)   # Prediction generation
    outputs = save_outputs_task(preds)      # Results persistence
    return outputs
```

### 🎓 **Educational Design Patterns**

#### **Defensive Programming**
```python
# Her modülde exception handling
try:
    import prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("⚠️ Prophet bulunamadı, geçiliyor...")
    PROPHET_AVAILABLE = False

# Graceful degradation
if not PROPHET_AVAILABLE:
    prophet_results = {"error": "Prophet not available"}
else:
    prophet_results = run_prophet_forecast()
```

#### **Configuration-Driven Development**
```python
# Tek bir yerde configuration
CONFIG = {
    'data_config': {...},
    'model_config': {...},
    'output_config': {...}
}

# Her modül config'i inherit eder
class BaseForecaster:
    def __init__(self, config=CONFIG):
        self.config = config
```

#### **Logging and Monitoring**
```python
import logging

# Structured logging
logger = logging.getLogger(__name__)
logger.info(f"📊 Veri yüklendi: {data.shape}")
logger.warning(f"⚠️ Missing values: {missing_count}")
logger.error(f"❌ Model eğitim hatası: {error}")
```

#### **Reproducibility Enforcement**
```python
# Seed management
np.random.seed(CONFIG['random_seed'])
random.seed(CONFIG['random_seed'])

# Deterministic model behavior
lgb_params = {
    'objective': 'regression',
    'random_state': CONFIG['random_seed'],
    'deterministic': True
}
```

## 🎯 **Eğitim Hedefleri ve Öğrenme Çıktıları**

### 📚 **Akademik Kazanımlar**

#### **Time Series Fundamentals**
- ✅ **Stationarity Testing**: ADF test ile trend detection
- ✅ **Seasonal Decomposition**: Trend + Seasonal + Residual
- ✅ **Autocorrelation Analysis**: ACF/PACF plots interpretation
- ✅ **Time-based Cross-Validation**: Temporal leakage prevention

#### **Statistical Modeling**
- ✅ **ARIMA Methodology**: Box-Jenkins approach implementation
- ✅ **Parameter Optimization**: Grid search ve information criteria
- ✅ **Model Diagnostics**: Residual analysis ve assumption testing
- ✅ **Forecast Intervals**: Uncertainty quantification

#### **Machine Learning for Time Series**
- ✅ **Feature Engineering**: Lag, rolling, seasonal features
- ✅ **Gradient Boosting**: LightGBM hyperparameter tuning
- ✅ **Multi-step Forecasting**: Iterative prediction strategy
- ✅ **Feature Importance**: SHAP values ve gain analysis

#### **Production MLOps**
- ✅ **Workflow Orchestration**: Prefect tasks ve flows
- ✅ **Containerization**: Docker multi-stage builds
- ✅ **Monitoring**: Logging ve error handling
- ✅ **Scheduling**: Cron-based automated execution

### 🔬 **Hands-on Deneyimler**

#### **Veri Mühendisliği**
```python
# M5 veri setini anlama
- 30.490 unique item-store kombinasyonu
- 1.969 günlük satış verisi (2011-2016)
- 3 eyalet (CA, TX, WI), 10 mağaza, 3 kategori
- Hierarchical structure: State > Store > Category > Item
```

#### **Model Karşılaştırması**
```python
# Objektif model comparison
performance_comparison = {
    'arima_vs_prophet': prophet_improvement,     # ~40% better sMAPE
    'prophet_vs_lightgbm': feature_importance,   # Rolling features critical
    'ensemble_potential': model_combination      # Weighted averaging
}
```

#### **Production Deployment**
```bash
# Container orchestration
docker-compose up --scale m5-forecast=3        # Horizontal scaling
kubectl apply -f k8s-cronjob.yaml             # Kubernetes scheduling
prefect deployment build --cron "0 9 * * *"   # Daily automation
```

## ⚠️ **Bilinen Limitasyonlar ve İyileştirme Alanları**

### 🚧 **Mevcut Kısıtlamalar**

#### **Data Scope**
- ❌ Sadece CA eyaleti (TX, WI dahil değil)
- ❌ Tek mağaza focus (CA_1 only)
- ❌ FOODS kategorisi (HOBBIES, HOUSEHOLD yok)
- ❌ 5 ürün ile sınırlı (30.490'dan sadece 5)

#### **Model Complexity**
- ❌ Ensemble methods yok (stacking, blending)
- ❌ Deep learning models yok (LSTM, Transformer)
- ❌ Hierarchical reconciliation yok
- ❌ External regressors minimal (weather, ekonomik indicators)

#### **Production Features**
- ❌ Real-time inference API yok
- ❌ Model versioning minimal
- ❌ A/B testing framework yok
- ❌ Automated retraining yok

### 🎯 **İyileştirme Roadmap'i**

#### **Kısa Vadeli İyileştirmeler (1-2 hafta)**
```python
# 1. Model diversity artırımı
models_to_add = [
    'ExponentialSmoothing',  # statsmodels Holt-Winters
    'SeasonalNaive',         # Seasonal baseline
    'LinearRegression',      # Trend-based approach  
    'XGBoost'               # Alternative gradient boosting
]

# 2. Feature engineering expansion
new_features = [
    'price_features',        # Price elasticity
    'event_features',        # Calendar events impact
    'weather_features',      # Weather correlation
    'macro_features'         # Economic indicators
]

# 3. Evaluation enhancement
advanced_metrics = [
    'WAPE',                 # Weighted APE
    'MSIS',                 # Mean Scaled Interval Score
    'Directional_Accuracy',  # Up/down prediction accuracy
    'Quantile_Loss'         # Probabilistic forecasting
]
```

#### **Orta Vadeli Geliştirmeler (1-2 ay)**
```python
# 1. Hierarchical forecasting
hierarchy_implementation = {
    'level_1': 'Total_Store_Sales',
    'level_2': 'Category_Sales',  
    'level_3': 'Item_Sales',
    'reconciliation': 'bottom_up + top_down + optimal'
}

# 2. Automated model selection
automl_pipeline = {
    'hyperparameter_optimization': 'Optuna + Bayesian',
    'feature_selection': 'Recursive Feature Elimination',
    'model_selection': 'Cross-validation + early stopping',
    'ensemble_creation': 'Stacking + Blending'
}

# 3. Real-time deployment
production_api = {
    'framework': 'FastAPI + Pydantic',
    'container': 'Docker + Kubernetes',
    'monitoring': 'Prometheus + Grafana',
    'ci_cd': 'GitHub Actions + ArgoCD'
}
```

#### **Uzun Vadeli Vizyon (3-6 ay)**
```python
# 1. Deep learning integration
deep_learning_models = [
    'DeepAR',               # Amazon'un probabilistic model
    'N-BEATS',              # Neural basis expansion
    'Transformer',          # Attention-based forecasting
    'TFT'                   # Temporal Fusion Transformer
]

# 2. Multi-modal forecasting
multi_modal_inputs = {
    'time_series': 'Historical sales data',
    'text_data': 'Product descriptions, reviews',
    'image_data': 'Product images, store layouts',
    'graph_data': 'Store-item relationships'
}

# 3. Causal inference
causal_modeling = {
    'treatment_effects': 'Promo impact measurement',
    'counterfactual': 'What-if scenario analysis',
    'intervention': 'Optimal pricing strategies',
    'attribution': 'Marketing channel effectiveness'
}
```

## 🔧 **Troubleshooting ve Sık Karşılaşılan Problemler**

### 🚨 **Installation Issues**

#### **Prophet Installation Problems**
```bash
# Problem: Prophet binary wheels bulunamıyor
# Solution 1: Conda kullan
conda install -c conda-forge prophet

# Solution 2: Build dependencies yükle
# macOS
brew install cmake

# Ubuntu/Debian
sudo apt-get install build-essential cmake

# Windows
# Visual Studio Build Tools gerekli

# Solution 3: Alternative prophet-forecasting
pip install prophet-forecasting
```

#### **LightGBM Compilation Errors**
```bash
# Problem: OpenMP bulunamıyor (macOS)
# Solution:
brew install libomp
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++

# Problem: CUDA support isteniyor (GPU)
# Solution:
pip install lightgbm --install-option=--gpu

# Problem: Memory allocation error
# Solution: Virtual memory artır veya data size azalt
ulimit -v 8388608  # 8GB virtual memory limit
```

### 📊 **Data Issues**

#### **Missing M5 Data Files**
```bash
# Problem: Kaggle data files bulunamıyor
# Solution 1: Manuel download
echo "1. https://www.kaggle.com/c/m5-forecasting-accuracy/data sayfasına git"
echo "2. sales_train_validation.csv, calendar.csv, sell_prices.csv indir"
echo "3. data/ klasörüne yerleştir"

# Solution 2: Subset kullan
python create_m5_subset.py  # Hazır subset oluştur

# Solution 3: Sample data
python create_sample_data.py  # Sentetik veri
```

#### **Memory Issues with Large Dataset**
```python
# Problem: 30GB+ M5 dataset memory'ye sığmıyor
# Solution 1: Chunked processing
chunk_size = 10000
for chunk in pd.read_csv('sales_train_validation.csv', chunksize=chunk_size):
    process_chunk(chunk)

# Solution 2: Parquet format kullan
df.to_parquet('data.parquet', compression='gzip')  # 10x smaller
df = pd.read_parquet('data.parquet')

# Solution 3: Data types optimize et
df['sales'] = df['sales'].astype('float32')        # 64bit -> 32bit
df['item_id'] = df['item_id'].astype('category')   # String -> Category
```

### 🐳 **Docker Issues**

#### **Container Build Failures**
```bash
# Problem: Docker build slow/failing
# Solution 1: Multi-stage build
FROM python:3.11-slim as builder
RUN pip install --user requirements.txt
FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local

# Solution 2: Cache optimization
COPY requirements.txt .
RUN pip install -r requirements.txt  # Bu layer cache'lenir
COPY . .  # Code değişiklikleri cache'i bozar

# Solution 3: .dockerignore kullan
echo "__pycache__" >> .dockerignore
echo "*.pyc" >> .dockerignore
echo ".git" >> .dockerignore
```

#### **Volume Mount Problems**
```bash
# Problem: Artifacts persist etmiyor
# Solution: Volume permissions check
docker run --rm -v $(pwd)/artifacts:/app/artifacts \
    --user $(id -u):$(id -g) \
    m5-forecast:dev

# Problem: Data files görünmüyor container'da
# Solution: Absolute path kullan
docker run --rm -v /full/path/to/data:/app/data m5-forecast:dev
```

### 🔄 **Prefect Workflow Issues**

#### **Flow Execution Failures**
```python
# Problem: Task fail ediyor ama flow devam ediyor
# Solution: Explicit failure handling
@task
def safe_model_training(data):
    try:
        model = train_model(data)
        return model
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise  # Re-raise to fail the flow

# Problem: Concurrent tasks dependency conflict
# Solution: Task dependencies explicit tanımla
@flow
def forecast_flow():
    data = load_data_task()
    features = feature_engineer_task(data, wait_for=[data])
    model = train_model_task(features, wait_for=[features])
```

#### **Scheduling Problems**
```bash
# Problem: Cron schedule çalışmıyor
# Solution: Timezone explicit set et
from prefect import flow
from prefect.blocks.system import DateTime

@flow
def scheduled_flow():
    pass

# Deployment ile timezone
prefect deployment build scheduled_flow.py:scheduled_flow \
    --cron "0 9 * * *" \
    --timezone "Europe/Istanbul"
```

### 📈 **Model Performance Issues**

#### **Poor Forecast Accuracy**
```python
# Problem: sMAPE > 50% (kötü performance)
# Diagnosis 1: Data quality check
data.isnull().sum()                    # Missing values
data.describe()                        # Outliers
data.plot()                           # Visual inspection

# Diagnosis 2: Feature correlation
correlation_matrix = features.corr()
high_corr = correlation_matrix[correlation_matrix > 0.9]

# Solution 1: Feature engineering improvement
new_features = [
    'sales_lag_14',                    # 2-week lag
    'sales_rolling_std_7',             # Volatility measure
    'sales_trend',                     # Linear trend
    'sales_seasonal_decompose'         # STL decomposition
]

# Solution 2: Hyperparameter tuning
from optuna import create_study
def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('lr', 0.01, 0.3),
        'num_leaves': trial.suggest_int('leaves', 10, 300),
        'min_data_in_leaf': trial.suggest_int('min_data', 5, 100)
    }
    model = lgb.train(params, train_data)
    return -model.best_score['valid_0']['rmse']

study = create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

#### **Cross-Validation Inconsistency**
```python
# Problem: CV scores arasında büyük variance
# Diagnosis: Fold stability check
cv_scores = [0.45, 0.28, 0.51]  # High variance
cv_std = np.std(cv_scores)      # > 0.1 problematic

# Solution 1: More folds
n_folds = 5  # Instead of 3

# Solution 2: Longer validation windows
validation_horizon = 42  # Instead of 28

# Solution 3: Overlap validation
def overlap_cv_splits(data, n_folds=5, horizon=28, overlap=0.5):
    # Each fold overlaps with previous by 50%
    pass
```

## 📚 **Referanslar ve İleri Okuma**

### 📖 **Academic Papers**

#### **Time Series Forecasting**
- **Hyndman, R.J., & Athanasopoulos, G. (2021)**. *Forecasting: Principles and Practice* (3rd ed.)
  - 📍 URL: https://otexts.com/fpp3/
  - 🎯 Coverage: ARIMA, ETS, seasonal decomposition, cross-validation
  - ⭐ Rating: Essential time series textbook

- **Taylor, S.J., & Letham, B. (2018)**. *Forecasting at Scale*. The American Statistician, 72(1), 37-45.
  - 📍 URL: https://peerj.com/preprints/3190/
  - 🎯 Coverage: Prophet methodology, decomposable time series models
  - ⭐ Rating: Prophet original paper

#### **Machine Learning for Time Series**
- **Januschowski, T., et al. (2020)**. *Criteria for classifying forecasting methods*. International Journal of Forecasting, 36(1), 167-177.
  - 🎯 Coverage: ML vs statistical methods comparison
  - ⭐ Rating: Methodology comparison framework

- **Oreshkin, B.N., et al. (2019)**. *N-BEATS: Neural basis expansion analysis for interpretable time series forecasting*. ICLR 2020.
  - 📍 URL: https://arxiv.org/abs/1905.10437
  - 🎯 Coverage: Deep learning for time series, neural basis expansion
  - ⭐ Rating: State-of-the-art neural forecasting

#### **Cross-Validation for Time Series**
- **Bergmeir, C., & Benítez, J.M. (2012)**. *On the use of cross-validation for time series predictor evaluation*. Information Sciences, 191, 192-213.
  - 🎯 Coverage: Temporal leakage, proper CV for time series
  - ⭐ Rating: CV methodology for time series

### 🛠️ **Technical Documentation**

#### **Libraries and Frameworks**
- **Prophet Documentation**: https://facebook.github.io/prophet/
  - 📖 User Guide, API Reference, Case Studies
  - 🔧 Installation troubleshooting, parameter tuning
  
- **LightGBM Documentation**: https://lightgbm.readthedocs.io/
  - 📖 Parameter reference, Python API
  - 🔧 GPU acceleration, distributed training

- **Prefect Documentation**: https://docs.prefect.io/
  - 📖 Workflow orchestration, task management
  - 🔧 Deployment patterns, cloud integration

- **Statsmodels Documentation**: https://www.statsmodels.org/stable/tsa.html
  - 📖 Time series analysis, ARIMA implementation
  - 🔧 Statistical tests, model diagnostics

#### **Dataset and Competition**
- **M5 Competition Kaggle**: https://www.kaggle.com/c/m5-forecasting-accuracy
  - 📊 Dataset download, competition leaderboard
  - 💬 Discussion forum, winning solutions
  
- **M5 Competition Paper**: Makridakis, S., et al. (2022). *The M5 competition: Background, organization, and implementation*. International Journal of Forecasting, 38(4), 1325-1336.
  - 🎯 Coverage: Competition design, evaluation metrics, lessons learned

### 🎓 **Online Courses and Tutorials**

#### **Time Series Forecasting**
- **Coursera: Time Series Forecasting** (University of Washington)
  - 📍 URL: https://coursera.org/learn/time-series-forecasting
  - 🎯 Coverage: Classical methods, ARIMA, seasonality
  - ⏱️ Duration: 4 weeks

- **Fast.ai: Practical Deep Learning for Time Series**
  - 📍 URL: https://course.fast.ai/
  - 🎯 Coverage: Deep learning applications, transfer learning
  - ⏱️ Duration: Self-paced

#### **MLOps and Production**
- **Google Cloud: MLOps Specialization**
  - 📍 URL: https://coursera.org/specializations/machine-learning-engineering-for-production-mlops
  - 🎯 Coverage: ML pipelines, monitoring, deployment
  - ⏱️ Duration: 4 courses

### 🏆 **Competition Solutions and Case Studies**

#### **M5 Competition Winners**
- **1st Place Solution**: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/163684
  - 🔧 LightGBM ensemble, feature engineering, hierarchical reconciliation
  
- **2nd Place Solution**: https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/164374
  - 🔧 Multi-level modeling, uncertainty quantification

#### **Industry Case Studies**
- **Uber Forecasting**: https://eng.uber.com/forecasting-introduction/
  - 🏢 Real-world scaling challenges, operational forecasting
  
- **Netflix Demand Forecasting**: https://netflixtechblog.com/forecasting-at-netflix-5ae9ea92c51d
  - 🏢 Content demand prediction, recommendation systems

### 🔬 **Research Communities and Forums**

#### **Academic Communities**
- **International Institute of Forecasters**: https://forecasters.org/
  - 📚 Journal access, conference proceedings
  
- **NIPS Time Series Workshop**: https://neurips.cc/
  - 🔬 Latest research, deep learning applications

#### **Practitioner Communities**
- **Stack Overflow - Time Series**: https://stackoverflow.com/questions/tagged/time-series
  - 💬 Technical Q&A, implementation help
  
- **Reddit - Machine Learning**: https://reddit.com/r/MachineLearning
  - 💬 Research discussions, paper reviews

- **Kaggle Learn - Time Series**: https://www.kaggle.com/learn/time-series
  - 📖 Interactive tutorials, hands-on exercises

### 📊 **Datasets for Practice**

#### **Public Time Series Datasets**
- **UCR Time Series Archive**: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
  - 📊 128 classification datasets, various domains
  
- **Federal Reserve Economic Data (FRED)**: https://fred.stlouisfed.org/
  - 📊 Economic indicators, financial time series
  
- **Google Trends**: https://trends.google.com/
  - 📊 Search volume data, cultural trends

#### **Synthetic Data Generators**
- **TSlearn Datasets**: https://tslearn.readthedocs.io/en/stable/gen_modules/tslearn.datasets.html
  - 🔧 Synthetic time series with known patterns
  
- **Darts Synthetic Data**: https://unit8co.github.io/darts/
  - 🔧 Configurable synthetic time series generator

---

## 🎓 **Proje Özeti ve Migration Rehberi**

### 🆕 **Bu Sürümdeki Yenilikler (v2.0)**

#### **🏗️ Modüler Mimari**
- **7 Ana Modül**: P1-P7 arası sistemli organizasyon
- **Package Structure**: Her modülde `__init__.py` ve import shortcuts
- **Backward Compatibility**: Eski dosyalar `legacy/` klasöründe korundu

#### **🎮 Yeni Çalıştırma Yöntemleri**
- **`run_modular.py`**: Module-specific execution (`--module P1`)
- **Enhanced Docker**: Modüler yapı destekli Dockerfile
- **Programmatic API**: Python import pattern için shortcuts

#### **📚 Kapsamlı Dokümantasyon**
- **1,400+ satır README**: Dataset hikayesi, modüler mimari, troubleshooting
- **STRUCTURE.md**: Detaylı yapısal rehber
- **DOCKER_README.md**: Container deployment rehberi

### 🔄 **Migration Rehberi (v1.0 → v2.0)**

#### **Eski Kullanım → Yeni Kullanım**
```bash
# ESKİ (v1.0)
python m5_forecasting.py           # Monolith approach
python arima_single_item.py        # Direct script execution

# YENİ (v2.0) 
python run_modular.py              # Full modular pipeline
python run_modular.py --module P3  # Specific module (ARIMA)
python P3_traditional_models/arima_single_item.py  # Direct access (korundu)
```

#### **Import Pattern Değişiklikleri**
```python
# ESKİ (v1.0)
from arima_single_item import main
from prefect_demand_forecast import demand_forecast_flow

# YENİ (v2.0)
from P3_traditional_models import run_arima  # Package level
from P7_automation.prefect_demand_forecast import demand_forecast_flow  # Module level

# FALLBACK (backward compatibility)
try:
    from P3_traditional_models import run_arima
except ImportError:
    from arima_single_item import main as run_arima  # Legacy support
```

### 📋 **Proje Karşılaştırması: Önce vs Sonra**

| Özellik | v1.0 (Monolith) | v2.0 (Modular) |
|---------|-----------------|-----------------|
| **Dosya Sayısı** | 11 Python dosyası | 7 modül + 16 dosya |
| **Organizasyon** | Flat structure | Hierarchical modules |
| **Çalıştırma** | Script-by-script | Module-by-module |
| **Import** | Direct imports | Package imports |
| **Docker** | Basic Dockerfile | Modular-aware container |
| **Documentation** | Basic README | 1,400+ line docs |
| **Testing** | Individual scripts | Module isolation |
| **Scalability** | Monolith deployment | Microservice ready |

### 🎯 **Öğrenme Yolculuğu Rehberi**

#### **🌟 Beginner Level (Hafta 1-2)**
```bash
# Modülleri sırayla öğren
python run_modular.py --module P1    # Veri anlama
python run_modular.py --module P2    # Feature engineering
python run_modular.py --module P3    # İlk model (ARIMA)
```

#### **🚀 Intermediate Level (Hafta 3-4)**
```bash
# Model karşılaştırması
python run_modular.py --module P4    # Prophet
python run_modular.py --module P5    # LightGBM
python run_modular.py --module P6    # Cross-validation
```

#### **🏆 Advanced Level (Hafta 5-6)**
```bash
# Production deployment
python run_modular.py --module P7    # Automation
docker build -t m5-forecast:dev .    # Containerization
docker-compose --profile production up -d  # Orchestration
```

### 📊 **Başarı Metrikleri ve Beklentiler**

#### **Model Performance Beklentileri**
- **ARIMA**: sMAPE ~46% (Baseline istatistiksel model)
- **Prophet**: sMAPE ~28% (En iyi single-model performance)
- **LightGBM**: sMAPE ~33% (Feature engineering sayesinde)

#### **Öğrenme Çıktıları**
- ✅ **Time Series Fundamentals**: Stationarity, seasonality, lag features
- ✅ **Model Comparison**: Statistical vs ML approaches
- ✅ **Production Skills**: Docker, Prefect, automated pipelines
- ✅ **Code Organization**: Modular architecture, package management

### 📝 **Contribution Guidelines**

Bu proje eğitim amaçlıdır ve katkılara açıktır. Modüler yapıda katkıda bulunmak için:

#### **🏗️ Modül Bazlı Katkı**
1. 🍴 Repository'yi fork edin
2. 🎯 Specific module'ü seçin (P1-P7)
3. 🌟 Module-specific branch oluşturun (`git checkout -b P3-enhancement`)
4. 📝 Değişikliklerinizi ilgili modülde yapın
5. 🧪 Module testing: `python run_modular.py --module P3`
6. 📤 Branch'i push edin (`git push origin P3-enhancement`)
7. 🔀 Pull Request oluşturun

#### **📋 Katkı Alanları**
- **P1**: Yeni data sources, preprocessing improvements
- **P2**: Advanced feature engineering, external data
- **P3**: Alternative statistical models (ETS, TBATS)
- **P4**: Prophet hyperparameter optimization, custom regressors
- **P5**: Ensemble methods, deep learning models
- **P6**: Advanced cross-validation, statistical tests
- **P7**: Real-time deployment, monitoring, alerts

#### **🎯 Code Standards**
- Her modüle `__init__.py` ekleyin
- Import shortcuts sağlayın
- Backward compatibility koruyun
- Comprehensive logging ekleyin
- Unit tests yazın (`test_P1.py`, `test_P2.py`, etc.)

### 📄 **License**

Bu proje MIT License altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

### 🤝 **Acknowledgments**

- **M5 Competition Organizers**: Walmart Labs, University of Nicosia
- **Library Maintainers**: Statsmodels, Prophet, LightGBM, Prefect teams
- **Educational Inspiration**: Fast.ai, Coursera Time Series courses
- **Community Support**: Kaggle community, Stack Overflow contributors

---

**⚠️ Disclaimer**: Bu proje eğitim amaçlıdır. Production ortamında kullanmadan önce kapsamlı testing, validation ve optimization yapılması önerilir. Financial ve business critical kararlar için profesyonel danışmanlık alınmalıdır.
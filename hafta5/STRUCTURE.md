# M5 Forecasting - Proje Yapısı

Bu dokümantasyon, projenin organize edilmiş modüler yapısını açıklar.

## 📁 Dizin Yapısı

```
hafta5/                              # 🏠 Ana proje klasörü
├── README.md                        # 📚 Kapsamlı proje dokümantasyonu (1,183 satır)
├── STRUCTURE.md                     # 📋 Bu dosya - yapısal rehber
├── DOCKER_README.md                 # 🐳 Docker deployment rehberi
├── requirements.txt                 # 📦 Python dependencies
├── Dockerfile                       # 🐳 Container definition
├── docker-compose.yml              # 🐳 Orchestration config
├── docker-commands.sh              # 📜 Docker helper script
├── run.py                          # 🎯 Ana single-run executor (Docker uyumlu)
├── run_modular.py                  # 🔧 Modüler pipeline runner
│
├── P1_data_preparation/            # 📊 Veri Hazırlama Modülü
│   ├── __init__.py                 # Module definition
│   ├── create_m5_subset.py         # M5 subset oluşturma
│   └── create_sample_data.py       # Sentetik veri oluşturma
│
├── P2_feature_engineering/         # ⚙️ Feature Engineering Modülü
│   ├── __init__.py                 # Module definition
│   └── feature_engineering.py     # Lag/rolling/seasonal features
│
├── P3_traditional_models/          # 📈 Geleneksel Modeller
│   ├── __init__.py                 # Module definition
│   └── arima_single_item.py       # ARIMA implementation
│
├── P4_modern_models/               # 🚀 Modern Modeller
│   ├── __init__.py                 # Module definition
│   └── prophet_single_item.py     # Facebook Prophet
│
├── P5_ml_models/                   # 🤖 Machine Learning Modelleri
│   ├── __init__.py                 # Module definition
│   └── lightgbm_multi_item.py     # LightGBM multi-item forecasting
│
├── P6_validation/                  # ✅ Model Validation
│   ├── __init__.py                 # Module definition
│   ├── time_series_cv.py          # Comprehensive cross-validation
│   └── time_series_cv_simple.py   # Simplified cross-validation
│
├── P7_automation/                  # 🔄 Otomatizasyon ve Pipeline
│   ├── __init__.py                 # Module definition
│   └── prefect_demand_forecast.py # Prefect workflow orchestration
│
├── legacy/                         # 📦 Legacy dosyalar
│   └── m5_forecasting.py          # Orijinal monolith implementation
│
├── data/                           # 📁 Veri dosyaları
│   ├── calendar.csv               # M5 calendar data
│   ├── sales_train_CA.csv         # Sales data (subset)
│   └── sell_prices.csv            # Price data
│
├── artifacts/                      # 📁 Çıktılar ve model artifacts
│   ├── datasets/                   # Feature engineered data
│   │   ├── train.csv
│   │   ├── valid.csv
│   │   ├── fe_train.parquet
│   │   └── fe_valid.parquet
│   ├── models/                     # Trained models
│   │   ├── arima_FOODS_3_090.pkl
│   │   ├── prophet_FOODS_3_090.json
│   │   └── lgbm.pkl
│   ├── figures/                    # Visualizations
│   │   ├── arima_forecast.png
│   │   ├── prophet_components.png
│   │   ├── lgbm_feature_importance.png
│   │   └── model_comparison.png
│   ├── preds/                      # Prediction outputs
│   │   ├── run_YYYYMMDD.csv       # Daily pipeline output
│   │   └── run_YYYYMMDD_summary.png
│   └── reports/                    # Analysis reports
│       ├── tscv_report.json
│       ├── tscv_report.md
│       └── model_comparison.json
│
└── venv/                          # 🐍 Python virtual environment
```

## 🎯 Modül Açıklamaları

### **P1: Data Preparation Module**
**Purpose**: M5 Competition verisinin hazırlanması ve preprocessing
- **create_m5_subset.py**: CA eyaleti, CA_1 mağazası, FOODS kategorisi subset'i
- **create_sample_data.py**: Test için sentetik veri oluşturma

**Usage**:
```bash
python P1_data_preparation/create_m5_subset.py
python run_modular.py --module P1
```

### **P2: Feature Engineering Module**
**Purpose**: Time series için feature engineering
- **feature_engineering.py**: Lag (1,7,28), rolling (7,28), seasonal features

**Usage**:
```bash
python P2_feature_engineering/feature_engineering.py
python run_modular.py --module P2
```

### **P3: Traditional Models Module**
**Purpose**: İstatistiksel time series modelleri
- **arima_single_item.py**: ARIMA(p,d,q) grid search, ADF test, Box-Jenkins

**Usage**:
```bash
python P3_traditional_models/arima_single_item.py
python run_modular.py --module P3
```

### **P4: Modern Models Module**  
**Purpose**: Modern forecasting yaklaşımları
- **prophet_single_item.py**: Facebook Prophet, seasonality decomposition

**Usage**:
```bash
python P4_modern_models/prophet_single_item.py
python run_modular.py --module P4
```

### **P5: ML Models Module**
**Purpose**: Machine learning tabanlı forecasting
- **lightgbm_multi_item.py**: Multi-item gradient boosting, feature importance

**Usage**:
```bash
python P5_ml_models/lightgbm_multi_item.py
python run_modular.py --module P5
```

### **P6: Validation Module**
**Purpose**: Model validation ve cross-validation
- **time_series_cv.py**: Comprehensive rolling-origin CV
- **time_series_cv_simple.py**: Simplified 3-fold CV

**Usage**:
```bash
python P6_validation/time_series_cv_simple.py
python run_modular.py --module P6
```

### **P7: Automation Module**
**Purpose**: Pipeline orchestration ve automation
- **prefect_demand_forecast.py**: Daily forecasting workflow

**Usage**:
```bash
python P7_automation/prefect_demand_forecast.py
python run_modular.py --module P7
```

## 🚀 Kullanım Senaryoları

### **1. Eğitim Amaçlı - Adım Adım**
```bash
# Her modülü ayrı ayrı çalıştır
python run_modular.py --module P1    # Data preparation
python run_modular.py --module P2    # Feature engineering
python run_modular.py --module P3    # ARIMA
python run_modular.py --module P4    # Prophet
python run_modular.py --module P5    # LightGBM
python run_modular.py --module P6    # Cross-validation
python run_modular.py --module P7    # Automation
```

### **2. Full Pipeline - Otomatik**
```bash
# Tüm modülleri sırayla çalıştır
python run_modular.py

# Veya Docker ile
docker build -t m5-forecast:dev .
docker run --rm -v $(pwd)/artifacts:/app/artifacts m5-forecast:dev
```

### **3. Production - Specific Module**
```bash
# Sadece prediction pipeline
python run.py

# Sadece cross-validation
python run_modular.py --module P6

# Docker production
docker-compose --profile production up -d
```

### **4. Development - Module Import**
```python
# Python script içinde modül import
from P1_data_preparation import create_subset
from P7_automation.prefect_demand_forecast import demand_forecast_flow

# Data hazırla
create_subset()

# Pipeline çalıştır
result = demand_forecast_flow(forecast_days=7)
```

## 🔧 Import Yapısı

### **Modüler Import (Yeni)**
```python
# Modül bazlı import
from P3_traditional_models.arima_single_item import main as run_arima
from P4_modern_models.prophet_single_item import main as run_prophet
from P5_ml_models.lightgbm_multi_item import main as run_lightgbm
from P7_automation.prefect_demand_forecast import demand_forecast_flow

# Package level import (init.py sayesinde)
from P3_traditional_models import run_arima
from P4_modern_models import run_prophet
```

### **Legacy Import (Backward Compatible)**
```python
# Fallback eski yapı (run.py'da mevcut)
try:
    from P7_automation.prefect_demand_forecast import demand_forecast_flow
except ImportError:
    from prefect_demand_forecast import demand_forecast_flow
```

## 📊 Benefits of Modular Structure

### **🎓 Educational Benefits**
- **Clear Separation**: Her modül tek responsibility'ye odaklanır
- **Step-by-step Learning**: Öğrenciler modülleri ayrı ayrı anlayabilir
- **Reusable Components**: Modüller başka projelerde tekrar kullanılabilir
- **Easy Testing**: Her modül bağımsız test edilebilir

### **🔧 Development Benefits**
- **Maintainability**: Kod bakımı ve güncelleme kolaylaşır
- **Debugging**: Hata ayıklama specific modülle sınırlanır
- **Collaboration**: Farklı geliştiriciler farklı modüllerde çalışabilir
- **Version Control**: Modül bazlı commit'ler daha anlamlı

### **🚀 Production Benefits**
- **Scalability**: Modüller independent deploy edilebilir
- **Performance**: Sadece gerekli modüller yüklenebilir
- **Resource Management**: Memory ve CPU usage optimize edilebilir
- **Docker Optimization**: Layer caching daha etkili

### **📦 Packaging Benefits**
- **Independent Deployment**: Her modül ayrı container olabilir
- **Microservices Ready**: Service-oriented architecture'a hazır
- **API Integration**: Her modül REST API olarak expose edilebilir
- **Cloud Native**: Kubernetes, serverless deployment uyumlu

## 🎯 Migration Guide

### **From Legacy to Modular**
```bash
# Eski kullanım
python m5_forecasting.py

# Yeni kullanım
python run_modular.py
# veya
python run.py  # Docker uyumlu
```

### **Import Updates**
```python
# Eski
from arima_single_item import main

# Yeni
from P3_traditional_models.arima_single_item import main
# veya
from P3_traditional_models import run_arima
```

## 📝 Next Steps

1. **API Layer**: FastAPI ile REST endpoints
2. **Microservices**: Her modül ayrı container/service
3. **CI/CD Pipeline**: Module-specific testing ve deployment
4. **Monitoring**: Module-level logging ve metrics
5. **Documentation**: Sphinx ile automatic API docs
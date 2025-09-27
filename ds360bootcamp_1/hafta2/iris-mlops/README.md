# 🚢 Titanic MLOps Projesi

Titanic veri seti ile kapsamlı MLOps pipeline örneği. Bu proje DVC, MLflow, FastAPI, Docker ve GitHub Actions kullanarak modern ML deployment'ı öğretir.

## 📁 Proje Yapısı

```
titanic-mlops/
├── src/                        # Kaynak kodlar
│   ├── api.py                  # FastAPI servisi
│   ├── clean_data.py           # Veri temizleme
│   ├── download_data.py        # Veri indirme
│   ├── train_model.py          # Model eğitimi
│   └── train_model_mlflow.py   # MLflow ile eğitim
├── data/                       # Veri dosyaları
│   ├── raw/                    # Ham veri
│   └── processed/              # İşlenmiş veri
├── models/                     # Eğitilmiş modeller
│   ├── features.json           # Model özellikleri
│   └── metrics.json            # Model metrikleri
├── tests/                      # Test dosyaları
│   └── test_model.py           # Model testleri
├── notebooks/                  # Jupyter notebook'lar
├── mlruns/                     # MLflow deney kayıtları
├── .github/workflows/          # CI/CD pipeline
│   └── ci.yml                  # GitHub Actions workflow
├── .dvc/                       # DVC yapılandırması
├── dvc.yaml                    # DVC pipeline
├── dvc.lock                    # DVC lock dosyası
├── .dvcignore                  # DVC ignore dosyası
├── .dockerignore               # Docker ignore dosyası
├── Dockerfile                  # Docker yapılandırması
├── requirements.txt            # Python bağımlılıkları
├── test_api.py                 # API testleri
└── README.md                   # Proje dokümantasyonu
```

## 🚀 Hızlı Başlangıç

### 1. Projeyi Klonla ve Kurulum Yap

```bash
git clone <repo-url>
cd titanic-mlops
pip install -r requirements.txt
```

### 2. DVC Pipeline Çalıştır

```bash
# Veri hazırlama + model eğitimi
dvc repro

# Sonuçları kontrol et
dvc metrics show
```

### 3. FastAPI Servisini Başlat

```bash
uvicorn src.api:app --reload
```

API dokümantasyonu: http://localhost:8000/docs

### 4. MLflow ile Model Eğitimi ve Takibi

```bash
# MLflow bağımlılığını yükle
pip install mlflow==2.7.1

# MLflow ile model eğit
python src/train_model_mlflow.py

# MLflow UI'yi başlat
mlflow ui
```

MLflow arayüzü: http://localhost:5000

### 5. Docker ile Çalıştır

```bash
# Image oluştur
docker build -t titanic-api .

# Container çalıştır
docker run -p 8000:8000 titanic-api
```

### 6. Testleri Çalıştır

```bash
# Model testleri
python -m pytest tests/ -v

# API testleri
python test_api.py
```

## 🛠️ Kullanılan Teknolojiler

### Data & Model Management
- **DVC**: Veri ve model versiyonlama
- **MLflow**: Deney takibi ve model registry

### API & Deployment  
- **FastAPI**: REST API framework
- **Docker**: Containerization
- **Uvicorn**: ASGI server

### CI/CD & Testing
- **GitHub Actions**: Otomatik pipeline
- **Pytest**: Unit testler

## 📊 API Endpoints

| Endpoint | Method | Açıklama |
|----------|--------|----------|
| `/` | GET | Ana sayfa |
| `/health` | GET | Sağlık kontrolü |
| `/predict` | POST | Hayatta kalma tahmini |
| `/model/info` | GET | Model bilgileri |
| `/predict/example` | GET | Örnek request format |

### Örnek Kullanım

#### cURL ile:
```bash
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "Pclass": 1,
    "Sex": "female", 
    "Age": 25.0,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 100.0,
    "Embarked": "S"
  }'
```

#### Swagger UI Test Örnekleri:

**Zengin kadın yolcu (yüksek hayatta kalma şansı):**
```json
{
  "Pclass": 1,
  "Sex": "female",
  "Age": 25.0,
  "SibSp": 0,
  "Parch": 0,
  "Fare": 100.0,
  "Embarked": "S"
}
```

**Fakir erkek yolcu (düşük hayatta kalma şansı):**
```json
{
  "Pclass": 3,
  "Sex": "male",
  "Age": 30.0,
  "SibSp": 0,
  "Parch": 0,
  "Fare": 8.5,
  "Embarked": "S"
}
```

**Orta sınıf aileli kadın:**
```json
{
  "Pclass": 2,
  "Sex": "female",
  "Age": 35.0,
  "SibSp": 1,
  "Parch": 1,
  "Fare": 25.0,
  "Embarked": "C"
}
```

## 🔄 DVC Pipeline

Pipeline aşamaları:

1. **data_preparation**: Veri indirme ve temizleme
2. **train_model**: Model eğit ve kaydet

```bash
# Pipeline çalıştır
dvc repro

# Değişiklikleri göster  
dvc status

# Remote'a gönder
dvc push
```

## 🧪 MLflow Deney Takibi

```bash
# MLflow ile model eğit
python src/train_model_mlflow.py

# MLflow UI'yi başlat
mlflow ui
```

MLflow arayüzü: http://localhost:5000

## 🚀 GitHub Actions CI/CD

Pipeline özellikleri:

- ✅ **Test**: Veri hazırlama, model eğitimi, API testleri
- ✅ **Docker**: Image build ve container testi  
- ✅ **DVC**: Pipeline doğrulama

Her push ve PR'da otomatik çalışır.

## 📈 Model Performansı

| Model | Accuracy | 
|-------|----------|
| Logistic Regression | ~73% |
| Random Forest | ~68% |

En iyi performanslı model (Logistic Regression) API'de kullanılır.

## 🔧 Geliştirme

### Yeni Model Eklemek

1. `src/train_model.py` dosyasını düzenle
2. `dvc repro` ile pipeline çalıştır
3. `test_api.py` ile test et

### Yeni Özellik Eklemek

1. `src/clean_data.py` veya `src/download_data.py` dosyasını güncelle
2. Model eğitimini tekrarla
3. API endpoint'ini güncelle

## 🐛 Sorun Giderme

### Model Yükleme Hatası
```bash
# Modeli yeniden eğit
python src/train_model.py
```

### Docker Build Hatası
```bash
# Cache'i temizle
docker system prune -f
docker build --no-cache -t titanic-api .
```

### DVC Hatası
```bash
# DVC cache'i temizle
dvc cache dir
rm -rf .dvc/cache
dvc repro
```

## 📚 Öğrenilen Konular

Bu projede öğrenilen MLOps kavramları:

- ✅ Veri versiyonlama (DVC)
- ✅ Model registry ve deney takibi (MLflow)  
- ✅ API geliştirme (FastAPI)
- ✅ Containerization (Docker)
- ✅ CI/CD pipeline (GitHub Actions)
- ✅ Test otomasyonu (Pytest)

## 📄 Lisans

Bu proje eğitim amaçlıdır ve MIT lisansı altında sunulmaktadır.
# MLOps ve Deployment Temelleri - Kapsamlı Rehber

Bu dokümanda 5 farklı MLOps aracının temellerini tek bir rehberde bulacaksınız.

## 🚀 Başlangıç

```bash
# Sanal ortamı aktif et
source ds360/bin/activate

# Tüm projeler için gerekli paketler yüklü
```

---

## 1️⃣ DVC ile Veri Versiyonlama

### Ne İşe Yarar?
- Büyük veri dosyalarını Git'te takip etmek yerine DVC ile takip ederiz
- Veri değişikliklerini versiyon olarak saklarız
- Takım çalışmasında veri senkronizasyonu sağlarız

### Temel Komutlar
```bash
cd 01-dvc-versioning

# DVC başlat
dvc init

# Veri dosyası oluştur
python create_data.py

# DVC ile takip et
dvc add data.csv

# Git'e commit et
git add data.csv.dvc .gitignore
git commit -m "İlk veri versiyonu"
```

### Önemli Dosyalar
- `data.csv.dvc` - DVC metadata dosyası (Git'te saklanır)
- `data.csv` - Gerçek veri (DVC tarafından takip edilir)
- `.dvcignore` - DVC'nin ignore ettiği dosyalar

### Pratik Örnek
```python
# create_data.py
import pandas as pd

data = {
    'isim': ['Ali', 'Ayşe', 'Mehmet', 'Fatma'],
    'yas': [25, 30, 35, 28],
    'maas': [5000, 6000, 7000, 5500]
}

df = pd.DataFrame(data)
df.to_csv('data.csv', index=False)
```

---

## 2️⃣ MLflow ile Deney Takibi

### Ne İşe Yarar?
- Model eğitim süreçlerini kaydetmek
- Farklı hiperparametreleri karşılaştırmak
- Model performansını takip etmek
- Eğitilmiş modelleri saklamak

### Temel Kavramlar
- **Experiment**: Deney grubu (örn: "Kredi Risk Modeli")
- **Run**: Tek bir model eğitimi
- **Parameter**: Model ayarları (learning_rate, n_estimators)
- **Metric**: Sonuç metrikleri (accuracy, precision)
- **Artifact**: Dosyalar (model, grafik, log)

### Kullanım
```bash
cd 02-mlflow-tracking

# Model eğit ve kaydet
python basit_model.py

# MLflow UI başlat
mlflow ui

# Tarayıcıda görüntüle: http://localhost:5000
```

### Pratik Örnek
```python
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression

# Experiment oluştur
mlflow.set_experiment("Basit Cinsiyet Tahmini")

with mlflow.start_run():
    # Model eğit
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Sonuçları kaydet
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", 0.85)
    mlflow.sklearn.log_model(model, "model")
```

---

## 3️⃣ FastAPI ile ML API Geliştirme

### Ne İşe Yarar?
- Eğitilmiş modelleri web API'si olarak serve etmek
- HTTP istekleriyle model tahminleri almak
- Otomatik API dokümantasyonu oluşturmak

### Temel Kavramlar
- **Endpoint**: API URL'leri (GET /health, POST /predict)
- **Pydantic Model**: Veri validation için
- **Request/Response**: Giriş ve çıkış formatları
- **Automatic Docs**: Otomatik oluşan API dokümantasyonu

### Kullanım
```bash
cd 03-fastapi-ml-api

# API'yi başlat
uvicorn app:app --reload

# Test et
curl -X POST "http://localhost:8000/tahmin" \
  -H "Content-Type: application/json" \
  -d '{"boy": 175.0, "kilo": 70.0}'

# Python ile test
python test_api.py
```

### Pratik Örnek
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Basit ML API")

class Kisi(BaseModel):
    boy: float
    kilo: float

class Tahmin(BaseModel):
    cinsiyet: str
    olasilik: float

@app.post("/tahmin", response_model=Tahmin)
def tahmin_yap(kisi: Kisi):
    # Basit kural: boy > 170 ise erkek
    if kisi.boy > 170:
        return Tahmin(cinsiyet="erkek", olasilik=0.8)
    else:
        return Tahmin(cinsiyet="kadın", olasilik=0.7)
```

### Faydalı URL'ler
- `http://localhost:8000` - Ana sayfa
- `http://localhost:8000/docs` - Swagger UI (otomatik docs)
- `http://localhost:8000/redoc` - ReDoc (alternatif docs)

---

## 4️⃣ Docker ile Containerization

### Ne İşe Yarar?
- Uygulamayı tüm bağımlılıklarıyla birlikte paketlemek
- "Benim bilgisayarımda çalışıyor" problemini çözmek
- Farklı ortamlarda tutarlı çalışma garantisi
- Kolay deployment ve scaling

### Temel Kavramlar
- **Image**: Uygulamanın şablonu
- **Container**: Image'den çalışan instance
- **Dockerfile**: Image'i nasıl oluşturacağını anlatan dosya
- **Port Mapping**: Container'daki port'u host'a bağlama

### Kullanım
```bash
cd 04-docker-deployment

# Docker image oluştur
docker build -t ml-api .

# Container çalıştır
docker run -p 8000:8000 ml-api

# Test et: http://localhost:8000
```

### Dockerfile Anatomy
```dockerfile
# Base image seç
FROM python:3.10-slim

# Çalışma dizini
WORKDIR /app

# Dependencies kopyala ve yükle
COPY requirements.txt .
RUN pip install -r requirements.txt

# Uygulama dosyalarını kopyala
COPY app.py .

# Port aç
EXPOSE 8000

# Başlangıç komutu
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Faydalı Docker Komutları
```bash
# Image'ları listele
docker images

# Container'ları listele
docker ps

# Container'ı durdur
docker stop <container_id>

# Container'ı sil
docker rm <container_id>

# Log'ları gör
docker logs <container_id>
```

---

## 5️⃣ GitHub Actions ile CI/CD Pipeline

### Ne İşe Yarar?
- Kod her değişikliğinde otomatik testler çalıştırmak
- Deployment süreçlerini otomatikleştirmek
- Kod kalitesini sürekli kontrol etmek
- Takım üyelerini sonuçlardan haberdar etmek

### Temel Kavramlar
- **Workflow**: Otomasyon süreci (.github/workflows/test.yml)
- **Job**: Paralel çalışan görev grupları
- **Step**: Sıralı işlem adımları
- **Trigger**: Workflow'u tetikleyen olaylar (push, PR)
- **Runner**: Workflow'un çalıştığı sanal makine

### Kullanım
```bash
cd 05-github-actions-ci

# Lokal testleri çalıştır
python test_app.py
python app.py

# GitHub'a push et
git init
git add .
git commit -m "İlk commit"
git push
```

### Workflow Örneği
```yaml
name: Test Uygulaması

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Python 3.10 kurulumu
      uses: actions/setup-python@v3
      with:
        python-version: '3.10'
    
    - name: Testleri çalıştır
      run: python test_app.py
```

### Yaygın Actions
- `actions/checkout@v3` - Repository'yi clone'la
- `actions/setup-python@v3` - Python environment kur
- `actions/setup-node@v3` - Node.js environment kur
- `actions/upload-artifact@v3` - Dosya yükle
- `actions/download-artifact@v3` - Dosya indir

---

## 🔄 Tam MLOps Workflow Örneği

### Gerçek Dünya Senaryosu

1. **Veri Bilimci** yeni veri ekler
   ```bash
   # Yeni veri versiyonu
   dvc add new_data.csv
   git add new_data.csv.dvc
   git commit -m "Yeni eğitim verisi eklendi"
   ```

2. **Model geliştirme** ve deney takibi
   ```python
   # MLflow ile model eğitimi
   with mlflow.start_run():
       model = RandomForestClassifier(n_estimators=100)
       model.fit(X_train, y_train)
       mlflow.log_metric("accuracy", accuracy)
       mlflow.sklearn.log_model(model, "rf_model")
   ```

3. **API geliştirme**
   ```python
   # FastAPI ile model serve etme
   @app.post("/predict")
   def predict(data: InputData):
       prediction = model.predict(data.features)
       return {"prediction": prediction}
   ```

4. **Containerization**
   ```bash
   # Docker ile paketleme
   docker build -t my-ml-api:v1.0 .
   docker push my-registry/my-ml-api:v1.0
   ```

5. **Otomatik deployment**
   ```yaml
   # GitHub Actions ile CD
   - name: Deploy to production
     run: |
       docker pull my-registry/my-ml-api:v1.0
       docker run -d -p 80:8000 my-ml-api:v1.0
   ```

---

## 🛠️ Best Practices

### DVC
- ✅ Büyük dosyalar için DVC kullan (>100MB)
- ✅ Remote storage yapılandır (S3, GCS)
- ✅ `.dvcignore` ile gereksiz dosyaları hariç tut

### MLflow
- ✅ Anlamlı experiment isimleri kullan
- ✅ Tüm önemli parametreleri logla
- ✅ Model signature'ı ekle
- ✅ Input example sağla

### FastAPI
- ✅ Pydantic ile veri validation yap
- ✅ HTTP status code'larını doğru kullan
- ✅ Error handling ekle
- ✅ Rate limiting uygula

### Docker
- ✅ Multi-stage build kullan
- ✅ .dockerignore ekle
- ✅ Non-root user kullan
- ✅ Image boyutunu minimize et

### GitHub Actions
- ✅ Secrets ile hassas bilgileri sakla
- ✅ Matrix strategy ile multiple environment test et
- ✅ Cache mekanizmalarını kullan
- ✅ Conditional jobs oluştur

---

## 🔗 Projeleri Birleştirme

Tüm araçları bir arada kullanmak için:

1. **DVC** ile veri versiyonlama
2. **MLflow** ile model eğitimi ve kayıt
3. **FastAPI** ile model'i API olarak serve etme
4. **Docker** ile API'yi containerize etme
5. **GitHub Actions** ile tüm süreci otomatikleştirme

Bu rehberi takip ederek modern MLOps pipeline'ı oluşturabilirsiniz! 🚀
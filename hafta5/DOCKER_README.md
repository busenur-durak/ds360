# M5 Forecasting - Docker Deployment

Bu dokümantasyon M5 talep tahmin pipeline'ını Docker ile çalıştırmak için hazırlanmıştır.

## 📋 Gereksinimler

- Docker Desktop (macOS/Windows) veya Docker Engine (Linux)
- 4GB+ RAM
- 2GB+ disk alanı

## 🚀 Hızlı Başlangıç

### 1. Docker Image Build

```bash
docker build -t m5-forecast:dev .
```

### 2. Pipeline Çalıştırma

```bash
# Tek seferlik çalıştırma
docker run --rm \
    -v $(pwd)/artifacts:/app/artifacts \
    -v $(pwd)/data:/app/data \
    m5-forecast:dev

# Docker Compose ile
docker-compose up m5-forecast
```

### 3. Çıktıları Kontrol Et

```bash
ls -la ./artifacts/preds/
# run_YYYYMMDD.csv - Tahmin sonuçları
# run_YYYYMMDD_summary.png - Görsel rapor
```

## 📁 Dosya Yapısı

```
hafta5/
├── Dockerfile              # Container tanımı
├── docker-compose.yml      # Orchestration
├── requirements.txt        # Python dependencies
├── run.py                  # Single-run executor
├── prefect_demand_forecast.py  # Main pipeline
├── data/                   # M5 dataset (mount edilecek)
└── artifacts/              # Çıktılar (mount edilecek)
    ├── datasets/
    ├── models/
    ├── figures/
    ├── preds/              # 📊 Ana çıktılar burada
    └── reports/
```

## 🔧 Kullanım Senaryoları

### Eğitim Amaçlı (Tek Seferlik)
```bash
# 1. Build
docker build -t m5-forecast:dev .

# 2. Run
docker run --rm -v $(pwd)/artifacts:/app/artifacts m5-forecast:dev

# 3. Sonuçları gör
cat artifacts/preds/run_$(date +%Y%m%d).csv
```

### Production Amaçlı (Prefect Server)
```bash
# 1. Full stack başlat
docker-compose --profile production up -d

# 2. Prefect UI erişim
open http://localhost:4200

# 3. Pipeline deploy et
docker exec prefect-server prefect deployment build /app/prefect_demand_forecast.py:demand_forecast_flow
```

### Debug Modu
```bash
# Interactive container
docker run -it --rm \
    -v $(pwd)/artifacts:/app/artifacts \
    -v $(pwd)/data:/app/data \
    m5-forecast:dev bash

# Container içinde
python run.py
# veya
python prefect_demand_forecast.py
```

## ⚠️ Dikkat Edilecekler

### Veri Dosyaları
Container'a veri mount etmeyi unutmayın:
```bash
# ❌ Yanlış - veri mount yok
docker run --rm m5-forecast:dev

# ✅ Doğru - veri mount var
docker run --rm -v $(pwd)/data:/app/data m5-forecast:dev
```

### Çıktı Klasörü
Artifacts volume mount edilmezse çıktılar kaybolur:
```bash
# ❌ Yanlış - çıktılar container ile birlikte yok olur
docker run --rm m5-forecast:dev

# ✅ Doğru - çıktılar host'ta kalır
docker run --rm -v $(pwd)/artifacts:/app/artifacts m5-forecast:dev
```

### Memory Kullanımı
LightGBM ve Prophet memory-intensive:
```bash
# Memory limit koy
docker run --rm --memory=4g -v $(pwd)/artifacts:/app/artifacts m5-forecast:dev
```

## 🐛 Troubleshooting

### Container Başlamıyor
```bash
# Logları kontrol et
docker logs m5-forecast-pipeline

# Interactive debug
docker run -it --rm m5-forecast:dev bash
```

### Import Hataları
```bash
# Python path kontrol
docker run --rm m5-forecast:dev python -c "import sys; print(sys.path)"

# Dependencies kontrol
docker run --rm m5-forecast:dev pip list
```

### Veri Bulunamıyor
```bash
# Mount kontrol
docker run --rm -v $(pwd)/data:/app/data m5-forecast:dev ls -la /app/data

# File permissions kontrol
ls -la data/
```

## 📊 Beklenen Çıktılar

### Başarılı Çalışma:
```
🐳 M5 FORECASTING - DOCKER PIPELINE
📅 Başlangıç zamanı: 2025-01-15 09:00:00
📁 Veri yüklendi: (9565, 12)
🔮 35 tahmin üretildi
📁 CSV: ./artifacts/preds/run_20250115.csv
📊 PNG: ./artifacts/preds/run_20250115_summary.png
✅ Pipeline başarıyla tamamlandı!
```

### CSV Format:
```csv
date,item_id,store_id,prediction
2016-04-25,FOODS_3_090,CA_1,45.2
2016-04-26,FOODS_3_090,CA_1,42.8
...
```

## 🚀 Production Deployment

### Kubernetes (opsiyonel)
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: m5-forecast
spec:
  schedule: "0 9 * * *"  # Her gün 09:00
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: forecast
            image: m5-forecast:prod
            volumeMounts:
            - name: artifacts
              mountPath: /app/artifacts
```

### CI/CD Pipeline
```yaml
# .github/workflows/docker.yml
- name: Build and Test
  run: |
    docker build -t m5-forecast:test .
    docker run --rm m5-forecast:test python -c "import lightgbm, prophet; print('OK')"
```

## 📚 Kaynaklar

- [Docker Documentation](https://docs.docker.com/)
- [Prefect Docker Guide](https://docs.prefect.io/guides/docker/)
- [M5 Competition](https://www.kaggle.com/c/m5-forecasting-accuracy)
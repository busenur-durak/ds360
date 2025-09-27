# Python 3.10 slim image kullan
FROM python:3.10-slim

# Çalışma dizini
WORKDIR /app

# Sistem paketlerini güncelle
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python requirements'ı kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyala
COPY src/ ./src/
COPY data/ ./data/

# Model eğitimi için gerekli dosyaları kopyala
COPY dvc.yaml ./

# Model eğit
RUN python src/data_preparation.py && python src/train_model.py

# Port 8000'i expose et
EXPOSE 8000

# Non-root user oluştur
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check ekle
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Uygulamayı başlat
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
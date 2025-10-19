# M5 Forecasting Pipeline - Docker Container
FROM python:3.11-slim

# System dependencies için minimal gereksinimler
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini
WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kod dosyalarını kopyala (modüler yapı ile)
COPY *.py ./
COPY P*/ ./
COPY data/ ./data/

# Artifacts dizini oluştur (volume mount için)
RUN mkdir -p artifacts/datasets artifacts/models artifacts/figures artifacts/preds artifacts/reports

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Varsayılan komut
CMD ["python", "run.py"]
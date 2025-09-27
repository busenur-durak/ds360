# Python image kullan
FROM python:3.10-slim

# Çalışma dizini oluştur
WORKDIR /app

# Requirements dosyasını kopyala
COPY requirements.txt .

# Paketleri yükle
RUN pip install -r requirements.txt

# Uygulama dosyalarını kopyala
COPY app.py .

# Port 8000'i aç
EXPOSE 8000

# Uygulamayı başlat
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
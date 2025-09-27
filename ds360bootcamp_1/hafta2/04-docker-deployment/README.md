# Docker ile ML API Deployment

Bu proje Docker ile uygulamayı container'a almayı öğretir.

## Adımlar

1. **Docker image oluştur:**
   ```bash
   docker build -t ml-api .
   ```

2. **Container çalıştır:**
   ```bash
   docker run -p 8000:8000 ml-api
   ```

3. **Test et:**
   - http://localhost:8000
   - http://localhost:8000/health

## Docker Komutları

```bash
# Image'ları listele
docker images

# Container'ları listele
docker ps

# Container'ı durdur
docker stop <container_id>

# Container'ı sil
docker rm <container_id>

# Image'ı sil
docker rmi ml-api
```

## Dockerfile Açıklaması

- `FROM python:3.10-slim` - Base image
- `WORKDIR /app` - Çalışma dizini
- `COPY requirements.txt .` - Dosya kopyala
- `RUN pip install` - Paket yükle
- `EXPOSE 8000` - Port aç
- `CMD` - Başlangıç komutu

## Avantajları

- **Taşınabilirlik**: Her yerde aynı şekilde çalışır
- **İzolasyon**: Diğer uygulamalardan bağımsız
- **Kolay deployment**: Tek komutla çalışır
- **Versiyon kontrolü**: Image'lar versiyonlanabilir
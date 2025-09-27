# FastAPI ile Basit ML API

Bu proje FastAPI ile ML API oluşturmayı öğretir.

## Adımlar

1. **API'yi başlat:**
   ```bash
   uvicorn app:app --reload
   ```

2. **Tarayıcıda test et:**
   - http://localhost:8000 - Ana sayfa
   - http://localhost:8000/docs - Otomatik API dokümantasyonu
   - http://localhost:8000/health - Sağlık kontrolü

3. **Python ile test et:**
   ```bash
   python test_api.py
   ```

## API Endpoint'leri

- `GET /` - Ana sayfa
- `GET /health` - Sağlık kontrolü
- `GET /ornek` - Örnek istek/cevap
- `POST /tahmin` - Cinsiyet tahmini

## Örnek Kullanım

```bash
curl -X POST "http://localhost:8000/tahmin" \
  -H "Content-Type: application/json" \
  -d '{"boy": 175.0, "kilo": 70.0}'
```

## Öğrenilen Kavramlar

- **FastAPI**: Modern Python web framework
- **Pydantic**: Veri validasyonu
- **Uvicorn**: ASGI server
- **Automatic docs**: Otomatik API dokümantasyonu
- **REST API**: GET, POST endpoint'leri
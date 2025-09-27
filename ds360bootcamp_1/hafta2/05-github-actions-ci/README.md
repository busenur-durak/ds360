# GitHub Actions ile CI Pipeline

Bu proje GitHub Actions ile otomatik test sürecini öğretir.

## Adımlar

1. **Lokal testleri çalıştır:**
   ```bash
   python test_app.py
   python app.py
   ```

2. **GitHub'a push et:**
   ```bash
   git init
   git add .
   git commit -m "İlk commit"
   git branch -M main
   git remote add origin <repo-url>
   git push -u origin main
   ```

3. **GitHub'da sonuçları kontrol et:**
   - Repository → Actions sekmesi
   - Workflow çalışmalarını gör

## GitHub Actions Kavramları

### Workflow (.github/workflows/test.yml)
- **on**: Ne zaman çalışacak (push, PR, schedule)
- **jobs**: Paralel çalışan görevler
- **runs-on**: Hangi işletim sistemi
- **steps**: Sıralı adımlar

### Yaygın Actions
- `actions/checkout@v3` - Kodu indir
- `actions/setup-python@v3` - Python kur
- `actions/setup-node@v3` - Node.js kur

## CI/CD Avantajları

- **Otomatik test**: Her değişiklikte testler çalışır
- **Erken hata tespiti**: Problemler hemen görülür
- **Kalite kontrolü**: Kötü kod merge edilmez
- **Güven**: Deployment öncesi doğrulama

## Örnek Senaryolar

- ✅ Test geçerse → Merge edilebilir
- ❌ Test failse → Merge engellenebilir
- 🔄 Her commit'te otomatik kontrol
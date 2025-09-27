# DVC ile Veri Versiyonlama

Bu proje DVC'nin temel kullanımını öğretir.

## Adımlar

1. **Veri oluştur:**
   ```bash
   python create_data.py
   ```

2. **DVC ile takip et:**
   ```bash
   dvc add data.csv
   ```

3. **Git'e commit et:**
   ```bash
   git add data.csv.dvc .gitignore create_data.py README.md
   git commit -m "İlk veri versiyonu"
   ```

4. **Veriyi değiştir ve yeni versiyon oluştur:**
   ```bash
   # create_data.py dosyasını düzenle
   python create_data.py
   dvc add data.csv
   git add data.csv.dvc
   git commit -m "Veri güncellendi - v2"
   ```

## DVC Komutları

- `dvc add dosya.csv` - Dosyayı DVC ile takip et
- `dvc status` - Değişiklikleri kontrol et
- `dvc push` - Remote storage'a yükle
- `dvc pull` - Remote storage'dan indir
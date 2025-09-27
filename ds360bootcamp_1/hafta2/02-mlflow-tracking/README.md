# MLflow ile Deney Takibi

Bu proje MLflow'un temel kullanımını öğretir.

## Adımlar

1. **Modeli çalıştır:**
   ```bash
   python basit_model.py
   ```

2. **MLflow UI'ı başlat:**
   ```bash
   mlflow ui
   ```

3. **Tarayıcıda görüntüle:**
   - http://localhost:5000 adresine git
   - Experiment'leri ve run'ları gör

## Öğrenilen Kavramlar

- **Experiment**: Deney grubu
- **Run**: Tek bir model eğitimi
- **Parameter**: Model ayarları (örn: learning_rate)
- **Metric**: Sonuç metrikleri (örn: accuracy)
- **Artifact**: Kayıtlı dosyalar (model, grafik vb.)

## MLflow Komutları

- `mlflow.set_experiment()` - Deney adı belirle
- `mlflow.start_run()` - Yeni run başlat
- `mlflow.log_param()` - Parametre kaydet
- `mlflow.log_metric()` - Metrik kaydet
- `mlflow.log_model()` - Model kaydet
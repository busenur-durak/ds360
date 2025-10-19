# Time Series Cross-Validation Raporu

**Tarih:** 2025-10-12 02:01:27
**Method:** Rolling-Origin Cross-Validation

## Neden Shuffle CV Olmaz?

1. **Temporal Leakage**: Gelecek verisi ile geçmiş tahmin edilir
2. **Pattern Bozukluk**: Zaman bağımlı pattern'ler parçalanır
3. **Gerçekçi Olmama**: Production'da shuffle yok, sadece geçmiş var

## Sonuçlar

**Fold Sayısı:** 3
**Validation Horizon:** 28 gün

### Ortalama Metrikler

| Metrik | Ortalama | Std Sapma |
|--------|----------|-----------|
| MAE | 8.91 | 0.73 |
| RMSE | 13.44 | 1.12 |
| sMAPE | 33.83 | 5.73 |

### Fold Detayları

| Fold | MAE | RMSE | sMAPE (%) |
|------|-----|------|-----------|
| 0 | 9.49 | 12.80 | 33.48 |
| 1 | 7.89 | 12.51 | 41.02 |
| 2 | 9.37 | 15.01 | 26.99 |
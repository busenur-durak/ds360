# 📊 M5 Forecasting Dataset

## 📥 Dataset Download Required

Bu proje **Kaggle M5 Competition** dataset'ini kullanır. Dataset dosyaları (~430MB) GitHub'da depolanmaz.

### 🎓 **Ders Katılımcıları İçin:**
- **Ders sırasında instructor'dan** dataset dosyalarını alabilirsiniz
- **Ders notları** ile birlikte paylaşılmıştır

### 🔗 **Alternatif Download Links:**
- **Main Source**: [Kaggle M5 Competition](https://www.kaggle.com/c/m5-forecasting-accuracy/data)
- **Direct Files**: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

### 📁 **Required Files:**
Aşağıdaki dosyaları bu klasöre (`data/`) yerleştirin:

```
data/
├── sales_train_validation.csv    # 120MB - Ana satış verisi
├── calendar.csv                  # 103KB - Tarih ve event bilgileri  
├── sell_prices.csv               # 203MB - Fiyat verileri
├── sales_train_evaluation.csv    # 121MB - Evaluation verisi (opsiyonel)
└── sample_submission.csv         # 5MB - Submission format (opsiyonel)
```

### ⚡ **Quick Setup:**
```bash
# 1. Kaggle CLI ile (eğer setup'ınız varsa)
kaggle competitions download -c m5-forecasting-accuracy

# 2. Manuel download
# Kaggle'dan indirip bu klasöre kopyalayın

# 3. Verification
python P1_data_preparation/create_m5_subset.py
```

### 🚨 **Important Notes:**
- **sales_train_validation.csv** ve **calendar.csv** kesinlikle gerekli
- **sell_prices.csv** feature engineering için kullanılır  
- Toplam boyut: ~430MB
- Download için Kaggle account gerekli

### 🔄 **Alternative: Sample Data**
Eğer gerçek data'yı download edemezseniz:
```bash
python P1_data_preparation/create_sample_data.py
```

Bu komut synthetic sample data oluşturur ve pipeline'ı test edebilirsiniz.
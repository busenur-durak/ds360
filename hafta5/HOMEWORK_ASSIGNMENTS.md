# 📚 M5 Forecasting - Ödev Seçenekleri

Merhaba! M5 Forecasting dersi sonrası pratik yapabileceğiniz ödevleri hazırladım. **İstediğiniz ödevleri seçebilirsiniz** 😊

---

## 🎯 **ÇOK KOLAY ÖDEVLER** ⭐☆☆☆☆

### **Ödev 1: Pipeline'ı Çalıştır ve Gözlemle**
**⏰ Süre**: 1 saat  
**🎯 Amaç**: Sistemi tanımak

**Yapılacaklar:**
```bash
# Terminal'de sırayla çalıştır:
python run_modular.py --module P1
python run_modular.py --module P2  
python run_modular.py --module P4  # Prophet model
```

**Göreviniz:**
- Çıktıları ekran görüntüsü alın
- `artifacts/figures/` klasöründeki grafikleri açın
- Hangi ürün en çok satıyor?
- Prophet'in tahmini mantıklı gözüküyor mu?

**✅ Teslim**: 3-4 ekran görüntüsü + 2 cümle yorumunuz

---

### **Ödev 6: README Yazma**
**⏰ Süre**: 1 saat  
**🎯 Amaç**: Deneyimi paylaşmak

**Yapılacaklar:**
- Kendi deneyiminiz için README yazın
- "Bu projeyi nasıl çalıştırdım?"
- "Hangi adımları takip ettim?" 
- "Hangi sorunlarla karşılaştım?"
- "Ne öğrendim?"

**✅ Teslim**: Word/txt dosyası (1 sayfa yeter)

---

## 🎯 **KOLAY ÖDEVLER** ⭐⭐☆☆☆

### **Ödev 2: Model Karşılaştırması**
**⏰ Süre**: 1.5 saat  
**🎯 Amaç**: Farklı modelleri anlama

**Yapılacaklar:**
```bash
# İki farklı model çalıştır:
python run_modular.py --module P3  # ARIMA (geleneksel)
python run_modular.py --module P4  # Prophet (modern)
```

**Göreviniz:**
- `artifacts/preds/` klasöründeki sonuçları karşılaştırın
- Hangi model daha iyi tahmin yapıyor?
- Grafiklere bakarak hangi model daha gerçekçi?
- sMAPE değerlerini karşılaştırın (düşük olan daha iyi)

**✅ Teslim**: 1 sayfa karşılaştırma raporu

---

### **Ödev 3: Excel ile Veri Analizi**
**⏰ Süre**: 2 saat  
**🎯 Amaç**: Veriyi tanımak

**Yapılacaklar:**
1. P1'i çalıştırın: `python run_modular.py --module P1`
2. `artifacts/datasets/train.csv` dosyasını Excel'de açın
3. Basit analizler yapın:
   - Hangi ürün (item_id) en çok satılıyor?
   - Hangi günler satış daha yüksek?
   - Aralık ayında satışlar artıyor mu?

**Göreviniz:**
- Excel'de pivot table oluşturun
- Ürün bazında ortalama satış hesaplayın
- Aylık satış grafiği çizin

**✅ Teslim**: Excel dosyası + 1 sayfa bulgular

---

### **Ödev 5: Prophet Grafiklerini Yorumlama**
**⏰ Süre**: 1.5 saat  
**🎯 Amaç**: Time series pattern'lerini anlama

**Yapılacaklar:**
```bash
python run_modular.py --module P4  # Prophet'i çalıştır
```

**Göreviniz:**
- `artifacts/figures/prophet_components.png` grafiğini açın
- Trend, weekly, yearly pattern'leri inceleyin
- Sorulara cevap verin:
  - Bu ürün hangi günlerde daha çok satılıyor?
  - Yıl içinde hangi dönemler peak?
  - Trend yukarı mı aşağı mı?

**✅ Teslim**: Grafik üzerine arrow/text ile pattern'leri işaretleyin

---

## 🎯 **ORTA ÖDEVLER** ⭐⭐⭐☆☆

### **Ödev 4: Parametre Değiştirme**
**⏰ Süre**: 2 saat  
**🎯 Amaç**: Kod'da basit değişiklik yapma

**Yapılacaklar:**
1. `P1_data_preparation/create_m5_subset.py` dosyasını açın
2. Kod'da `CONFIG` bölümünü bulun
3. `n_items = 5` yerine `n_items = 3` yazın
4. Dosyayı kaydedin ve çalıştırın:

```bash
python run_modular.py --module P1
python run_modular.py --module P4
```

**Göreviniz:**
- 3 ürün vs 5 ürün sonuçlarını karşılaştırın
- Hangi durum daha hızlı çalışıyor?
- Tahmin kalitesi değişti mi?

**✅ Teslim**: Before/after ekran görüntüleri + yorumlar

---

## 🎯 **BONUS ÖDEVLER** (İsteğe Bağlı) 🏆

### **Ödev 7: Farklı Ürün Deneme**
**⏰ Süre**: 1 saat  
**🎯 Amaç**: Merak ettiğiniz şeyleri test etme

**Yapılacaklar:**
- En çok satan vs en az satan ürün için Prophet çalıştırın
- Tahminler nasıl değişiyor?
- Hangi ürün daha tahmin edilebilir?

**✅ Teslim**: Karşılaştırma ve gözlemleriniz

---

### **Ödev 8: Ekstra Model Çalıştırma**
**⏰ Süre**: 1.5 saat  
**🎯 Amaç**: Tüm pipeline'ı deneme

**Yapılacaklar:**
```bash
# Tüm modülleri sırayla çalıştır
python run_modular.py  # Full pipeline
```

**Göreviniz:**
- ARIMA, Prophet, LightGBM sonuçlarını karşılaştırın
- Hangi model en iyi performance veriyor?
- Cross-validation sonuçlarını inceleyin

**✅ Teslim**: 3 modelin karşılaştırma tablosu

---

## 🚀 **NASIL BAŞLAYACAĞIM?**

### **🔥 Hızlı Başlangıç**
1. **İlk kez yapıyorum**: Ödev 1 veya 6 ile başlayın
2. **Biraz deneyimim var**: Ödev 2 veya 3'ü deneyin
3. **Zorlansın biraz**: Ödev 4 veya 5'i seçin
4. **Meraklıyım**: Bonus ödevlere bakın

### **💡 İpuçları**
- Ödevleri birden fazla yapabilirsiniz
- Takıldığınız yerde sormaktan çekinmeyin
- Hata aldığınızda panic yapmayın - normal!
- Sonuçları anlamaya odaklanın, kod yazmaya değil

### **📧 Teslim**
- **Format**: Word, PDF, txt, Excel - ne rahatsa
- **Süre**: Kendi hızınızda, acele yok
- **Soru**: Takıldığınızda sorabilirsiniz

---

## 🎯 **NE ÖĞRENECEĞIM?**

Bu ödevleri yaparak şunları öğreneceksiniz:

✅ **Time series forecasting** ne demek  
✅ **ARIMA vs Prophet** farkları  
✅ **Feature engineering** neden önemli  
✅ **Model evaluation** nasıl yapılır  
✅ **Data analysis** basic skills  
✅ **Python pipeline** nasıl çalışır  

**En önemlisi**: Gerçek dünya verisinde **pattern recognition** yeteneğiniz gelişecek! 🧠

---

**🎉 Kolay gelsin! Hangi ödevleri seçerseniz seçin, öğrenmek için yapıyoruz. Mükemmel olması gerekmiyor! 😊**
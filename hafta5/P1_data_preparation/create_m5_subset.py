#!/usr/bin/env python3
"""
M5 Veri Seti Küçük Çalışma Seti Üretici

Bu script M5 veri setinden küçük bir alt-küme oluşturur:
- CA eyaleti, CA_1 mağazası, FOODS kategorisi
- En yüksek satışlı 5 ürün
- Günlük zaman serisi formatında
- Train/Validation split (son 28 gün validation)

Kullanım: python create_m5_subset.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def create_m5_subset():
    """M5 veri setinden küçük çalışma seti oluştur"""
    
    print("🎯 M5 Küçük Çalışma Seti Oluşturucu")
    print("=" * 50)
    
    # Çıktı klasörlerini oluştur
    os.makedirs('./artifacts/datasets', exist_ok=True)
    os.makedirs('./artifacts/figures', exist_ok=True)
    
    # 1. Veri dosyalarını oku
    print("\n📁 1. Veri dosyaları okunuyor...")
    
    try:
        # Sales verisi
        print("   • sales_train_validation.csv okunuyor...")
        sales_df = pd.read_csv('/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta5/data/sales_train_validation.csv')
        print(f"   ✓ Satış verisi: {sales_df.shape}")
        
        # Calendar verisi
        print("   • calendar.csv okunuyor...")
        calendar_df = pd.read_csv('/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta5/data/calendar.csv')
        calendar_df['date'] = pd.to_datetime(calendar_df['date'])
        print(f"   ✓ Takvim verisi: {calendar_df.shape}")
        
        # Prices verisi (opsiyonel, kullanmayacağız ama kontrol edelim)
        try:
            prices_df = pd.read_csv('/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta5/data/sell_prices.csv')
            print(f"   ✓ Fiyat verisi: {prices_df.shape}")
        except FileNotFoundError:
            print("   ⚠️  Fiyat verisi bulunamadı (isteğe bağlı)")
            
    except FileNotFoundError as e:
        print(f"   ❌ Veri dosyası bulunamadı: {e}")
        print("   💡 Önce create_sample_data.py çalıştırın veya gerçek M5 verisini indirin")
        return None, None, None
    
    # 2. CA_1 mağazası ve FOODS kategorisini filtrele
    print("\n🏪 2. CA_1 mağazası ve FOODS kategorisi filtreleniyor...")
    
    # CA_1 mağazası filtresi
    ca1_mask = (sales_df['store_id'] == 'CA_1')
    ca1_sales = sales_df[ca1_mask].copy()
    print(f"   • CA_1 mağazası ürün sayısı: {len(ca1_sales)}")
    
    # FOODS kategorisi filtresi
    # M5'te kategori 'cat_id' sütununda, FOODS genelde FOODS ile başlar
    foods_mask = ca1_sales['cat_id'].str.contains('FOOD', case=False, na=False)
    foods_sales = ca1_sales[foods_mask].copy()
    print(f"   • FOODS kategorisi ürün sayısı: {len(foods_sales)}")
    
    if len(foods_sales) == 0:
        print("   ⚠️  FOODS kategorisi bulunamadı, tüm kategorileri kullanıyoruz...")
        foods_sales = ca1_sales.copy()
    
    # 3. En yüksek satışlı 5 ürünü bul
    print("\n📊 3. En yüksek satışlı 5 ürün bulunuyor...")
    
    # Satış sütunlarını al (d_1, d_2, ... formatında)
    sales_cols = [col for col in foods_sales.columns if col.startswith('d_')]
    print(f"   • Toplam {len(sales_cols)} gün verisi mevcut")
    
    # Her ürün için toplam satışı hesapla
    foods_sales['total_sales'] = foods_sales[sales_cols].sum(axis=1)
    
    # En yüksek satışlı 5 ürünü seç
    top5_items = foods_sales.nlargest(5, 'total_sales')
    
    print("   • En yüksek satışlı 5 ürün:")
    for i, (_, item) in enumerate(top5_items.iterrows(), 1):
        print(f"     {i}. {item['item_id']} (Total: {item['total_sales']:,.0f})")
    
    # 4. Günlük zaman serisi oluştur (uzun format)
    print("\n📈 4. Günlük zaman serisi oluşturuluyor...")
    
    # Sadece top 5 ürünü al
    selected_items = top5_items[['id', 'item_id', 'store_id', 'state_id'] + sales_cols].copy()
    
    # Uzun formata çevir
    long_data = []
    
    for _, item_row in selected_items.iterrows():
        item_id = item_row['item_id']
        store_id = item_row['store_id']
        
        # Her gün için satış verisi
        for d_col in sales_cols:
            sales_value = item_row[d_col]
            
            # NaN değerleri 0 yap
            if pd.isna(sales_value):
                sales_value = 0
            
            long_data.append({
                'item_id': item_id,
                'store_id': store_id,
                'd': d_col,
                'sales': int(sales_value)
            })
    
    # DataFrame'e çevir
    long_df = pd.DataFrame(long_data)
    
    # Calendar ile birleştir (tarih bilgisi için)
    long_df = long_df.merge(calendar_df[['d', 'date']], on='d', how='left')
    
    # Tarih sıralaması
    long_df = long_df.sort_values(['item_id', 'date']).reset_index(drop=True)
    
    print(f"   ✓ Uzun format veri: {long_df.shape}")
    print(f"   • Tarih aralığı: {long_df['date'].min()} - {long_df['date'].max()}")
    print(f"   • Toplam gün sayısı: {long_df['date'].nunique()}")
    
    # 5. Eksik günleri 0 ile doldur
    print("\n🔧 5. Eksik günler kontrol ediliyor ve dolduruluyor...")
    
    # Her ürün için tam tarih aralığı oluştur
    all_dates = pd.date_range(start=long_df['date'].min(), 
                             end=long_df['date'].max(), 
                             freq='D')
    
    complete_data = []
    
    for item_id in long_df['item_id'].unique():
        item_data = long_df[long_df['item_id'] == item_id].copy()
        store_id = item_data['store_id'].iloc[0]
        
        # Eksik tarihleri bul
        existing_dates = set(item_data['date'])
        missing_dates = [d for d in all_dates if d not in existing_dates]
        
        if missing_dates:
            print(f"   • {item_id}: {len(missing_dates)} eksik gün dolduruldu")
            
            # Eksik günleri ekle
            for missing_date in missing_dates:
                complete_data.append({
                    'item_id': item_id,
                    'store_id': store_id,
                    'date': missing_date,
                    'sales': 0
                })
        
        # Mevcut verileri ekle
        for _, row in item_data.iterrows():
            complete_data.append({
                'item_id': row['item_id'],
                'store_id': row['store_id'],
                'date': row['date'],
                'sales': row['sales']
            })
    
    # Tam veri seti
    complete_df = pd.DataFrame(complete_data)
    complete_df = complete_df.sort_values(['item_id', 'date']).reset_index(drop=True)
    
    print(f"   ✓ Tam veri seti: {complete_df.shape}")
    
    # 6. Train/Validation split
    print("\n✂️ 6. Train/Validation bölünmesi yapılıyor...")
    
    # Tüm tarihleri al ve sırala
    all_dates_sorted = sorted(complete_df['date'].unique())
    
    # Son 28 günü validation, geri kalanını train yap
    validation_days = 28
    
    if len(all_dates_sorted) <= validation_days:
        print(f"   ⚠️  Yeterli veri yok. Toplam {len(all_dates_sorted)} gün, {validation_days} gün validation gerekli")
        validation_days = max(1, len(all_dates_sorted) // 4)  # %25'ini validation yap
        print(f"   • Validation gün sayısı {validation_days} olarak ayarlandı")
    
    # Tarih sınırları
    split_date = all_dates_sorted[-validation_days]
    train_end_date = all_dates_sorted[-validation_days-1] if len(all_dates_sorted) > validation_days else all_dates_sorted[0]
    
    # Train ve validation setleri
    train_df = complete_df[complete_df['date'] <= train_end_date].copy()
    valid_df = complete_df[complete_df['date'] >= split_date].copy()
    
    print(f"   • Train: {train_df['date'].min()} - {train_df['date'].max()} ({len(train_df)} satır)")
    print(f"   • Valid: {valid_df['date'].min()} - {valid_df['date'].max()} ({len(valid_df)} satır)")
    
    # Index'i tarih yap
    train_df = train_df.set_index('date')
    valid_df = valid_df.set_index('date')
    
    # 7. Çıktıları kaydet
    print("\n💾 7. Sonuçlar kaydediliyor...")
    
    # CSV dosyaları
    train_path = './artifacts/datasets/train.csv'
    valid_path = './artifacts/datasets/valid.csv'
    
    train_df.to_csv(train_path)
    valid_df.to_csv(valid_path)
    
    print(f"   ✓ Train verisi: {train_path}")
    print(f"   ✓ Valid verisi: {valid_path}")
    
    # 8. Görselleştirme
    print("\n📊 8. Günlük toplam satış grafiği oluşturuluyor...")
    
    # Günlük toplam satış hesapla
    daily_total = complete_df.groupby('date')['sales'].sum().reset_index()
    
    # Grafik oluştur
    plt.figure(figsize=(15, 8))
    
    # Train ve validation bölgelerini ayır
    train_dates = train_df.reset_index()['date'].unique()
    valid_dates = valid_df.reset_index()['date'].unique()
    
    train_total = daily_total[daily_total['date'].isin(train_dates)]
    valid_total = daily_total[daily_total['date'].isin(valid_dates)]
    
    # Train verisi
    plt.plot(train_total['date'], train_total['sales'], 
             label='Train', color='blue', linewidth=2)
    
    # Validation verisi
    plt.plot(valid_total['date'], valid_total['sales'], 
             label='Validation', color='red', linewidth=2)
    
    # Split çizgisi
    plt.axvline(x=split_date, color='gray', linestyle='--', alpha=0.7, 
                label=f'Train/Valid Split ({split_date.strftime("%Y-%m-%d")})')
    
    # Grafik düzenlemeleri
    plt.title('M5 Seçilen 5 Ürün - Günlük Toplam Satış\n' + 
              f'CA_1 Mağazası, FOODS Kategorisi', fontsize=16, fontweight='bold')
    plt.xlabel('Tarih', fontsize=12)
    plt.ylabel('Günlük Toplam Satış', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # X ekseni etiketlerini döndür
    plt.xticks(rotation=45)
    
    # Layout ayarla
    plt.tight_layout()
    
    # Kaydet
    figure_path = './artifacts/figures/overall_daily_sales.png'
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Grafik: {figure_path}")
    
    plt.close()
    
    # 9. Özet bilgiler
    print("\n📋 ÖZET BİLGİLER")
    print("=" * 50)
    print(f"• Seçilen ürünler: {', '.join(complete_df['item_id'].unique())}")
    print(f"• Toplam gün sayısı: {len(all_dates_sorted)}")
    print(f"• Train gün sayısı: {len(train_df.reset_index()['date'].unique())}")
    print(f"• Validation gün sayısı: {len(valid_df.reset_index()['date'].unique())}")
    print(f"• Ortalama günlük satış: {daily_total['sales'].mean():.1f}")
    print(f"• Maksimum günlük satış: {daily_total['sales'].max()}")
    print(f"• Minimum günlük satış: {daily_total['sales'].min()}")
    
    # Ürün bazında istatistikler
    print(f"\n📊 ÜRÜN BAZINDA İSTATİSTİKLER:")
    item_stats = complete_df.groupby('item_id')['sales'].agg(['sum', 'mean', 'std', 'max']).round(2)
    for item_id, stats in item_stats.iterrows():
        print(f"• {item_id}: Toplam={stats['sum']:,.0f}, Ort={stats['mean']:.1f}, "
              f"Std={stats['std']:.1f}, Max={stats['max']:.0f}")
    
    print(f"\n✅ İşlem tamamlandı!")
    print(f"📁 Çıktılar: ./artifacts/ klasöründe")
    
    return train_df, valid_df, daily_total

def main():
    """run_modular.py için wrapper fonksiyonu"""
    result = create_m5_subset()
    if result is None or (isinstance(result, tuple) and result[0] is None):
        print(f"❌ Veri dosyası bulunamadı. Sample data kullanın.")
        return False
    else:
        print(f"✅ M5 CA_1 FOODS subset created successfully!")
        return True

if __name__ == "__main__":
    try:
        result = create_m5_subset()
        if result is None or (isinstance(result, tuple) and result[0] is None):
            print(f"\n❌ Veri dosyası bulunamadı. Script durduruluyor.")
        else:
            train_data, valid_data, daily_sales = result
            print(f"\n🎉 M5 küçük çalışma seti başarıyla oluşturuldu!")
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()
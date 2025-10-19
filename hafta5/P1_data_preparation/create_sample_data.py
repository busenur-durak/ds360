#!/usr/bin/env python3
"""
M5 Forecasting için örnek veri oluşturucu

Gerçek M5 verisi yoksa, bu script ile test amaçlı örnek veri oluşturabilirsiniz.
Eğitim amacıyla kullanım için yeterli komplekslikte veri üretir.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_sample_m5_data():
    """M5 formatında örnek veri oluştur"""
    
    print("🎲 Örnek M5 verisi oluşturuluyor...")
    
    # Tarih aralığı (2 yıl)
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(date_range)
    
    print(f"📅 Tarih aralığı: {start_date.date()} - {end_date.date()} ({n_days} gün)")
    
    # 1. Calendar.csv oluştur
    print("📆 Calendar verisi oluşturuluyor...")
    
    calendar_data = []
    for i, date in enumerate(date_range):
        calendar_data.append({
            'd': f'd_{i+1}',
            'date': date.strftime('%Y-%m-%d'),
            'wm_yr_wk': date.isocalendar()[1],  # Yılın haftası
            'weekday': date.weekday() + 1,  # 1=Pazartesi, 7=Pazar
            'wday': date.weekday() + 1,
            'month': date.month,
            'year': date.year,
            'event_name_1': '',  # Boş bırak
            'event_type_1': '',
            'event_name_2': '',
            'event_type_2': '',
            'snap_CA': np.random.choice([0, 1], p=[0.9, 0.1]),  # Nadir olaylar
            'snap_TX': np.random.choice([0, 1], p=[0.9, 0.1]),
            'snap_WI': np.random.choice([0, 1], p=[0.9, 0.1])
        })
    
    calendar_df = pd.DataFrame(calendar_data)
    
    # 2. Sales train validation verisi oluştur
    print("🛒 Satış verisi oluşturuluyor...")
    
    # Ürün hierarşisi
    states = ['CA', 'TX', 'WI']
    stores_per_state = 4
    items_per_store = 10
    
    sales_data = []
    item_counter = 0
    
    for state in states:
        if state != 'CA':  # Sadece CA kullanacağız
            continue
            
        for store_num in range(1, 2):  # Sadece ilk mağaza
            store_id = f'{state}_{store_num}'
            
            for item_num in range(1, items_per_store + 1):
                item_counter += 1
                item_id = f'ITEM_{item_counter:03d}'
                
                # Ürün kategorisi (basit)
                dept_id = f'DEPT_{(item_counter % 3) + 1}'
                cat_id = f'CAT_{(item_counter % 2) + 1}'
                
                # ID oluştur
                id_str = f'{item_id}_{dept_id}_{cat_id}_{store_id}_validation'
                
                # Satış verileri oluştur (trend + mevsimsellik + gürültü)
                base_demand = np.random.uniform(10, 50)  # Temel talep
                trend = np.linspace(0, 5, n_days)  # Hafif artış trendi
                
                # Haftalık mevsimsellik (hafta sonu daha yüksek)
                weekly_pattern = []
                for d in date_range:
                    if d.weekday() >= 5:  # Cumartesi-Pazar
                        weekly_pattern.append(1.3)
                    else:
                        weekly_pattern.append(1.0)
                weekly_pattern = np.array(weekly_pattern)
                
                # Aylık mevsimsellik
                monthly_pattern = []
                for d in date_range:
                    if d.month in [11, 12]:  # Kasım-Aralık daha yüksek
                        monthly_pattern.append(1.5)
                    elif d.month in [6, 7, 8]:  # Yaz ayları
                        monthly_pattern.append(1.2)
                    else:
                        monthly_pattern.append(1.0)
                monthly_pattern = np.array(monthly_pattern)
                
                # Gürültü
                noise = np.random.normal(0, base_demand * 0.1, n_days)
                
                # Final satış değerleri
                sales_values = base_demand + trend + \
                              (base_demand * (weekly_pattern - 1)) + \
                              (base_demand * (monthly_pattern - 1)) + \
                              noise
                
                # Negatif değerleri 0 yap
                sales_values = np.maximum(sales_values, 0)
                
                # Bazı günlerde sıfır satış (daha gerçekçi)
                zero_mask = np.random.random(n_days) < 0.05  # %5 ihtimalle sıfır
                sales_values[zero_mask] = 0
                
                # Integer'a çevir
                sales_values = np.round(sales_values).astype(int)
                
                # Satır oluştur
                row = {
                    'id': id_str,
                    'item_id': item_id,
                    'dept_id': dept_id,
                    'cat_id': cat_id,
                    'store_id': store_id,
                    'state_id': state
                }
                
                # Satış değerlerini ekle
                for i, sales in enumerate(sales_values):
                    row[f'd_{i+1}'] = sales
                
                sales_data.append(row)
    
    sales_df = pd.DataFrame(sales_data)
    print(f"🎯 {len(sales_df)} ürün x {n_days} gün satış verisi oluşturuldu")
    
    # 3. Sell prices verisi oluştur (opsiyonel, basit)
    print("💰 Fiyat verisi oluşturuluyor...")
    
    prices_data = []
    for _, row in sales_df.iterrows():
        base_price = np.random.uniform(5, 50)  # 5-50 dolar arası
        
        # Her hafta için fiyat (basit)
        weeks = calendar_df['wm_yr_wk'].unique()
        for week in weeks[:20]:  # İlk 20 hafta
            price_variation = np.random.uniform(0.9, 1.1)  # ±10% değişim
            final_price = round(base_price * price_variation, 2)
            
            prices_data.append({
                'store_id': row['store_id'],
                'item_id': row['item_id'],
                'wm_yr_wk': week,
                'sell_price': final_price
            })
    
    prices_df = pd.DataFrame(prices_data)
    
    # 4. Dosyaları kaydet
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    calendar_path = os.path.join(data_dir, 'calendar.csv')
    sales_path = os.path.join(data_dir, 'sales_train_validation.csv')
    prices_path = os.path.join(data_dir, 'sell_prices.csv')
    
    calendar_df.to_csv(calendar_path, index=False)
    sales_df.to_csv(sales_path, index=False)
    prices_df.to_csv(prices_path, index=False)
    
    print(f"\n✅ Örnek veri oluşturuldu:")
    print(f"📄 {calendar_path} - {calendar_df.shape}")
    print(f"📄 {sales_path} - {sales_df.shape}")
    print(f"📄 {prices_path} - {prices_df.shape}")
    
    # 5. Veri özeti
    print(f"\n📊 VERİ ÖZETİ:")
    print(f"  • Toplam gün sayısı: {n_days}")
    print(f"  • Toplam ürün sayısı: {len(sales_df)}")
    print(f"  • Eyalet: CA")
    print(f"  • Mağaza: CA_1")
    print(f"  • Ortalama günlük satış: {sales_df[[col for col in sales_df.columns if col.startswith('d_')]].mean().mean():.1f}")
    print(f"  • Maksimum günlük satış: {sales_df[[col for col in sales_df.columns if col.startswith('d_')]].max().max()}")
    
    return calendar_df, sales_df, prices_df

if __name__ == "__main__":
    print("=" * 60)
    print("M5 FORECASTING - ÖRNEK VERİ OLUŞTURUCU")
    print("=" * 60)
    
    try:
        create_sample_m5_data()
        print("\n🎉 Örnek veri başarıyla oluşturuldu!")
        print("🚀 Artık m5_forecasting.py script'ini çalıştırabilirsiniz.")
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()
#!/usr/bin/env python3
"""
M5 Forecasting - Feature Engineering (Özellik Üretimi)

Bu script lag ve rolling özellikleri ile temel zaman serisi özelliklerini üretir.
Zaman serisi tahmininde geçmiş değerler (lag) ve hareketli ortalamalar (rolling) 
en önemli özelliklerdir.

Neden Lag ve Rolling Özellikler?
- Lag: Geçmiş satış değerleri gelecekteki satışları etkiler (trend, pattern)
- Rolling: Kısa dönem trendleri yakalamak için (gürültüyü azaltır)
- Tarih: Mevsimsellik ve döngüsel pattern'ler için kritik

Kullanım: python feature_engineering.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def create_features():
    """Lag ve rolling özellikleri ile feature engineering yap"""
    
    print("🔧 M5 Feature Engineering - Lag ve Rolling Özellikler")
    print("=" * 60)
    
    # Çıktı klasörlerini kontrol et
    if not os.path.exists('/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta5/artifacts/datasets'):
        print("❌ ./artifacts/datasets/ klasörü bulunamadı!")
        print("💡 Önce create_m5_subset.py çalıştırın")
        return None, None, None, None, None, None
    
    # 1. Veri yükleme
    print("\n📁 1. Train ve Validation verileri yükleniyor...")
    
    try:
        # Train verisi
        train_df = pd.read_csv('/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta5/artifacts/datasets/train.csv', parse_dates=['date'], index_col='date')
        print(f"   ✓ Train verisi: {train_df.shape}")
        
        # Validation verisi
        valid_df = pd.read_csv('/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta5/artifacts/datasets/valid.csv', parse_dates=['date'], index_col='date')
        print(f"   ✓ Valid verisi: {valid_df.shape}")
        
        # Veri tiplerini kontrol et
        print(f"   • Train tarih aralığı: {train_df.index.min()} - {train_df.index.max()}")
        print(f"   • Valid tarih aralığı: {valid_df.index.min()} - {valid_df.index.max()}")
        
    except FileNotFoundError as e:
        print(f"❌ Dosya bulunamadı: {e}")
        print("💡 Önce create_m5_subset.py çalıştırın")
        return None, None, None, None, None, None
    
    # Veriyi birleştir (feature engineering için tam zaman serisi gerekli)
    print(f"\n🔗 2. Train ve Valid birleştiriliyor (FE için tam seri gerekli)...")
    
    # Birleştir ve sırala
    all_df = pd.concat([train_df, valid_df]).sort_index()
    print(f"   ✓ Birleşik veri: {all_df.shape}")
    print(f"   • Toplam tarih aralığı: {all_df.index.min()} - {all_df.index.max()}")
    
    # Her ürün için ayrı feature engineering
    print(f"\n⚙️ 3. Feature Engineering başlıyor...")
    
    feature_data = []
    
    for item_id in all_df['item_id'].unique():
        print(f"   • İşleniyor: {item_id}")
        
        # Ürün verisini al
        item_df = all_df[all_df['item_id'] == item_id].copy()
        item_df = item_df.sort_index()  # Tarih sıralaması önemli
        
        # ===================
        # LAG ÖZELLİKLERİ
        # ===================
        # Neden lag? Geçmiş satış değerleri gelecekteki satışların en güçlü göstergesidir.
        # - lag_1: Dün ne sattık? (kısa dönem trend)
        # - lag_7: 1 hafta önce ne sattık? (haftalık pattern)
        # - lag_28: 4 hafta önce ne sattık? (aylık pattern)
        
        print(f"     → Lag özellikleri ekleniyor...")
        item_df['lag_1'] = item_df['sales'].shift(1)    # 1 gün önce
        item_df['lag_7'] = item_df['sales'].shift(7)    # 1 hafta önce
        item_df['lag_28'] = item_df['sales'].shift(28)  # 4 hafta önce
        
        # ===================
        # ROLLING ÖZELLİKLERİ
        # ===================
        # Neden rolling? Ham veriler gürültülü olabilir, hareketli ortalama trend'i yakalıyor.
        # - roll_mean_7: Son 1 haftanın ortalaması (kısa dönem trend)
        # - roll_mean_28: Son 4 haftanın ortalaması (orta dönem trend)
        # min_periods=1: İlk günlerde bile hesaplama yap
        
        print(f"     → Rolling özellikler ekleniyor...")
        item_df['roll_mean_7'] = item_df['sales'].rolling(window=7, min_periods=1).mean()
        item_df['roll_mean_28'] = item_df['sales'].rolling(window=28, min_periods=1).mean()
        
        # ===================
        # TARİH ÖZELLİKLERİ
        # ===================
        # Neden tarih özellikleri? Mevsimsellik ve döngüsel pattern'ler kritik.
        # - dow: Haftanın günü (0=Pazartesi, 6=Pazar) - hafta sonu/içi farkı
        # - dom: Ayın günü (1-31) - ay başı/sonu farkı  
        # - weekofyear: Yılın haftası (1-53) - yıllık trend
        # - month: Ay (1-12) - mevsimsel pattern
        
        print(f"     → Tarih özellikleri ekleniyor...")
        item_df['dow'] = item_df.index.dayofweek        # 0-6 (Pazartesi-Pazar)
        item_df['dom'] = item_df.index.day              # 1-31
        item_df['weekofyear'] = item_df.index.isocalendar().week  # 1-53
        item_df['month'] = item_df.index.month          # 1-12
        
        # Store ve item bilgilerini koru
        item_df['item_id'] = item_id
        item_df['store_id'] = item_df['store_id'].iloc[0]
        
        feature_data.append(item_df)
    
    # Tüm ürünleri birleştir
    feature_df = pd.concat(feature_data, ignore_index=False)
    feature_df = feature_df.sort_index()
    
    print(f"   ✓ Feature engineering tamamlandı: {feature_df.shape}")
    
    # 4. NaN değerleri handle et
    print(f"\n🔧 4. NaN değerleri kontrol ediliyor ve dolduruluyor...")
    
    # NaN istatistikleri
    nan_counts = feature_df.isnull().sum()
    nan_features = nan_counts[nan_counts > 0]
    
    if len(nan_features) > 0:
        print(f"   • NaN olan özellikler:")
        for feature, count in nan_features.items():
            print(f"     - {feature}: {count} NaN ({count/len(feature_df)*100:.1f}%)")
    else:
        print(f"   ✓ Hiç NaN değer yok!")
    
    # NaN doldurma stratejisi
    print(f"   • NaN doldurma uygulanıyor...")
    
    # Lag özellikleri: Başlangıçta NaN normal (geçmiş veri yok)
    # İlk değerleri 0 ile doldur (muhafazakar yaklaşım)
    lag_features = ['lag_1', 'lag_7', 'lag_28']
    for lag_col in lag_features:
        if lag_col in feature_df.columns:
            before_count = feature_df[lag_col].isnull().sum()
            feature_df[lag_col] = feature_df[lag_col].fillna(0)
            after_count = feature_df[lag_col].isnull().sum()
            print(f"     - {lag_col}: {before_count} → {after_count} NaN")
    
    # Rolling özellikler: min_periods=1 kullandık, NaN olmamalı
    roll_features = ['roll_mean_7', 'roll_mean_28']
    for roll_col in roll_features:
        if roll_col in feature_df.columns:
            before_count = feature_df[roll_col].isnull().sum()
            if before_count > 0:
                feature_df[roll_col] = feature_df[roll_col].fillna(method='ffill').fillna(0)
                after_count = feature_df[roll_col].isnull().sum()
                print(f"     - {roll_col}: {before_count} → {after_count} NaN")
    
    # Final kontrol
    final_nan = feature_df.isnull().sum().sum()
    print(f"   ✓ Final NaN sayısı: {final_nan}")
    
    # 5. Train/Valid'e tekrar böl
    print(f"\n✂️ 5. Train/Validation'a tekrar bölünüyor...")
    
    # Orijinal split tarihini bul
    train_end_date = train_df.index.max()
    valid_start_date = valid_df.index.min()
    
    print(f"   • Train son tarihi: {train_end_date}")
    print(f"   • Valid ilk tarihi: {valid_start_date}")
    
    # Feature'lı veriyi böl
    fe_train = feature_df[feature_df.index <= train_end_date].copy()
    fe_valid = feature_df[feature_df.index >= valid_start_date].copy()
    
    print(f"   ✓ FE Train: {fe_train.shape}")
    print(f"   ✓ FE Valid: {fe_valid.shape}")
    
    # 6. X, y'ye ayır
    print(f"\n🎯 6. Özellik matrisi (X) ve hedef (y) ayrılıyor...")
    
    # Hedef değişken
    target_col = 'sales'
    
    # Özellik sütunları (sales, item_id, store_id hariç)
    feature_cols = [col for col in fe_train.columns 
                   if col not in [target_col, 'item_id', 'store_id']]
    
    print(f"   • Özellik sayısı: {len(feature_cols)}")
    print(f"   • Özellikler: {feature_cols}")
    
    # Train set
    X_train = fe_train[feature_cols].copy()
    y_train = fe_train[target_col].copy()
    
    # Valid set
    X_valid = fe_valid[feature_cols].copy()
    y_valid = fe_valid[target_col].copy()
    
    print(f"   ✓ X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"   ✓ X_valid: {X_valid.shape}, y_valid: {y_valid.shape}")
    
    # Metadata'yı ayrı tut (isteğe bağlı)
    train_meta = fe_train[['item_id', 'store_id']].copy()
    valid_meta = fe_valid[['item_id', 'store_id']].copy()
    
    # 7. Parquet olarak kaydet
    print(f"\n💾 7. Feature engineered veriler kaydediliyor...")
    
    # Tam feature dataset'leri kaydet (meta bilgilerle)
    fe_train_path = '/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta5/artifacts/datasets/fe_train.parquet'
    fe_valid_path = '/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta5/artifacts/datasets/fe_valid.parquet'
    
    fe_train.to_parquet(fe_train_path)
    fe_valid.to_parquet(fe_valid_path)
    
    print(f"   ✓ FE Train: {fe_train_path}")
    print(f"   ✓ FE Valid: {fe_valid_path}")
    
    # X, y matrislerini de kaydet (model için direkt kullanım)
    X_train.to_parquet('/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta5/artifacts/datasets/X_train.parquet')
    y_train.to_frame('sales').to_parquet('/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta5/artifacts/datasets/y_train.parquet')
    X_valid.to_parquet('/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta5/artifacts/datasets/X_valid.parquet')
    y_valid.to_frame('sales').to_parquet('/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta5/artifacts/datasets/y_valid.parquet')
    
    print(f"   ✓ X_train, y_train, X_valid, y_valid kaydedildi")
    
    # 8. Özellik analizi
    print(f"\n📊 8. Özellik analizi yapılıyor...")
    
    # Describe istatistikleri
    print(f"\n📈 ÖZELLİK İSTATİSTİKLERİ:")
    print("=" * 50)
    
    feature_stats = X_train.describe()
    print(feature_stats.round(2))
    
    # Korelasyon analizi
    print(f"\n🔗 HEDEF İLE KORELASYON:")
    print("=" * 30)
    
    # Training setinde hedef ile korelasyon
    corr_with_target = X_train.corrwith(y_train).sort_values(ascending=False)
    
    print("En yüksek korelasyonlu özellikler:")
    for feature, corr in corr_with_target.items():
        print(f"  {feature:15}: {corr:6.3f}")
    
    # 9. Görselleştirme
    print(f"\n📊 9. Özellik dağılımları görselleştiriliyor...")
    
    # Özellik histogramları
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_features == 1 else axes
    
    for i, feature in enumerate(feature_cols):
        if i < len(axes):
            ax = axes[i]
            
            # Histogram
            X_train[feature].hist(bins=30, alpha=0.7, ax=ax)
            ax.set_title(f'{feature}\nMean: {X_train[feature].mean():.2f}')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
    
    # Boş subplot'ları gizle
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Feature Distributions (Training Set)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Kaydet
    hist_path = '/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta5/artifacts/figures/feature_distributions.png'
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Feature histogramları: {hist_path}")
    plt.close()
    
    # Korelasyon ısı haritası
    plt.figure(figsize=(10, 8))
    
    # Tüm özelliklerin birbirleriyle korelasyonu
    correlation_matrix = X_train.corr()
    
    # Heatmap
    mask = np.triu(correlation_matrix)  # Üst üçgeni gizle
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={'label': 'Correlation'})
    
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Kaydet
    corr_path = '/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta5/artifacts/figures/feature_correlations.png'
    plt.savefig(corr_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Korelasyon matrisi: {corr_path}")
    plt.close()
    
    # 10. Özet rapor
    print(f"\n📋 FEATURE ENGINEERİNG ÖZETİ")
    print("=" * 50)
    print(f"• Toplam özellik sayısı: {len(feature_cols)}")
    print(f"• Lag özellikleri: {len([f for f in feature_cols if 'lag' in f])}")
    print(f"• Rolling özellikleri: {len([f for f in feature_cols if 'roll' in f])}")
    print(f"• Tarih özellikleri: {len([f for f in feature_cols if f in ['dow', 'dom', 'weekofyear', 'month']])}")
    print(f"• Train örnekleri: {len(X_train):,}")
    print(f"• Valid örnekleri: {len(X_valid):,}")
    print(f"• Hedef ortalaması (train): {y_train.mean():.2f}")
    print(f"• Hedef std (train): {y_train.std():.2f}")
    
    print(f"\n🎯 EN ÖNEMLİ ÖZELLİKLER (korelasyon bazında):")
    top_features = corr_with_target.abs().nlargest(5)
    for i, (feature, corr) in enumerate(top_features.items(), 1):
        print(f"  {i}. {feature}: {corr:.3f}")
    
    print(f"\n✅ Feature Engineering tamamlandı!")
    print(f"📁 Çıktılar: ./artifacts/datasets/ ve ./artifacts/figures/")
    
    return fe_train, fe_valid, X_train, y_train, X_valid, y_valid

def main():
    """run_modular.py için wrapper fonksiyonu"""
    result = create_features()
    if result is None or (isinstance(result, tuple) and result[0] is None):
        print(f"❌ Feature engineering başarısız. P1'i önce çalıştırın.")
        return False
    else:
        print(f"✅ Feature Engineering tamamlandı!")
        return True

if __name__ == "__main__":
    try:
        print("🚀 M5 Feature Engineering başlatılıyor...")
        
        result = create_features()
        if result is None or (isinstance(result, tuple) and result[0] is None):
            print(f"\n❌ Feature engineering başarısız.")
        else:
            fe_train, fe_valid, X_train, y_train, X_valid, y_valid = result
            print(f"\n🎉 İşlem başarıyla tamamlandı!")
            print(f"📊 Artık makine öğrenmesi modellerini eğitebilirsiniz.")
        
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()
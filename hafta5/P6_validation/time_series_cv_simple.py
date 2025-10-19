#!/usr/bin/env python3
"""
Time Series Cross-Validation (Rolling-Origin) for LightGBM - Simplified Version

NEDEN SHUFFLE CV OLMAZ?
1. Temporal Leakage: Gelecek verisi ile geçmiş tahmin edilir (data leakage)
2. Pattern Bozukluk: Zaman bağımlı pattern'ler parçalanır
3. Gerçekçi Olmama: Production'da shuffle yok, sadece geçmiş var
"""

import pandas as pd
import numpy as np
import warnings
import os
import json
from datetime import datetime, timedelta

# LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("❌ LightGBM kütüphanesi bulunamadı. 'pip install lightgbm' ile kurun.")
    LIGHTGBM_AVAILABLE = False

# Preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

def load_data(artifacts_path='/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta5/artifacts'):
    """Feature engineered verileri yükle"""
    print("📁 Veri yükleniyor...")
    
    train_df = pd.read_parquet(f'{artifacts_path}/datasets/fe_train.parquet')
    valid_df = pd.read_parquet(f'{artifacts_path}/datasets/fe_valid.parquet')
    full_df = pd.concat([train_df, valid_df]).sort_index()
    
    print(f"   ✓ Full data: {full_df.shape}")
    print(f"   • Tarih: {full_df.index.min()} - {full_df.index.max()}")
    
    return full_df

def create_cv_splits(full_df, validation_horizon=28, n_splits=3):
    """Rolling-origin CV splits oluştur"""
    print(f"📊 {n_splits} katlı time series splits oluşturuluyor...")
    
    all_dates = sorted(full_df.index.unique())
    total_days = len(all_dates)
    
    splits = []
    for fold in range(n_splits):
        val_end_offset = fold * validation_horizon
        val_start_offset = (fold + 1) * validation_horizon
        
        if val_start_offset >= total_days:
            continue
            
        val_end_idx = total_days - 1 - val_end_offset
        val_start_idx = total_days - val_start_offset
        train_end_idx = val_start_idx - 1
        
        if train_end_idx < validation_horizon:
            continue
            
        splits.append({
            'fold': fold,
            'train_dates': all_dates[:train_end_idx + 1],
            'val_dates': all_dates[val_start_idx:val_end_idx + 1]
        })
        
        print(f"   Fold {fold}: Train {len(splits[-1]['train_dates'])}d, Valid {len(splits[-1]['val_dates'])}d")
    
    return splits

def prepare_fold_data(full_df, split_info):
    """Bir fold için veri hazırla"""
    train_dates = split_info['train_dates']
    val_dates = split_info['val_dates']
    
    train_df = full_df[full_df.index.isin(train_dates)].copy()
    val_df = full_df[full_df.index.isin(val_dates)].copy()
    
    # Kategorik encoding
    for col in ['item_id', 'store_id']:
        le = LabelEncoder()
        train_df[f'{col}_encoded'] = le.fit_transform(train_df[col])
        
        val_encoded = []
        for val in val_df[col]:
            if val in le.classes_:
                val_encoded.append(le.transform([val])[0])
            else:
                val_encoded.append(0)
        val_df[f'{col}_encoded'] = val_encoded
    
    # Features
    feature_cols = [col for col in train_df.columns if col in [
        'lag_1', 'lag_7', 'lag_28', 'roll_mean_7', 'roll_mean_28',
        'dow', 'dom', 'weekofyear', 'month', 'item_id_encoded', 'store_id_encoded'
    ]]
    
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['sales']
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df['sales']
    
    return X_train, y_train, X_val, y_val

def train_lightgbm(X_train, y_train, X_val, y_val):
    """LightGBM eğit"""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1,
        'random_state': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    return model

def calculate_metrics(y_true, y_pred):
    """Metrikleri hesapla"""
    y_pred = np.maximum(y_pred, 0)
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # sMAPE
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100 if mask.sum() > 0 else float('inf')
    
    return {'MAE': mae, 'RMSE': rmse, 'sMAPE': smape}

def run_cv_fold(full_df, split_info):
    """Tek fold çalıştır"""
    fold = split_info['fold']
    print(f"   🔄 Fold {fold} çalıştırılıyor...")
    
    # Veri hazırlama
    X_train, y_train, X_val, y_val = prepare_fold_data(full_df, split_info)
    
    # Model eğitimi
    model = train_lightgbm(X_train, y_train, X_val, y_val)
    
    # Tahmin ve metrikler
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    metrics = calculate_metrics(y_val.values, y_pred)
    
    print(f"     • MAE: {metrics['MAE']:.2f}, sMAPE: {metrics['sMAPE']:.2f}%")
    
    return {
        'fold': fold,
        'metrics': metrics,
        'best_iteration': model.best_iteration
    }

def run_time_series_cv():
    """Time Series CV ana fonksiyon"""
    print("📅 Time Series Cross-Validation (Rolling-Origin)")
    print("⚠️  NEDEN SHUFFLE CV OLMAZ?")
    print("   1. Temporal Leakage: Gelecek verisi ile geçmiş tahmin edilir")
    print("   2. Pattern Bozukluk: Zaman bağımlı pattern'ler parçalanır")
    print("   3. Gerçekçi Olmama: Production'da shuffle yok, sadece geçmiş var")
    print("=" * 70)
    
    # 1. Veri yükleme
    full_df = load_data()
    
    # 2. Splits oluştur
    splits = create_cv_splits(full_df, validation_horizon=28, n_splits=3)
    
    # 3. CV çalıştır
    print(f"\n🔄 {len(splits)} katlı CV çalıştırılıyor...")
    cv_results = []
    
    for split_info in splits:
        try:
            result = run_cv_fold(full_df, split_info)
            cv_results.append(result)
        except Exception as e:
            print(f"   ❌ Fold {split_info['fold']} hatası: {e}")
    
    # 4. Sonuçları topla
    print(f"\n📊 CV Sonuçları ({len(cv_results)} fold):")
    
    mae_values = [r['metrics']['MAE'] for r in cv_results]
    rmse_values = [r['metrics']['RMSE'] for r in cv_results]
    smape_values = [r['metrics']['sMAPE'] for r in cv_results]
    
    summary = {
        'MAE': {'mean': np.mean(mae_values), 'std': np.std(mae_values)},
        'RMSE': {'mean': np.mean(rmse_values), 'std': np.std(rmse_values)},
        'sMAPE': {'mean': np.mean(smape_values), 'std': np.std(smape_values)}
    }
    
    print(f"   📈 Ortalama Metrikler:")
    for metric, stats in summary.items():
        print(f"   • {metric}: {stats['mean']:.2f} ±{stats['std']:.2f}")
    
    # 5. Raporları kaydet
    os.makedirs('./artifacts/reports', exist_ok=True)
    
    # JSON raporu
    report = {
        'method': 'Time Series CV (Rolling-Origin)',
        'validation_horizon': 28,
        'n_folds': len(cv_results),
        'timestamp': datetime.now().isoformat(),
        'why_not_shuffle': [
            'Temporal Leakage: Gelecek verisi ile geçmiş tahmin edilir',
            'Pattern Bozukluk: Zaman bağımlı pattern\'ler parçalanır',
            'Gerçekçi Olmama: Production\'da shuffle yok, sadece geçmiş var'
        ],
        'summary_metrics': summary,
        'fold_results': cv_results
    }
    
    with open('./artifacts/reports/tscv_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Markdown raporu
    md_lines = [
        "# Time Series Cross-Validation Raporu",
        "",
        f"**Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "**Method:** Rolling-Origin Cross-Validation",
        "",
        "## Neden Shuffle CV Olmaz?",
        "",
        "1. **Temporal Leakage**: Gelecek verisi ile geçmiş tahmin edilir",
        "2. **Pattern Bozukluk**: Zaman bağımlı pattern'ler parçalanır",
        "3. **Gerçekçi Olmama**: Production'da shuffle yok, sadece geçmiş var",
        "",
        "## Sonuçlar",
        "",
        f"**Fold Sayısı:** {len(cv_results)}",
        f"**Validation Horizon:** 28 gün",
        "",
        "### Ortalama Metrikler",
        "",
        "| Metrik | Ortalama | Std Sapma |",
        "|--------|----------|-----------|"
    ]
    
    for metric, stats in summary.items():
        md_lines.append(f"| {metric} | {stats['mean']:.2f} | {stats['std']:.2f} |")
    
    md_lines.extend([
        "",
        "### Fold Detayları",
        "",
        "| Fold | MAE | RMSE | sMAPE (%) |",
        "|------|-----|------|-----------|"
    ])
    
    for result in cv_results:
        m = result['metrics']
        md_lines.append(f"| {result['fold']} | {m['MAE']:.2f} | {m['RMSE']:.2f} | {m['sMAPE']:.2f} |")
    
    with open('./artifacts/reports/tscv_report.md', 'w') as f:
        f.write('\n'.join(md_lines))
    
    print(f"\n✅ Raporlar kaydedildi:")
    print(f"   📄 ./artifacts/reports/tscv_report.json")
    print(f"   📄 ./artifacts/reports/tscv_report.md")
    
    return cv_results, summary

if __name__ == "__main__":
    if not LIGHTGBM_AVAILABLE:
        print("❌ LightGBM gerekli")
        exit(1)
    
    try:
        cv_results, summary = run_time_series_cv()
        print(f"\n🎉 Time Series CV tamamlandı!")
        print(f"📊 Ortalama sMAPE: {summary['sMAPE']['mean']:.2f}% ±{summary['sMAPE']['std']:.2f}%")
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
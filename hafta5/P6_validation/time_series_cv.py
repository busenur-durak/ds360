#!/usr/bin/env python3
"""
Time Series Cross-Validation (Rolling-Origin) for LightGBM

Bu script zaman serisi için uygun çapraz doğrulama yapar:
- Rolling-origin (expanding window) yaklaşımı
- 3 katlı test ile robust performans ölçümü
- LightGBM pipeline'ı fonksiyonlaştırır

NEDEN SHUFFLE CV OLMAZ?
1. Temporal Leakage: Gelecek verisi ile geçmiş tahmin edilir (data leakage)
2. Pattern Bozukluk: Zaman bağımlı pattern'ler parçalanır
3. Gerçekçi Olmama: Production'da shuffle yok, sadece geçmiş var

Rolling-Origin CV:
- Train window: Sabit veya genişleyen
- Valid window: 28 gün (business horizon)
- 3 kat: 28, 56, 84 gün önceki validation

Kullanım: python time_series_cv.py
"""

import pandas as pd
import numpy as np
import warnings
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

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

class TimeSeriesCV:
    """
    Time Series Cross-Validation sınıfı
    
    Rolling-origin validation ile zaman serisi için uygun CV yapar.
    Temporal leakage'ı önler ve gerçekçi performans ölçümü sağlar.
    """
    
    def __init__(self, artifacts_path='./artifacts', validation_horizon=28, n_splits=3):
        self.artifacts_path = artifacts_path
        self.validation_horizon = validation_horizon
        self.n_splits = n_splits
        self.cv_results = []
        self.summary_metrics = {}
        
        # Reports klasörü oluştur
        os.makedirs(f'{artifacts_path}/reports', exist_ok=True)
        
        print("📅 Time Series Cross-Validation (Rolling-Origin)")
        print("⚠️  NEDEN SHUFFLE CV OLMAZ?")
        print("   1. Temporal Leakage: Gelecek verisi ile geçmiş tahmin edilir")
        print("   2. Pattern Bozukluk: Zaman bağımlı pattern'ler parçalanır") 
        print("   3. Gerçekçi Olmama: Production'da shuffle yok, sadece geçmiş var")
        print("=" * 70)
    
    def load_data(self):
        """Feature engineered verileri yükle"""
        
        print("\n📁 1. Veri yükleniyor...")
        
        try:
            train_path = f'{self.artifacts_path}/datasets/fe_train.parquet'
            valid_path = f'{self.artifacts_path}/datasets/fe_valid.parquet'
            
            self.train_df = pd.read_parquet(train_path)
            self.valid_df = pd.read_parquet(valid_path)
            
            # Birleştir (tam zaman serisi için)
            self.full_df = pd.concat([self.train_df, self.valid_df]).sort_index()
            
            print(f"   ✓ Train: {self.train_df.shape}")
            print(f"   ✓ Valid: {self.valid_df.shape}")
            print(f"   ✓ Full: {self.full_df.shape}")
            print(f"   • Tarih aralığı: {self.full_df.index.min()} - {self.full_df.index.max()}")
            print(f"   • Ürün sayısı: {self.full_df['item_id'].nunique()}")
            
        except FileNotFoundError as e:
            print(f"   ❌ Dosya bulunamadı: {e}")
            raise
    
    def create_time_series_splits(self):
        """Rolling-origin CV split'leri oluştur"""
        
        print(f"\n📊 2. {self.n_splits} katlı time series splits oluşturuluyor...")
        
        # Tüm tarihleri al
        all_dates = sorted(self.full_df.index.unique())
        total_days = len(all_dates)
        
        print(f"   • Toplam gün: {total_days}")
        print(f"   • Validation horizon: {self.validation_horizon} gün")
        
        splits = []
        
        for fold in range(self.n_splits):
            # Validation son tarihini belirle
            # Fold 0: En son 28 gün
            # Fold 1: Son 56-28 günler arası 
            # Fold 2: Son 84-56 günler arası
            
            val_end_offset = fold * self.validation_horizon
            val_start_offset = (fold + 1) * self.validation_horizon
            
            if val_start_offset >= total_days:
                print(f"   ⚠️  Fold {fold}: Yeterli veri yok, atlanıyor")
                continue
            
            # Tarih indeksleri
            val_end_idx = total_days - 1 - val_end_offset
            val_start_idx = total_days - val_start_offset
            
            # Train: Başlangıçtan validation başına kadar
            train_end_idx = val_start_idx - 1
            
            if train_end_idx < self.validation_horizon:  # Minimum train size
                print(f"   ⚠️  Fold {fold}: Train çok küçük, atlanıyor")
                continue
            
            # Tarih aralıkları
            train_dates = all_dates[:train_end_idx + 1]
            val_dates = all_dates[val_start_idx:val_end_idx + 1]
            
            split_info = {
                'fold': fold,
                'train_start': train_dates[0],
                'train_end': train_dates[-1],
                'val_start': val_dates[0],
                'val_end': val_dates[-1],
                'train_days': len(train_dates),
                'val_days': len(val_dates)
            }
            
            splits.append(split_info)
            
            print(f"   📋 Fold {fold}:")
            print(f"     • Train: {split_info['train_start']} - {split_info['train_end']} ({split_info['train_days']} gün)")
            print(f"     • Valid: {split_info['val_start']} - {split_info['val_end']} ({split_info['val_days']} gün)")
        
        self.splits = splits
        print(f"   ✓ {len(splits)} geçerli split oluşturuldu")
        
        return splits
    
    def prepare_fold_data(self, split_info):
        """Bir fold için veri hazırla"""
        
        # Train ve validation verilerini ayır
        train_mask = (self.full_df.index >= split_info['train_start']) & \
                     (self.full_df.index <= split_info['train_end'])
        val_mask = (self.full_df.index >= split_info['val_start']) & \
                   (self.full_df.index <= split_info['val_end'])
        
        fold_train = self.full_df[train_mask].copy()
        fold_val = self.full_df[val_mask].copy()
        
        return fold_train, fold_val
    
    def encode_categorical_features(self, train_df, val_df):
        """Kategorik özellikleri encode et"""
        
        categorical_cols = ['item_id', 'store_id']
        label_encoders = {}
        
        for col in categorical_cols:
            if col in train_df.columns:
                le = LabelEncoder()
                
                # Train ile fit
                train_df[f'{col}_encoded'] = le.fit_transform(train_df[col])
                
                # Valid transform (unseen handling)
                val_encoded = []
                for val in val_df[col]:
                    if val in le.classes_:
                        val_encoded.append(le.transform([val])[0])
                    else:
                        val_encoded.append(0)  # Unseen = 0
                val_df[f'{col}_encoded'] = val_encoded
                
                label_encoders[col] = le
        
        return train_df, val_df, label_encoders
    
    def prepare_features_target(self, train_df, val_df):
        """Özellik ve hedef değişkenleri hazırla"""
        
        # Özellik sütunları
        lag_cols = [col for col in train_df.columns if 'lag_' in col]
        roll_cols = [col for col in train_df.columns if 'roll_' in col]
        date_cols = ['dow', 'dom', 'weekofyear', 'month']
        encoded_cols = [col for col in train_df.columns if '_encoded' in col]
        
        feature_cols = lag_cols + roll_cols + date_cols + encoded_cols
        target_col = 'sales'
        
        # Train
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[target_col]
        
        # Valid
        X_val = val_df[feature_cols].fillna(0)
        y_val = val_df[target_col]
        
        return X_train, y_train, X_val, y_val, feature_cols
    
    def train_lightgbm_fold(self, X_train, y_train, X_val, y_val):
        """Bir fold için LightGBM eğit"""
        
        # LightGBM parametreleri
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        # Dataset'ler
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Model eğitimi
        model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            num_boost_round=500,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)  # Silent
            ]
        )
        
        return model
    
    def calculate_metrics(self, y_true, y_pred):
        """Metrikleri hesapla"""
        
        # Negatif değerleri 0 yap
        y_pred = np.maximum(y_pred, 0)
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # MAPE
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = float('inf')
        
        # sMAPE
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if mask.sum() > 0:
            smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
        else:
            smape = float('inf')
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'sMAPE': smape
        }
    
    def run_single_fold(self, split_info):
        """Tek fold çalıştır"""
        
        fold = split_info['fold']
        print(f"   🔄 Fold {fold} çalıştırılıyor...")
        
        # 1. Veri hazırlama
        fold_train, fold_val = self.prepare_fold_data(split_info)
        
        # 2. Encoding
        fold_train, fold_val, label_encoders = self.encode_categorical_features(fold_train, fold_val)
        
        # 3. Feature/target hazırlama
        X_train, y_train, X_val, y_val, feature_cols = self.prepare_features_target(fold_train, fold_val)
        
        print(f"     • Train: {X_train.shape}, Valid: {X_val.shape}")
        
        # 4. Model eğitimi
        model = self.train_lightgbm_fold(X_train, y_train, X_val, y_val)
        
        # 5. Tahmin ve metrikler
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        metrics = self.calculate_metrics(y_val.values, y_pred)
        
        print(f"     • MAE: {metrics['MAE']:.2f}, sMAPE: {metrics['sMAPE']:.2f}%")
        
        # Sonuçları kaydet
        fold_result = {
            'fold': fold,
            'split_info': split_info,
            'metrics': metrics,
            'model_info': {
                'best_iteration': model.best_iteration,
                'train_score': model.best_score['train']['rmse'],
                'valid_score': model.best_score['valid']['rmse']
            },
            'data_info': {
                'train_shape': list(X_train.shape),
                'valid_shape': list(X_val.shape),
                'n_items_train': fold_train['item_id'].nunique(),
                'n_items_valid': fold_val['item_id'].nunique()
            }
        }
        
        return fold_result
    
    def run_cross_validation(self):
        """Tam cross-validation çalıştır"""
        
        print(f"\n🔄 3. {self.n_splits} katlı cross-validation çalıştırılıyor...")
        
        self.cv_results = []
        
        for split_info in self.splits:
            try:
                fold_result = self.run_single_fold(split_info)
                self.cv_results.append(fold_result)
            except Exception as e:
                print(f"   ❌ Fold {split_info['fold']} hatası: {e}")
                continue
        
        print(f"   ✓ {len(self.cv_results)} fold başarıyla tamamlandı")
    
    def aggregate_metrics(self):
        """Metrikleri topla ve ortalama al"""
        
        print(f"\n📊 4. Metrikler toplanıyor ve ortalama hesaplanıyor...")
        
        if not self.cv_results:
            print("   ❌ Hiç fold sonucu yok")
            return
        
        # Metrik listelerini topla
        metric_names = ['MAE', 'RMSE', 'MAPE', 'sMAPE']
        aggregated = {}
        
        for metric in metric_names:
            values = [result['metrics'][metric] for result in self.cv_results 
                     if result['metrics'][metric] != float('inf')]
            
            if values:
                aggregated[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
            else:
                aggregated[metric] = {
                    'mean': float('inf'),
                    'std': 0,
                    'min': float('inf'),
                    'max': float('inf'),
                    'values': []
                }
        
        self.summary_metrics = aggregated
        
        # Sonuçları yazdır
        print(f"   📈 CROSS-VALIDATION SONUÇLARI ({len(self.cv_results)} fold):")
        print(f"   {'Metrik':<8} {'Ortalama':<10} {'Std':<8} {'Min':<8} {'Max':<8}")
        print(f"   {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
        
        for metric in metric_names:
            if aggregated[metric]['mean'] != float('inf'):
                print(f"   {metric:<8} {aggregated[metric]['mean']:<10.2f} "
                      f"{aggregated[metric]['std']:<8.2f} {aggregated[metric]['min']:<8.2f} "
                      f"{aggregated[metric]['max']:<8.2f}")
            else:
                print(f"   {metric:<8} {'N/A':<10} {'N/A':<8} {'N/A':<8} {'N/A':<8}")
        
        # Fold detayları
        print(f"\n   📋 FOLD DETAYLARI:")
        for result in self.cv_results:
            fold = result['fold']
            mae = result['metrics']['MAE']
            smape = result['metrics']['sMAPE']
            train_days = result['split_info']['train_days']
            val_days = result['split_info']['val_days']
            print(f"   • Fold {fold}: MAE={mae:.2f}, sMAPE={smape:.2f}% "
                  f"(Train: {train_days}d, Valid: {val_days}d)")
    
    def generate_reports(self):
        """JSON ve Markdown raporları oluştur"""
        
        print(f"\n📝 5. Raporlar oluşturuluyor...")
        
        # JSON raporu
        json_report = {
            'experiment_info': {
                'method': 'Time Series Cross-Validation (Rolling-Origin)',
                'validation_horizon': self.validation_horizon,
                'n_splits_requested': self.n_splits,
                'n_splits_completed': len(self.cv_results),
                'timestamp': datetime.now().isoformat()
            },
            'methodology': {
                'approach': 'Rolling-Origin Validation',
                'why_not_shuffle': [
                    'Temporal Leakage: Gelecek verisi ile geçmiş tahmin edilir (data leakage)',
                    'Pattern Bozukluk: Zaman bağımlı pattern\'ler parçalanır',
                    'Gerçekçi Olmama: Production\'da shuffle yok, sadece geçmiş var'
                ],
                'split_strategy': 'Expanding window train, fixed validation horizon'
            },
            'summary_metrics': self.summary_metrics,
            'fold_results': self.cv_results
        }
        
        json_path = f'{self.artifacts_path}/reports/tscv_report.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, ensure_ascii=False, default=str)
        print(f"   ✓ JSON raporu: {json_path}")
        
        # Markdown raporu
        md_content = self._generate_markdown_report()
        
        md_path = f'{self.artifacts_path}/reports/tscv_report.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"   ✓ Markdown raporu: {md_path}")
    
    def _generate_markdown_report(self):
        """Markdown raporu oluştur"""
        
        md = f"""# Time Series Cross-Validation Raporu

**Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Method:** Rolling-Origin Cross-Validation
**Model:** LightGBM Regressor

## 🎯 Amaç

Zaman serisi için uygun çapraz doğrulama ile model performansını robust şekilde ölçmek.

## ⚠️ Neden Shuffle CV Olmaz?

1. **Temporal Leakage**: Gelecek verisi ile geçmiş tahmin edilir (data leakage)
2. **Pattern Bozukluk**: Zaman bağımlı pattern'ler parçalanır
3. **Gerçekçi Olmama**: Production'da shuffle yok, sadece geçmiş var

## 📊 Cross-Validation Yapısı

- **Validation Horizon:** {self.validation_horizon} gün
- **Toplam Fold:** {len(self.cv_results)} (başarılı)
- **Yaklaşım:** Rolling-Origin (Expanding window train)

### Fold Detayları

| Fold | Train Başlangıç | Train Bitiş | Valid Başlangıç | Valid Bitiş | Train Gün | Valid Gün |
|------|-----------------|-------------|-----------------|-------------|-----------|-----------|"""
        
        for result in self.cv_results:
            split_info = result['split_info']
            md += f"""
| {split_info['fold']} | {split_info['train_start'].strftime('%Y-%m-%d')} | {split_info['train_end'].strftime('%Y-%m-%d')} | {split_info['val_start'].strftime('%Y-%m-%d')} | {split_info['val_end'].strftime('%Y-%m-%d')} | {split_info['train_days']} | {split_info['val_days']} |"""
        
        md += f"""

## 📈 Performans Sonuçları

### Özet Metrikler

| Metrik | Ortalama | Std Sapma | Min | Max |
|--------|----------|-----------|-----|-----|"""
        
        for metric_name, metric_data in self.summary_metrics.items():
            if metric_data['mean'] != float('inf'):
                md += f\"\"\"
| {metric_name} | {metric_data['mean']:.2f} | {metric_data['std']:.2f} | {metric_data['min']:.2f} | {metric_data['max']:.2f} |\"\"\"
            else:
                md += f\"\"\"
| {metric_name} | N/A | N/A | N/A | N/A |\"\"\"
        
        md += f\"\"\"

### Fold Bazında Detaylar

| Fold | MAE | RMSE | sMAPE (%) | Model Iterasyon |
|------|-----|------|-----------|-----------------|\"\"\"
        
        for result in self.cv_results:
            metrics = result['metrics']
            model_info = result['model_info']
            md += f\"\"\"
| {result['fold']} | {metrics['MAE']:.2f} | {metrics['RMSE']:.2f} | {metrics['sMAPE']:.2f} | {model_info['best_iteration']} |\"\"\"
        
        md += f\"\"\"

## 🔍 Analiz ve Yorumlar

### Model Tutarlılığı
\"\"\"
        
        if len(self.cv_results) >= 2:
            smape_values = [r['metrics']['sMAPE'] for r in self.cv_results]
            smape_std = np.std(smape_values)
            
            if smape_std < 5:
                md += \"- **Yüksek tutarlılık**: sMAPE standart sapması düşük ({:.2f}%)\\n\".format(smape_std)
            elif smape_std < 10:
                md += \"- **Orta tutarlılık**: sMAPE standart sapması orta ({:.2f}%)\\n\".format(smape_std)
            else:
                md += \"- **Düşük tutarlılık**: sMAPE standart sapması yüksek ({:.2f}%)\\n\".format(smape_std)
        
        # En iyi fold
        best_fold = min(self.cv_results, key=lambda x: x['metrics']['sMAPE'])
        worst_fold = max(self.cv_results, key=lambda x: x['metrics']['sMAPE'])
        
        md += f\"\"\"
- **En iyi fold**: Fold {best_fold['fold']} (sMAPE: {best_fold['metrics']['sMAPE']:.2f}%)
- **En kötü fold**: Fold {worst_fold['fold']} (sMAPE: {worst_fold['metrics']['sMAPE']:.2f}%)

### Production Önerileri

1. **Model Güvenilirliği**: CV sonuçları model performansının robust bir ölçümünü sağlar
2. **Temporal Validation**: Rolling-origin yaklaşımı production senaryosunu yansıtır
3. **Performance Beklentisi**: Ortalama sMAPE {self.summary_metrics['sMAPE']['mean']:.2f}% ±{self.summary_metrics['sMAPE']['std']:.2f}%

### Sınırlamalar

- Basit iteratif forecasting kullanıldı (production için iyileştirilebilir)
- Sadece LightGBM test edildi (ensemble modeller denenebilir)
- Sabit validation horizon (adaptive horizon test edilebilir)

---
*Bu rapor otomatik olarak oluşturulmuştur - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
\"\"\"
        
        return md
    
    def run_full_pipeline(self):
        \"\"\"Tam time series CV pipeline'ını çalıştır\"\"\"
        
        try:
            # 1. Veri yükleme
            self.load_data()
            
            # 2. Split'leri oluştur
            self.create_time_series_splits()
            
            # 3. Cross-validation çalıştır
            self.run_cross_validation()
            
            # 4. Metrikleri topla
            self.aggregate_metrics()
            
            # 5. Raporları oluştur
            self.generate_reports()
            
            print(f\"\\n🎉 Time Series Cross-Validation tamamlandı!\")
            print(f\"📊 {len(self.cv_results)} fold başarıyla çalıştırıldı\")
            if self.summary_metrics and 'sMAPE' in self.summary_metrics:
                print(f\"📈 Ortalama sMAPE: {self.summary_metrics['sMAPE']['mean']:.2f}% ±{self.summary_metrics['sMAPE']['std']:.2f}%\")
            print(f\"📁 Raporlar: {self.artifacts_path}/reports/\")
            
            return self.cv_results, self.summary_metrics
            
        except Exception as e:
            print(f\"\\n❌ Time Series CV Pipeline hatası: {e}\")
            import traceback
            traceback.print_exc()
            raise

def main():
    \"\"\"Ana çalıştırma fonksiyonu\"\"\"
    
    if not LIGHTGBM_AVAILABLE:
        print(\"❌ LightGBM kütüphanesi gerekli. 'pip install lightgbm' ile kurun.\")
        return
    
    print(\"=\" * 70)
    print(\"TIME SERIES CROSS-VALIDATION (ROLLING-ORIGIN)\")
    print(\"🎯 Amaç: Zaman serisi için uygun CV ile robust performans ölçümü\")
    print(\"=\" * 70)
    
    try:
        # Time Series CV'yi başlat
        tscv = TimeSeriesCV(validation_horizon=28, n_splits=3)
        
        # Tam pipeline'ı çalıştır
        cv_results, summary_metrics = tscv.run_full_pipeline()
        
        print(f\"\\n✅ İşlem başarıyla tamamlandı!\")
        print(f\"\\n💡 Time Series CV Faydaları:\")
        print(f\"   • Temporal leakage'ı önler\")
        print(f\"   • Gerçekçi performans ölçümü\")
        print(f\"   • Production senaryosunu yansıtır\")
        print(f\"   • Model tutarlılığını test eder\")
        
    except KeyboardInterrupt:
        print(f\"\\n⏹️  Kullanıcı tarafından durduruldu\")
    except Exception as e:
        print(f\"\\n💥 Beklenmeyen hata: {e}\")

if __name__ == \"__main__\":
    main()
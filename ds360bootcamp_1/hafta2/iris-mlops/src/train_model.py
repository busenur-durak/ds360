import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import os

def train_model(model_type='random_forest'):
    """Model eğit ve kaydet"""
    
    # İşlenmiş veriyi yükle
    df = pd.read_csv('data/processed/titanic_processed.csv')
    
    # X ve y'yi ayır - seaborn dataset kolonlarına göre güncellendi
    feature_cols = ['pclass', 'sex_encoded', 'age', 'sibsp', 'parch', 
                   'fare', 'embarked_encoded', 'family_size', 'is_alone', 'age_group_encoded']
    X = df[feature_cols]
    y = df['survived']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Model seç
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    else:  # logistic_regression
        model = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
    
    # Model eğit
    model.fit(X_train, y_train)
    
    # Tahminler
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrikler
    accuracy = accuracy_score(y_test, y_pred)
    
    # Model kaydet
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{model_type}_model.pkl'
    joblib.dump(model, model_path)
    
    # Metrikleri kaydet
    metrics = {
        'model_type': model_type,
        'accuracy': float(accuracy),
        'n_features': len(feature_cols),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test)
    }
    
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Özellik listesini kaydet
    with open('models/features.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)
    
    print(f"✅ Model eğitildi: {model_type}")
    print(f"📊 Accuracy: {accuracy:.4f}")
    print(f"💾 Model kaydedildi: {model_path}")
    
    # Detaylı rapor
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("burda sadece değişiklik var")
    
    return model, metrics

if __name__ == "__main__":
    # Random Forest modeli eğit
    model, metrics = train_model('random_forest')
    
    # Logistic Regression da eğit
    model_lr, metrics_lr = train_model('logistic_regression')
    
    print("\n🎯 Her iki model de eğitildi ve kaydedildi!")
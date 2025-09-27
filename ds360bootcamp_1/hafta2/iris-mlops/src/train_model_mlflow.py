import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import joblib
import os

def train_with_mlflow():
    """MLflow ile model eğitimi ve takip"""
    
    # İşlenmiş veriyi yükle
    df = pd.read_csv('/Users/yaseminarslan/Desktop/ds360_ikincihafta/hafta2/titanic-mlops/data/processed/titanic_processed.csv')
    
    # X ve y'yi ayır - seaborn dataset kolonlarına göre güncellendi
    feature_cols = ['pclass', 'sex_encoded', 'age', 'sibsp', 'parch', 
                   'fare', 'embarked_encoded', 'family_size', 'is_alone', 'age_group_encoded']
    X = df[feature_cols]
    y = df['survived']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # MLflow experiment
    mlflow.set_experiment("Titanic Survival Prediction")
    
    # Farklı modeller test et
    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    best_model = None
    best_accuracy = 0
    
    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{model_name}_experiment"):
            
            # Model parametrelerini logla
            if model_name == 'random_forest':
                mlflow.log_param("n_estimators", 100)
                mlflow.log_param("max_depth", 10)
            
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("n_features", len(feature_cols))
            
            # Model eğit
            model.fit(X_train, y_train)
            
            # Tahminler
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrikler
            accuracy = accuracy_score(y_test, y_pred)
            
            # MLflow'a kaydet
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("train_size", len(X_train))
            mlflow.log_metric("test_size", len(X_test))
            
            # Model kaydet
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=f"titanic_{model_name}"
            )
            
            # En iyi modeli takip et
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name
            
            print(f"✅ {model_name} - Accuracy: {accuracy:.4f}")
    
    print(f"\n🏆 En iyi model: {best_model} (Accuracy: {best_accuracy:.4f})")
    
    return best_model, best_accuracy

if __name__ == "__main__":
    print("🚀 MLflow ile model eğitimine başlıyor...")
    best_model, best_accuracy = train_with_mlflow()
    print("\n📊 MLflow UI'yi görmek için: mlflow ui")
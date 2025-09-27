import pytest
import pandas as pd
import numpy as np
import joblib
import json
from src.data_preparation import prepare_data, load_titanic_data

def test_data_preparation():
    """Veri hazırlama fonksiyonunu test et"""
    # Test verisi oluştur
    df = load_titanic_data()
    X, y, features = prepare_data(df)
    
    # Assertions
    assert X.shape[0] == y.shape[0], "X ve y boyutları eşleşmeli"
    assert len(features) == X.shape[1], "Özellik sayısı eşleşmeli"
    assert 'Sex_encoded' in features, "Sex_encoded özelliği olmalı"
    assert X.isnull().sum().sum() == 0, "Eksik değer olmamalı"

def test_model_loading():
    """Model yükleme testleri"""
    try:
        model = joblib.load('models/logistic_regression_model.pkl')
        assert model is not None, "Model yüklenmeli"
        
        # Features kontrolü
        with open('models/features.json', 'r') as f:
            features = json.load(f)
        assert len(features) == 9, "9 özellik olmalı"
        
    except FileNotFoundError:
        pytest.skip("Model dosyası bulunamadı")

def test_model_prediction():
    """Model tahmin testleri"""
    try:
        model = joblib.load('models/logistic_regression_model.pkl')
        
        # Test verisi
        test_input = np.array([[3, 1, 25, 0, 0, 8.5, 0, 1, 1]])  # Erkek, 3. sınıf
        
        prediction = model.predict(test_input)
        prediction_proba = model.predict_proba(test_input)
        
        assert prediction[0] in [0, 1], "Tahmin 0 veya 1 olmalı"
        assert len(prediction_proba[0]) == 2, "İki sınıf olasılığı olmalı"
        assert np.isclose(prediction_proba[0].sum(), 1.0), "Olasılıklar toplamı 1 olmalı"
        
    except FileNotFoundError:
        pytest.skip("Model dosyası bulunamadı")

def test_metrics():
    """Metrik dosyası testleri"""
    try:
        with open('models/metrics.json', 'r') as f:
            metrics = json.load(f)
        
        assert 'accuracy' in metrics, "Accuracy metriği olmalı"
        assert 0 <= metrics['accuracy'] <= 1, "Accuracy 0-1 arasında olmalı"
        
    except FileNotFoundError:
        pytest.skip("Metrics dosyası bulunamadı")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
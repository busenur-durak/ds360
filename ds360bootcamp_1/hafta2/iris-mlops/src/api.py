from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import json
import os

# FastAPI uygulaması
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Titanic yolcu verilerine göre hayatta kalma tahmini",
    version="1.0.0"
)

# Veri modelleri
class PassengerData(BaseModel):
    pclass: int  # 1, 2, veya 3
    sex: str     # 'male' veya 'female'
    age: float
    sibsp: int   # Kardeş/eş sayısı
    parch: int   # Ebeveyn/çocuk sayısı
    fare: float
    embarked: str  # 'S', 'C', veya 'Q'

class PredictionResponse(BaseModel):
    survived_probability: float
    survival_prediction: str
    passenger_class: int
    
# Global model değişkeni
model = None
features = None

def load_model():
    """Eğitilmiş modeli yükle"""
    global model, features
    
    try:
        # En iyi performanslı modeli yükle (logistic regression)
        model_path = "models/logistic_regression_model.pkl"
        model = joblib.load(model_path)
        
        # Özellik listesini yükle
        with open("models/features.json", "r") as f:
            features = json.load(f)
            
        print("✅ Model başarıyla yüklendi!")
        
    except FileNotFoundError:
        print("❌ Model dosyası bulunamadı. Önce model eğitilmeli.")
        raise

def preprocess_passenger_data(passenger: PassengerData):
    """Yolcu verisini model için hazırla"""
    
    # Sex encoding (male=1, female=0)
    sex_encoded = 1 if passenger.sex.lower() == 'male' else 0
    
    # Embarked encoding (S=0, C=1, Q=2)
    embarked_map = {'S': 0, 'C': 1, 'Q': 2}
    embarked_encoded = embarked_map.get(passenger.embarked.upper(), 0)
    
    # Yeni özellikler
    family_size = passenger.sibsp + passenger.parch + 1
    is_alone = 1 if family_size == 1 else 0
    
    # Yaş grubu encoding
    if passenger.age <= 18:
        age_group_encoded = 0  # child
    elif passenger.age <= 35:
        age_group_encoded = 1  # young_adult
    elif passenger.age <= 60:
        age_group_encoded = 2  # adult
    else:
        age_group_encoded = 3  # senior
    
    # Özellik vektörü oluştur - clean_data.py'deki sıraya göre
    feature_vector = [
        passenger.pclass,      # pclass
        sex_encoded,           # sex_encoded
        passenger.age,         # age
        passenger.sibsp,       # sibsp
        passenger.parch,       # parch
        passenger.fare,        # fare
        embarked_encoded,      # embarked_encoded
        family_size,           # family_size
        is_alone,              # is_alone
        age_group_encoded      # age_group_encoded
    ]
    
    return np.array(feature_vector).reshape(1, -1)

@app.on_event("startup")
async def startup_event():
    """Uygulama başlarken modeli yükle"""
    load_model()

@app.get("/")
async def root():
    """Ana endpoint"""
    return {
        "message": "Titanic Survival Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "model_info": "/model/info"
        }
    }

@app.get("/health")
async def health_check():
    """Sağlık kontrolü"""
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded
    }

@app.get("/model/info")
async def model_info():
    """Model bilgileri"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model yüklenmedi")
    
    # Metrics dosyasından bilgi al
    try:
        with open("models/metrics.json", "r") as f:
            metrics = json.load(f)
        return {
            "model_type": "logistic_regression",
            "features": features,
            "metrics": metrics
        }
    except FileNotFoundError:
        return {
            "model_type": "logistic_regression",
            "features": features,
            "metrics": "Metrics dosyası bulunamadı"
        }

@app.post("/predict", response_model=PredictionResponse)
async def predict_survival(passenger: PassengerData):
    """Hayatta kalma tahmini yap"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model yüklenmedi")
    
    try:
        # Veriyi hazırla
        X = preprocess_passenger_data(passenger)
        
        # Tahmin yap
        prediction = model.predict(X)[0]
        prediction_proba = model.predict_proba(X)[0]
        
        # Sonucu hazırla
        survived_prob = float(prediction_proba[1])
        survival_text = "Hayatta kalır" if prediction == 1 else "Hayatta kalamaz"
        
        return PredictionResponse(
            survived_probability=survived_prob,
            survival_prediction=survival_text,
            passenger_class=passenger.pclass
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Tahmin hatası: {str(e)}")

@app.get("/predict/example")
async def prediction_example():
    """Örnek tahmin isteği"""
    return {
        "example_request": {
            "pclass": 3,
            "sex": "male",
            "age": 25.0,
            "sibsp": 0,
            "parch": 0,
            "fare": 8.5,
            "embarked": "S"
        },
        "how_to_use": "POST /predict ile yukarıdaki formatı kullanın"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
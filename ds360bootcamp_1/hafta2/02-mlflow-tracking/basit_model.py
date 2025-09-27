# Basit makine öğrenmesi modeli
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# Basit veri oluştur
data = {
    'boy': [160, 170, 180, 165, 175],
    'kilo': [60, 70, 80, 65, 75],
    'cinsiyet': [0, 1, 1, 0, 1]  # 0: kadın, 1: erkek
}

df = pd.DataFrame(data)
X = df[['boy', 'kilo']]
y = df['cinsiyet']

# Veriyi böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# MLflow experiment başlat
mlflow.set_experiment("Basit Cinsiyet Tahmini")

with mlflow.start_run():
    # Model oluştur ve eğit
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Tahmin yap
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # MLflow'a kaydet
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Model eğitildi!")
    print(f"Doğruluk: {accuracy:.2f}")
    print("Sonuçlar MLflow'a kaydedildi.")
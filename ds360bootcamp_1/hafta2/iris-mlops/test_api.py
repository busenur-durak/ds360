import requests
import json

# API base URL
BASE_URL = "http://localhost:8001"

def test_api():
    """API'yi test et"""
    
    print("🔄 Titanic API testleri başlıyor...")
    
    try:
        # 1. Sağlık kontrolü
        response = requests.get(f"{BASE_URL}/health")
        print(f"1. ✅ Sağlık kontrolü: {response.json()}")
        
        # 2. Ana sayfa
        response = requests.get(f"{BASE_URL}/")
        print(f"2. ✅ Ana sayfa: {response.json()['message']}")
        
        # 3. Model bilgileri
        response = requests.get(f"{BASE_URL}/model/info")
        model_info = response.json()
        print(f"3. ✅ Model tipi: {model_info.get('model_type', 'Bilinmiyor')}")
        
        # 4. Örnek veri
        response = requests.get(f"{BASE_URL}/predict/example")
        example = response.json()
        print(f"4. ✅ Örnek veri alındı")
        
        # 5. Test tahminleri
        test_passengers = [
            {
                "Pclass": 1,
                "Sex": "female",
                "Age": 25.0,
                "SibSp": 0,
                "Parch": 0,
                "Fare": 100.0,
                "Embarked": "S"
            },
            {
                "Pclass": 3,
                "Sex": "male",
                "Age": 30.0,
                "SibSp": 0,
                "Parch": 0,
                "Fare": 8.5,
                "Embarked": "S"
            },
            {
                "Pclass": 2,
                "Sex": "female",
                "Age": 35.0,
                "SibSp": 1,
                "Parch": 1,
                "Fare": 25.0,
                "Embarked": "C"
            }
        ]
        
        print("\n🎯 Tahmin testleri:")
        for i, passenger in enumerate(test_passengers, 1):
            response = requests.post(f"{BASE_URL}/predict", json=passenger)
            
            if response.status_code == 200:
                result = response.json()
                print(f"{i}. Yolcu ({passenger['Sex']}, Sınıf {passenger['Pclass']}):")
                print(f"   📊 Hayatta kalma olasılığı: {result['survived_probability']:.2%}")
                print(f"   🎯 Tahmin: {result['survival_prediction']}")
            else:
                print(f"{i}. ❌ Hata: {response.status_code}")
        
        print("\n✅ Tüm testler başarıyla tamamlandı!")
        
    except requests.exceptions.ConnectionError:
        print("❌ API'ye bağlanılamıyor.")
        print("Önce API'yi başlatın: uvicorn src.api:app --reload")
    except Exception as e:
        print(f"❌ Test hatası: {e}")

if __name__ == "__main__":
    test_api()
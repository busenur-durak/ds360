import requests
import json

# API testi
base_url = "http://localhost:8000"

def test_api():
    print("🔄 API testleri başlıyor...")
    
    # 1. Ana sayfa testi
    response = requests.get(f"{base_url}/")
    print("1. Ana sayfa:", response.json())
    
    # 2. Sağlık kontrolü
    response = requests.get(f"{base_url}/health")
    print("2. Sağlık:", response.json())
    
    # 3. Örnek veri
    response = requests.get(f"{base_url}/ornek")
    print("3. Örnek:", response.json())
    
    # 4. Tahmin testi
    test_data = {"boy": 175.0, "kilo": 70.0}
    response = requests.post(f"{base_url}/tahmin", json=test_data)
    print("4. Tahmin:", response.json())
    
    print("✅ Tüm testler tamamlandı!")

if __name__ == "__main__":
    try:
        test_api()
    except requests.exceptions.ConnectionError:
        print("❌ API'ye bağlanılamıyor. Önce 'uvicorn app:app --reload' çalıştırın.")
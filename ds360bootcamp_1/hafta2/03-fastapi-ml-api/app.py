from fastapi import FastAPI
from pydantic import BaseModel

# FastAPI uygulaması oluştur
app = FastAPI(title="Basit ML API", version="1.0.0")

# Veri modeli (giriş için)
class Kisi(BaseModel):
    boy: float
    kilo: float

# Veri modeli (çıkış için)
class Tahmin(BaseModel):
    cinsiyet: str
    olasilik: float

# Basit tahmin fonksiyonu
def cinsiyet_tahmin_et(boy: float, kilo: float):
    # Çok basit kural: boy > 170 ise erkek
    if boy > 170:
        return "erkek", 0.8
    else:
        return "kadın", 0.7

# Ana sayfa
@app.get("/")
def ana_sayfa():
    return {"mesaj": "ML API'ye hoş geldiniz!"}

# Sağlık kontrolü
@app.get("/health")
def saglik():
    return {"durum": "sağlıklı"}

# Tahmin endpoint'i
@app.post("/tahmin", response_model=Tahmin)
def tahmin_yap(kisi: Kisi):
    cinsiyet, olasilik = cinsiyet_tahmin_et(kisi.boy, kisi.kilo)
    
    return Tahmin(
        cinsiyet=cinsiyet,
        olasilik=olasilik
    )

# Örnek veri endpoint'i
@app.get("/ornek")
def ornek_veri():
    return {
        "ornek_istek": {
            "boy": 175.0,
            "kilo": 70.0
        },
        "ornek_cevap": {
            "cinsiyet": "erkek",
            "olasilik": 0.8
        }
    }
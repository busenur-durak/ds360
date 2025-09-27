# FastAPI uygulaması (Docker için)
from fastapi import FastAPI

app = FastAPI(title="Docker ML API")

@app.get("/")
def ana_sayfa():
    return {"mesaj": "Docker'da çalışan ML API!"}

@app.get("/health")
def saglik():
    return {"durum": "Docker container sağlıklı"}
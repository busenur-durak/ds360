# Basit veri oluşturma scripti
import pandas as pd

# Basit bir CSV dosyası oluştur
data = {
    'isim': ['Ali', 'Ayşe', 'Mehmet', 'Fatma'],
    'yas': [25, 30, 35, 28],
    'maas': [5000, 6000, 7000, 5500]
}

df = pd.DataFrame(data)
df.to_csv('data.csv', index=False)
print("data.csv dosyası oluşturuldu!")
print(df)
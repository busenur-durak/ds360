import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def clean_titanic_data(input_path='data/raw/titanic.csv', output_path='data/processed/titanic_processed.csv'):
    """Titanic veri setini temizle ve özellik mühendisliği yap"""
    
    # Veriyi yükle
    df = pd.read_csv(input_path)
    
    # Kopyasını al
    df_clean = df.copy()
    
    # Eksik değerleri doldur
    df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
    df_clean['embarked'] = df_clean['embarked'].fillna(df_clean['embarked'].mode()[0])
    
    # Gereksiz kolonları çıkar
    if 'deck' in df_clean.columns:
        df_clean = df_clean.drop('deck', axis=1)
    
    # Kategorik değişkenleri encode et
    le_sex = LabelEncoder()
    df_clean['sex_encoded'] = le_sex.fit_transform(df_clean['sex'])
    
    le_embarked = LabelEncoder()
    df_clean['embarked_encoded'] = le_embarked.fit_transform(df_clean['embarked'])
    
    # Yeni özellikler oluştur
    df_clean['family_size'] = df_clean['sibsp'] + df_clean['parch'] + 1
    df_clean['is_alone'] = (df_clean['family_size'] == 1).astype(int)
    
    # Fare'deki eksik değerleri doldur (eğer varsa)
    if df_clean['fare'].isnull().any():
        df_clean['fare'] = df_clean['fare'].fillna(df_clean['fare'].median())
    
    # Yaş grupları oluştur
    df_clean['age_group'] = pd.cut(df_clean['age'], 
                                   bins=[0, 18, 35, 60, 100], 
                                   labels=['child', 'young_adult', 'adult', 'senior'])
    
    le_age_group = LabelEncoder()
    df_clean['age_group_encoded'] = le_age_group.fit_transform(df_clean['age_group'])
    
    # Çıktı dizinini oluştur
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Temizlenmiş veriyi kaydet
    df_clean.to_csv(output_path, index=False)
    
    print("✅ Veri temizlendi ve kaydedildi:", output_path)
    print(f"Orijinal boyut: {df.shape}")
    print(f"Temizlenmiş boyut: {df_clean.shape}")
    print(f"Eksik değerler:\n{df_clean.isnull().sum().sum()} toplam eksik değer")
    print("merhaba ey değişiklik")
    
    # Özellik listesini döndür
    features = ['pclass', 'sex_encoded', 'age', 'sibsp', 'parch', 
                'fare', 'embarked_encoded', 'family_size', 'is_alone', 'age_group_encoded']
    
    print(f"Model özellikleri: {features}")
    
    return df_clean, features

if __name__ == "__main__":
    clean_titanic_data()
import pandas as pd
import seaborn as sns
import os

def download_titanic_data():
    """Seaborn'dan Titanic veri setini indir"""
    
    # Veri dizinlerini oluştur
    os.makedirs('data/raw', exist_ok=True)
    
    # Seaborn'dan Titanic veri setini yükle
    df = sns.load_dataset('titanic')
    
    # Ham veriyi kaydet
    df.to_csv('data/raw/titanic.csv', index=False)
    
    print("✅ Titanic veri seti indirildi: data/raw/titanic.csv")
    print(f"Veri boyutu: {df.shape}")
    print(f"Kolonlar: {list(df.columns)}")
    print(f"Eksik değerler:\n{df.isnull().sum()}")
    
    return df

if __name__ == "__main__":
    download_titanic_data()
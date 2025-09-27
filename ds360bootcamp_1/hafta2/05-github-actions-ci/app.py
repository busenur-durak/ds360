# Basit Python uygulaması
def topla(a, b):
    """İki sayıyı toplar"""
    return a + b

def carp(a, b):
    """İki sayıyı çarpar"""
    return a * b

if __name__ == "__main__":
    print("Hesap makinesi çalışıyor!")
    print(f"5 + 3 = {topla(5, 3)}")
    print(f"4 * 6 = {carp(4, 6)}")
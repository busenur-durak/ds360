# Test dosyası
from app import topla, carp

def test_topla():
    """Toplama fonksiyonunu test et"""
    assert topla(2, 3) == 5
    assert topla(0, 0) == 0
    assert topla(-1, 1) == 0

def test_carp():
    """Çarpma fonksiyonunu test et"""
    assert carp(2, 3) == 6
    assert carp(0, 5) == 0
    assert carp(-2, 3) == -6

if __name__ == "__main__":
    test_topla()
    test_carp()
    print("✅ Tüm testler geçti!")
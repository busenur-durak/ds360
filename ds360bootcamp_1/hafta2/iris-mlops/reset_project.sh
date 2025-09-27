#!/bin/bash

echo "🧹 Titanic MLOps Projesini Sıfırlıyor..."

# DVC cache ve lock dosyalarını temizle
echo "📦 DVC cache temizleniyor..."
rm -rf .dvc/cache/*
rm -f dvc.lock

# MLflow experiment verilerini temizle
echo "🔬 MLflow experiment verileri temizleniyor..."
rm -rf mlruns/*

# Models klasörünü temizle
echo "🤖 Model dosyaları temizleniyor..."
rm -f models/*

# Data klasörlerini temizle
echo "📊 Veri dosyaları temizleniyor..."
rm -f data/raw/*
rm -f data/processed/*

# Python cache dosyalarını temizle
echo "🐍 Python cache temizleniyor..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Pytest cache temizle
echo "🧪 Test cache temizleniyor..."
rm -rf .pytest_cache

echo "✅ Proje sıfırlandı!"
echo ""
echo "🚀 Artık şu komutlarla başlayabilirsiniz:"
echo "   dvc repro          # Tüm pipeline'ı çalıştır"
echo "   dvc status         # Pipeline durumunu kontrol et"
echo "   dvc metrics show   # Sonuçları görüntüle"
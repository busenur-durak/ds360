#!/bin/bash
# M5 Forecasting Pipeline - Docker Commands

echo "🐳 M5 FORECASTING DOCKER SETUP"
echo "================================"

# 1. Docker Image Build
echo "📦 Building Docker image..."
docker build -t m5-forecast:dev .

# 2. Run Pipeline (with volume mounts)
echo "🚀 Running pipeline..."
docker run --rm \
    -v $(pwd)/artifacts:/app/artifacts \
    -v $(pwd)/data:/app/data \
    m5-forecast:dev

# 3. Alternative: Interactive mode for debugging
echo "🔍 For debugging, run interactive mode:"
echo "docker run -it --rm -v \$(pwd)/artifacts:/app/artifacts -v \$(pwd)/data:/app/data m5-forecast:dev bash"

# 4. Check outputs
echo "📊 Check outputs in ./artifacts/preds/"
ls -la ./artifacts/preds/

echo "✅ Docker pipeline complete!"
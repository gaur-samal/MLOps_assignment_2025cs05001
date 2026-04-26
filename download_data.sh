#!/bin/bash
# Heart Disease UCI Dataset Download Script
# Source: UCI Machine Learning Repository

set -e

DATA_DIR="heart_disease_data"
DATA_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
FILES=("processed.cleveland.data" "processed.hungarian.data" "processed.switzerland.data" "processed.va.data" "heart-disease.names")

echo "=========================================="
echo "Heart Disease UCI Dataset Downloader"
echo "=========================================="

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"

echo "Downloading dataset files..."

for file in "${FILES[@]}"; do
    if [ -f "$DATA_DIR/$file" ]; then
        echo "✓ $file already exists, skipping..."
    else
        echo "Downloading $file..."
        curl -s -o "$DATA_DIR/$file" "${DATA_URL}${file}" && echo "✓ Downloaded $file"
    fi
done

echo ""
echo "=========================================="
echo "Download complete!"
echo "Data directory: $DATA_DIR"
echo "=========================================="

# Verify files
echo ""
echo "Verifying downloaded files:"
for file in "${FILES[@]}"; do
    if [ -f "$DATA_DIR/$file" ]; then
        lines=$(wc -l < "$DATA_DIR/$file")
        echo "✓ $file ($lines lines)"
    else
        echo "✗ $file - MISSING"
    fi
done

echo ""
echo "Primary dataset: processed.cleveland.data (303 samples, 14 features)"
echo "Ready for analysis!"

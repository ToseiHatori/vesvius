#!/bin/bash
# Download required data for Vesuvius evaluation
# Requires: kaggle CLI configured with API credentials

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Downloading Vesuvius Competition Data ==="

# Create directories
mkdir -p tfrecords

# Download competition data (contains TFRecords for training/validation)
echo "Downloading competition data..."
kaggle competitions download -c vesuvius-challenge-surface-detection -p ./competition_data

# Extract
echo "Extracting..."
cd competition_data
unzip -o "*.zip"
cd ..

# Note: The TFRecords might be in a separate dataset
# Check if TFRecords exist in competition data, otherwise they need to be created
# or downloaded from a separate dataset

echo ""
echo "=== Download Complete ==="
echo ""
echo "If TFRecords are not included in the competition data,"
echo "you need to either:"
echo "  1. Create them from the competition's train_images using a conversion script"
echo "  2. Download from a separate Kaggle dataset if available"
echo ""
echo "Place TFRecord files (*.tfrec) in: $SCRIPT_DIR/tfrecords/"

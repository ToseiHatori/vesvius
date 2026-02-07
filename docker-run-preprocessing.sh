#!/bin/bash
# Run nnUNet preprocessing in Kaggle Docker environment
#
# Usage:
#   ./docker-run-preprocessing.sh
#
# Prerequisites:
#   - Docker installed
#   - Dataset downloaded and extracted to ./data/vesuvius-challenge-surface-detection/
#   - NVIDIA GPU with nvidia-docker (optional, but recommended)

set -e

PROJECT_DIR="/home/ben/Dev/vesvius"
DATA_DIR="${PROJECT_DIR}/data"
OUTPUT_DIR="${PROJECT_DIR}/nnunet_output"
WORK_DIR="${PROJECT_DIR}/nnunet_work"

# Kaggle Python GPU image (includes PyTorch, scikit-image, etc.)
KAGGLE_IMAGE="gcr.io/kaggle-gpu-images/python:latest"

# Create output directories
mkdir -p "${OUTPUT_DIR}" "${WORK_DIR}"

echo "=============================================="
echo "Vesuvius nnUNet Preprocessing (Kaggle Docker)"
echo "=============================================="
echo ""
echo "Project dir: ${PROJECT_DIR}"
echo "Data dir:    ${DATA_DIR}"
echo "Output dir:  ${OUTPUT_DIR}"
echo ""

# Check if dataset exists, extract if needed
DATASET_DIR="${DATA_DIR}/vesuvius-challenge-surface-detection"
if [ ! -d "${DATASET_DIR}/train_images" ]; then
    echo "Dataset not found at ${DATASET_DIR}/train_images"

    # Check if zip exists and extract to subdirectory
    ZIP_FILE="${DATA_DIR}/vesuvius-challenge-surface-detection.zip"
    if [ -f "${ZIP_FILE}" ]; then
        echo "Found zip file, extracting..."
        mkdir -p "${DATASET_DIR}"
        cd "${DATASET_DIR}"
        unzip -q "${ZIP_FILE}"
        cd - > /dev/null
        echo "Extraction complete."
    else
        echo "ERROR: Neither dataset directory nor zip file found!"
        echo "Please download the dataset:"
        echo "  cd ${DATA_DIR}"
        echo "  kaggle competitions download -c vesuvius-challenge-surface-detection"
        exit 1
    fi
fi

# Verify extraction
if [ ! -d "${DATASET_DIR}/train_images" ]; then
    echo "ERROR: train_images directory not found after extraction!"
    echo "Please check the dataset structure."
    exit 1
fi

echo "Dataset found: $(ls ${DATASET_DIR}/train_images/*.tif 2>/dev/null | wc -l) training images"

# Check for GPU support
GPU_FLAG=""
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "GPU detected, enabling NVIDIA runtime"
    GPU_FLAG="--gpus all"
else
    echo "No GPU detected, running on CPU"
fi

echo ""
echo "Pulling Kaggle image (this may take a while on first run)..."
docker pull ${KAGGLE_IMAGE}

echo ""
echo "Starting preprocessing..."
echo ""

# Run the preprocessing script in Docker
docker run --rm \
    ${GPU_FLAG} \
    --shm-size=16g \
    -v "${PROJECT_DIR}:/workspace" \
    -v "${DATA_DIR}:/kaggle/input" \
    -v "${OUTPUT_DIR}:/kaggle/working" \
    -v "${WORK_DIR}:/kaggle/temp" \
    -e "nnUNet_raw=/kaggle/temp/nnUNet_data/nnUNet_raw" \
    -e "nnUNet_preprocessed=/kaggle/working/nnUNet_preprocessed" \
    -e "nnUNet_results=/kaggle/working/nnUNet_results" \
    -e "nnUNet_USE_BLOSC2=1" \
    -e "nnUNet_compile=true" \
    -w /workspace \
    ${KAGGLE_IMAGE} \
    bash -c "
        echo 'Installing nnunetv2...'
        pip install -q nnunetv2 nibabel tifffile

        echo ''
        echo 'Running preprocessing script...'
        python /workspace/surface-nnunet-preprocessing-docker.py
    "

echo ""
echo "=============================================="
echo "Preprocessing complete!"
echo "Output: ${OUTPUT_DIR}/nnUNet_preprocessed/"
echo "=============================================="

#!/bin/bash
# Run nnUNet preprocessing with nohup support
#
# Usage:
#   ./run-preprocessing-nohup.sh        # Run in foreground
#   ./run-preprocessing-nohup.sh nohup  # Run in background with nohup
#
# Logs are saved to: preprocessing.log
# Monitor with: tail -f preprocessing.log

set -e

PROJECT_DIR="/home/ben/Dev/vesvius"
LOG_FILE="${PROJECT_DIR}/preprocessing.log"

# If nohup mode requested, re-exec with nohup
if [ "$1" = "nohup" ]; then
    echo "Starting preprocessing in background with nohup..."
    echo "Log file: ${LOG_FILE}"
    echo "Monitor with: tail -f ${LOG_FILE}"
    nohup "$0" > "${LOG_FILE}" 2>&1 &
    echo "PID: $!"
    exit 0
fi

# Main execution starts here
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================="
echo "Vesuvius nnUNet Preprocessing (Kaggle Docker)"
echo "Started at: $(date)"
echo "=============================================="
echo ""

DATA_DIR="${PROJECT_DIR}/data"
OUTPUT_DIR="${PROJECT_DIR}/nnunet_output"
WORK_DIR="${PROJECT_DIR}/nnunet_work"
DATASET_DIR="${DATA_DIR}/vesuvius-challenge-surface-detection"

# Kaggle Python GPU image
KAGGLE_IMAGE="gcr.io/kaggle-gpu-images/python:latest"

# Create output directories
mkdir -p "${OUTPUT_DIR}" "${WORK_DIR}"

echo "Project dir: ${PROJECT_DIR}"
echo "Data dir:    ${DATA_DIR}"
echo "Output dir:  ${OUTPUT_DIR}"
echo ""

# Check if dataset exists, extract if needed
if [ ! -d "${DATASET_DIR}/train_images" ]; then
    echo "Dataset not found at ${DATASET_DIR}/train_images"

    ZIP_FILE="${DATA_DIR}/vesuvius-challenge-surface-detection.zip"
    if [ -f "${ZIP_FILE}" ]; then
        echo "Found zip file, extracting... (this may take a while)"
        mkdir -p "${DATASET_DIR}"
        cd "${DATASET_DIR}"
        unzip -q "${ZIP_FILE}"
        cd "${PROJECT_DIR}"
        echo "Extraction complete at $(date)"
    else
        echo "ERROR: Neither dataset directory nor zip file found!"
        exit 1
    fi
fi

# Verify extraction
if [ ! -d "${DATASET_DIR}/train_images" ]; then
    echo "ERROR: train_images directory not found after extraction!"
    exit 1
fi

NUM_IMAGES=$(ls "${DATASET_DIR}"/train_images/*.tif 2>/dev/null | wc -l)
echo "Dataset found: ${NUM_IMAGES} training images"

# Check for GPU support
GPU_FLAG=""
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "GPU detected, enabling NVIDIA runtime"
    GPU_FLAG="--gpus all"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "No GPU detected, running on CPU (preprocessing only needs CPU)"
fi

echo ""
echo "Pulling Kaggle image (skip if already cached)..."
docker pull ${KAGGLE_IMAGE}

echo ""
echo "Starting Docker container for preprocessing..."
echo "Start time: $(date)"
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
        echo 'Python environment:'
        python --version
        pip show nnunetv2 | grep Version

        echo ''
        echo 'Running preprocessing script...'
        python /workspace/surface-nnunet-preprocessing-docker.py
    "

RESULT=$?

echo ""
echo "=============================================="
if [ ${RESULT} -eq 0 ]; then
    echo "Preprocessing COMPLETE!"
    echo "Output: ${OUTPUT_DIR}/nnUNet_preprocessed/"
    ls -la "${OUTPUT_DIR}/nnUNet_preprocessed/" 2>/dev/null | head -10
else
    echo "Preprocessing FAILED with exit code ${RESULT}"
fi
echo "End time: $(date)"
echo "=============================================="

exit ${RESULT}

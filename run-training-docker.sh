#!/bin/bash
# Run nnUNet training with Docker (Kaggle GPU image)
#
# Usage:
#   ./run-training-docker.sh                    # Train with defaults
#   ./run-training-docker.sh --epochs 50        # Train with 50 epochs
#   ./run-training-docker.sh --inference        # Inference only
#   ./run-training-docker.sh nohup              # Run in background
#
# All arguments after the script name are passed to the Python script.
# Use 'nohup' as first argument to run in background.

set -e

PROJECT_DIR="/home/ben/Dev/vesvius"
LOG_FILE="${PROJECT_DIR}/training.log"

# Check for nohup mode
if [ "$1" = "nohup" ]; then
    shift  # Remove 'nohup' from arguments
    echo "Starting training in background with nohup..."
    echo "Log file: ${LOG_FILE}"
    echo "Monitor with: tail -f ${LOG_FILE}"
    nohup "$0" "$@" > "${LOG_FILE}" 2>&1 &
    echo "PID: $!"
    exit 0
fi

# Main execution
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================="
echo "Vesuvius nnUNet Training (Docker/Kaggle)"
echo "Started at: $(date)"
echo "Arguments: $@"
echo "=============================================="
echo ""

DATA_DIR="${PROJECT_DIR}/data"
OUTPUT_DIR="${PROJECT_DIR}/nnunet_output"
WORK_DIR="${PROJECT_DIR}/nnunet_work"

# Kaggle Python GPU image
KAGGLE_IMAGE="gcr.io/kaggle-gpu-images/python:latest"

# Create directories
mkdir -p "${OUTPUT_DIR}" "${WORK_DIR}"

echo "Project dir: ${PROJECT_DIR}"
echo "Data dir:    ${DATA_DIR}"
echo "Output dir:  ${OUTPUT_DIR}"
echo ""

# Verify preprocessed data exists
PREPROCESSED_DIR="${OUTPUT_DIR}/nnUNet_preprocessed/Dataset100_VesuviusSurface"
if [ ! -d "${PREPROCESSED_DIR}" ]; then
    echo "ERROR: Preprocessed data not found at ${PREPROCESSED_DIR}"
    echo "Run preprocessing first: ./run-preprocessing-nohup.sh"
    exit 1
fi

echo "Preprocessed data found: ${PREPROCESSED_DIR}"

# Check for GPU
GPU_FLAG=""
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "GPU detected, enabling NVIDIA runtime"
    GPU_FLAG="--gpus all"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: No GPU detected! Training will be slow."
fi

echo ""
echo "Pulling Kaggle image (skip if cached)..."
docker pull ${KAGGLE_IMAGE}

echo ""
echo "Starting Docker container for training..."
echo "Start time: $(date)"
echo ""

# Build arguments string for Python script
PYTHON_ARGS="$@"
if [ -z "$PYTHON_ARGS" ]; then
    PYTHON_ARGS="--train"  # Default to training
fi

# Run the training script in Docker
docker run --rm \
    ${GPU_FLAG} \
    --shm-size=32g \
    --ipc=host \
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
        echo 'Installing dependencies...'
        pip install -q nnunetv2 nibabel tifffile tqdm

        echo ''
        echo 'Python environment:'
        python --version
        pip show nnunetv2 | grep Version

        echo ''
        echo 'Running training script with args: ${PYTHON_ARGS}'
        python /workspace/surface-nnunet-training-local.py ${PYTHON_ARGS}
    "

RESULT=$?

echo ""
echo "=============================================="
if [ ${RESULT} -eq 0 ]; then
    echo "Training COMPLETE!"
    echo "Output: ${OUTPUT_DIR}/nnUNet_results/"
    ls -la "${OUTPUT_DIR}/nnUNet_results/" 2>/dev/null | head -10
else
    echo "Training FAILED with exit code ${RESULT}"
fi
echo "End time: $(date)"
echo "=============================================="

exit ${RESULT}

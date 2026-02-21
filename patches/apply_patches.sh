#!/bin/bash
# Apply nnUNet patches for transfer learning support
#
# This script copies modified nnUNet trainer files to support
# NNUNET_INITIAL_LR environment variable for transfer learning.
#
# Usage:
#   docker-compose exec nnunet /workspace/patches/apply_patches.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNUNET_DIR="/usr/local/lib/python3.12/dist-packages/nnunetv2"

echo "Applying nnUNet patches..."

# Backup original file
TRAINER_FILE="${NNUNET_DIR}/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs.py"
if [ -f "$TRAINER_FILE" ] && [ ! -f "${TRAINER_FILE}.bak" ]; then
    cp "$TRAINER_FILE" "${TRAINER_FILE}.bak"
    echo "Backed up original: ${TRAINER_FILE}.bak"
fi

# Copy patched file
cp "${SCRIPT_DIR}/nnUNetTrainer_Xepochs.py" "$TRAINER_FILE"
echo "Applied patch: nnUNetTrainer_Xepochs.py"

# Clear Python cache
find "${NNUNET_DIR}" -name "*.pyc" -delete 2>/dev/null || true
find "${NNUNET_DIR}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "Cleared Python cache"

echo "Done! Transfer learning with --initial-lr is now supported."

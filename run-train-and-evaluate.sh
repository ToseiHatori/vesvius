#!/bin/bash
# =============================================================================
# Vesuvius: Training + Softmax + Evaluation Pipeline
# =============================================================================
#
# 学習 → validation推論（NPZ/softmax） → 評価（none + hysteresis + opening_closing）を一連で実行
#
# Usage:
#   # 学習から評価まで一連実行（nohup推奨）
#   nohup ./run-train-and-evaluate.sh \
#       --trainer nnUNetTrainer_2000epochs \
#       --plans nnUNetResEncUNetMPlans \
#       --config 3d_lowres \
#       --fold 0 \
#       > logs/train_eval_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
#   # 学習をスキップ（推論と評価のみ）
#   ./run-train-and-evaluate.sh --skip-train --trainer ... --fold 0
#
#   # 推論もスキップ（評価のみ）
#   ./run-train-and-evaluate.sh --eval-only --npz-dir /path/to/npz
#
#   # Transfer learning（既存モデルから継続学習）
#   nohup ./run-train-and-evaluate.sh \
#       --trainer nnUNetTrainer_1000epochs \
#       --config 3d_fullres \
#       --fold 0 \
#       --pretrained-weights /path/to/checkpoint_final.pth \
#       --initial-lr 0.001 \
#       > logs/train_transfer.log 2>&1 &
#
# =============================================================================

set -e

PROJECT_DIR="/home/ben/Dev/vesvius"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default values
TRAINER="nnUNetTrainer"
PLANS="nnUNetResEncUNetMPlans"
CONFIG="3d_lowres"
FOLD="0"
EPOCHS=""
PRETRAINED_WEIGHTS=""
INITIAL_LR=""
SKIP_TRAIN=false
SKIP_INFERENCE=false
EVAL_ONLY=false
NPZ_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --trainer)
            TRAINER="$2"
            shift 2
            ;;
        --plans)
            PLANS="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --fold)
            FOLD="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --pretrained-weights)
            PRETRAINED_WEIGHTS="$2"
            shift 2
            ;;
        --initial-lr)
            INITIAL_LR="$2"
            shift 2
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-inference)
            SKIP_INFERENCE=true
            shift
            ;;
        --eval-only)
            EVAL_ONLY=true
            SKIP_TRAIN=true
            SKIP_INFERENCE=true
            shift
            ;;
        --npz-dir)
            NPZ_DIR="$2"
            shift 2
            ;;
        --host-baseline)
            TRAINER="nnUNetTrainerMedialSurfaceRecall"
            PLANS="nnUNetHostBaselinePlans"
            # CONFIGは明示的に --config で指定可能（デフォルトは3d_lowres）
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build model directory name and inference trainer name (match Python's get_trainer_name logic)
if [[ "$TRAINER" != "nnUNetTrainer" ]]; then
    # Custom trainers don't use epoch suffix
    INFERENCE_TRAINER="${TRAINER}"
    MODEL_NAME="${TRAINER}__${PLANS}__${CONFIG}"
elif [[ "$EPOCHS" != "" ]]; then
    INFERENCE_TRAINER="nnUNetTrainer_${EPOCHS}epochs"
    MODEL_NAME="nnUNetTrainer_${EPOCHS}epochs__${PLANS}__${CONFIG}"
else
    INFERENCE_TRAINER="${TRAINER}"
    MODEL_NAME="${TRAINER}__${PLANS}__${CONFIG}"
fi

# Set NPZ directory if not specified
if [[ "$NPZ_DIR" == "" ]]; then
    NPZ_DIR="/workspace/nnunet_output/nnUNet_results/Dataset100_VesuviusSurface/${MODEL_NAME}/fold_${FOLD}/validation_npz"
fi

echo "=============================================="
echo "Vesuvius Training + Evaluation Pipeline"
echo "Started at: $(date)"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Trainer: ${TRAINER}"
echo "  Plans: ${PLANS}"
echo "  Config: ${CONFIG}"
echo "  Fold: ${FOLD}"
echo "  Epochs: ${EPOCHS:-default}"
echo "  Pretrained: ${PRETRAINED_WEIGHTS:-none}"
echo "  Initial LR: ${INITIAL_LR:-default}"
echo "  Model: ${MODEL_NAME}"
echo "  NPZ Dir: ${NPZ_DIR}"
echo ""

# Create directories
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/docs/results"

# =============================================================================
# GPU Check Function
# =============================================================================
check_gpu() {
    local task_name="$1"
    echo "Checking GPU availability for ${task_name}..."

    # Try nvidia-smi inside container
    if docker-compose -f "${PROJECT_DIR}/docker-compose.yml" exec -T nnunet nvidia-smi > /dev/null 2>&1; then
        echo "GPU is available."
        return 0
    fi

    echo "WARNING: GPU not detected. Restarting container..."
    docker-compose -f "${PROJECT_DIR}/docker-compose.yml" restart nnunet

    # Wait for container to be ready
    sleep 10

    # Check again
    if docker-compose -f "${PROJECT_DIR}/docker-compose.yml" exec -T nnunet nvidia-smi > /dev/null 2>&1; then
        echo "GPU is now available after restart."
        return 0
    fi

    echo "ERROR: GPU still not available after restart. Aborting."
    exit 1
}

# =============================================================================
# Step 1: Training
# =============================================================================
if [ "$SKIP_TRAIN" = false ]; then
    echo "=============================================="
    echo "[Step 1/5] Running training..."
    echo "=============================================="

    check_gpu "training"

    TRAIN_ARGS="--train --config ${CONFIG} --fold ${FOLD} --trainer ${TRAINER} --plans ${PLANS}"
    if [[ "$EPOCHS" != "" ]]; then
        TRAIN_ARGS="${TRAIN_ARGS} --epochs ${EPOCHS}"
    fi
    if [[ "$PRETRAINED_WEIGHTS" != "" ]]; then
        TRAIN_ARGS="${TRAIN_ARGS} --pretrained-weights ${PRETRAINED_WEIGHTS}"
    fi
    if [[ "$INITIAL_LR" != "" ]]; then
        TRAIN_ARGS="${TRAIN_ARGS} --initial-lr ${INITIAL_LR}"
    fi

    docker-compose -f "${PROJECT_DIR}/docker-compose.yml" exec -T nnunet \
        python /workspace/surface-nnunet-training-local.py ${TRAIN_ARGS}

    echo ""
    echo "Training completed."
else
    echo "=============================================="
    echo "[Step 1/5] Skipping training"
    echo "=============================================="
fi

# =============================================================================
# Step 2: Validation Inference with NPZ output
# =============================================================================
if [ "$SKIP_INFERENCE" = false ]; then
    echo ""
    echo "=============================================="
    echo "[Step 2/5] Running validation inference (NPZ/softmax)..."
    echo "=============================================="

    check_gpu "inference"

    # Get validation input directory from preprocessed data
    PREPROC_DIR="/workspace/nnunet_output/nnUNet_preprocessed/Dataset100_VesuviusSurface"
    VAL_INPUT="/workspace/nnunet_output/validation_input_fold_${FOLD}"
    VAL_OUTPUT="/workspace/nnunet_output/nnUNet_results/Dataset100_VesuviusSurface/${MODEL_NAME}/fold_${FOLD}/validation_npz"

    # Prepare validation input (symlink raw images for validation cases)
    docker-compose -f "${PROJECT_DIR}/docker-compose.yml" exec -T nnunet bash -c "
        mkdir -p ${VAL_INPUT}
        rm -f ${VAL_INPUT}/*

        # Get validation cases from splits_final.json
        python3 -c \"
import json
from pathlib import Path

splits_file = Path('${PREPROC_DIR}/splits_final.json')
with open(splits_file) as f:
    splits = json.load(f)
val_cases = splits[${FOLD}]['val']

raw_dir = Path('/workspace/nnunet_work/nnUNet_data/nnUNet_raw/Dataset100_VesuviusSurface/imagesTr')
val_input = Path('${VAL_INPUT}')

for case_id in val_cases:
    src = raw_dir / f'{case_id}_0000.tif'
    dst = val_input / f'{case_id}_0000.tif'
    if src.exists() and not dst.exists():
        dst.symlink_to(src)
        print(f'Linked: {case_id}')
    # Also copy JSON if exists
    src_json = raw_dir / f'{case_id}.json'
    dst_json = val_input / f'{case_id}.json'
    if src_json.exists() and not dst_json.exists():
        dst_json.symlink_to(src_json)
\"
        echo 'Validation input prepared.'
    "

    # Run inference with save_probabilities
    # Override nnUNet environment variables to use local paths
    docker-compose -f "${PROJECT_DIR}/docker-compose.yml" exec -T nnunet \
        env nnUNet_results=/workspace/nnunet_output/nnUNet_results \
            nnUNet_preprocessed=/workspace/nnunet_output/nnUNet_preprocessed \
        nnUNetv2_predict \
            -d 100 \
            -c ${CONFIG} \
            -f ${FOLD} \
            -i ${VAL_INPUT} \
            -o ${VAL_OUTPUT} \
            -tr ${INFERENCE_TRAINER} \
            -p ${PLANS} \
            --save_probabilities \
            -npp 1 -nps 1 \
            --verbose

    echo ""
    echo "Validation inference completed."
    echo "NPZ files saved to: ${VAL_OUTPUT}"
else
    echo ""
    echo "=============================================="
    echo "[Step 2/5] Skipping inference"
    echo "=============================================="
fi

# =============================================================================
# Step 3: Evaluation without post-processing
# =============================================================================
echo ""
echo "=============================================="
echo "[Step 3/5] Evaluating without post-processing..."
echo "=============================================="

GT_DIR="/workspace/nnunet_output/nnUNet_preprocessed/Dataset100_VesuviusSurface/gt_segmentations"
OUTPUT_NONE="/workspace/docs/results/eval_${TIMESTAMP}_${MODEL_NAME}_fold${FOLD}_none.csv"

docker-compose -f "${PROJECT_DIR}/docker-compose.yml" exec -T nnunet \
    python /workspace/evaluate_metrics.py \
        --npz-dir "${NPZ_DIR}" \
        --gt-dir "${GT_DIR}" \
        --postprocess none \
        --workers 24 \
        --output-csv "${OUTPUT_NONE}"

echo ""
echo "Evaluation (none) completed."

# =============================================================================
# Step 4: Evaluation with hysteresis post-processing
# =============================================================================
echo ""
echo "=============================================="
echo "[Step 4/5] Evaluating with hysteresis post-processing..."
echo "=============================================="

OUTPUT_HYST="/workspace/docs/results/eval_${TIMESTAMP}_${MODEL_NAME}_fold${FOLD}_hysteresis.csv"

docker-compose -f "${PROJECT_DIR}/docker-compose.yml" exec -T nnunet \
    python /workspace/evaluate_metrics.py \
        --npz-dir "${NPZ_DIR}" \
        --gt-dir "${GT_DIR}" \
        --postprocess hysteresis \
        --workers 24 \
        --output-csv "${OUTPUT_HYST}"

echo ""
echo "Evaluation (hysteresis) completed."

# =============================================================================
# Step 5: Evaluation with opening_closing post-processing (best)
# =============================================================================
echo ""
echo "=============================================="
echo "[Step 5/5] Evaluating with opening_closing post-processing..."
echo "=============================================="

OUTPUT_OC="/workspace/docs/results/eval_${TIMESTAMP}_${MODEL_NAME}_fold${FOLD}_opening_closing.csv"

docker-compose -f "${PROJECT_DIR}/docker-compose.yml" exec -T nnunet \
    python /workspace/evaluate_metrics.py \
        --npz-dir "${NPZ_DIR}" \
        --gt-dir "${GT_DIR}" \
        --postprocess opening_closing \
        --workers 24 \
        --output-csv "${OUTPUT_OC}"

echo ""
echo "Evaluation (opening_closing) completed."

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  None:           docs/results/eval_${TIMESTAMP}_${MODEL_NAME}_fold${FOLD}_none.csv"
echo "  Hysteresis:     docs/results/eval_${TIMESTAMP}_${MODEL_NAME}_fold${FOLD}_hysteresis.csv"
echo "  Opening/Closing: docs/results/eval_${TIMESTAMP}_${MODEL_NAME}_fold${FOLD}_opening_closing.csv"
echo "=============================================="

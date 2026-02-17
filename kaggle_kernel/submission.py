#!/usr/bin/env python3
"""
Vesuvius Surface Detection - Kaggle Submission Script
nnUNet 3d_lowres, fold_0 + fold_1 ensemble (2x T4 GPU), no TTA, opening+closing post-processing
"""

import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue

# =============================================================================
# INSTALL DEPENDENCIES
# =============================================================================

def install_packages():
    """Install required packages from offline source."""
    # Install without dependency check - Kaggle has most dependencies pre-installed
    pkg_dir = "/kaggle/input/vesvius-nnunet-packages"
    packages = [
        "imagecodecs-2026.1.14-cp311-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        "fft_conv_pytorch-1.2.0-py3-none-any.whl",
        "nnunetv2-2.6.4-py3-none-any.whl",
        "acvl_utils-0.2.5-py3-none-any.whl",
        "dynamic_network_architectures-0.4.3-py3-none-any.whl",
        "batchgenerators-0.25.1-py3-none-any.whl",
        "batchgeneratorsv2-0.3.0-py3-none-any.whl",
        "connected_components_3d-3.26.1-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl",
        "nibabel-5.3.2-py3-none-any.whl",
        "tifffile-2025.10.16-py3-none-any.whl",
    ]
    for pkg in packages:
        cmd = f"pip install --no-deps -q {pkg_dir}/{pkg}"
        subprocess.run(cmd, shell=True, check=True)

print("Installing packages...")
install_packages()

# =============================================================================
# IMPORTS (after installation)
# =============================================================================

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile
from scipy import ndimage as ndi

# =============================================================================
# CONFIGURATION
# =============================================================================

# Kaggle paths
INPUT_DIR = Path("/kaggle/input/vesuvius-challenge-surface-detection")
OUTPUT_DIR = Path("/kaggle/working")
MODEL_DATASET_DIR = Path("/kaggle/input/vesvius-model-weights")

# Model configuration
DATASET_ID = 100
DATASET_NAME = f"Dataset{DATASET_ID:03d}_VesuviusSurface"
TRAINER = "nnUNetTrainer"
PLANS = "nnUNetResEncUNetMPlans"
CONFIG = "3d_lowres"
FOLDS = ["0", "1"]  # Ensemble of fold_0 and fold_1

# GPU configuration for parallel inference
GPU_FOLD_MAP = {
    "0": 0,  # fold_0 -> GPU 0
    "1": 1,  # fold_1 -> GPU 1
}

# Inference settings
DISABLE_TTA = True  # Faster inference
USE_HYSTERESIS = True  # Hysteresis post-processing (validation: 0.5886 vs 0.5690 for argmax)

# =============================================================================
# POST-PROCESSING
# =============================================================================

def build_anisotropic_struct(z_radius: int, xy_radius: int) -> Optional[np.ndarray]:
    """Build anisotropic structuring element for morphological operations."""
    z, r = z_radius, xy_radius
    if z == 0 and r == 0:
        return None
    if z == 0 and r > 0:
        size = 2 * r + 1
        struct = np.zeros((1, size, size), dtype=bool)
        cy, cx = r, r
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx <= r * r:
                    struct[0, cy + dy, cx + dx] = True
        return struct
    if z > 0 and r == 0:
        struct = np.zeros((2 * z + 1, 1, 1), dtype=bool)
        struct[:, 0, 0] = True
        return struct
    depth = 2 * z + 1
    size = 2 * r + 1
    struct = np.zeros((depth, size, size), dtype=bool)
    cz, cy, cx = z, r, r
    for dz in range(-z, z + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx <= r * r:
                    struct[cz + dz, cy + dy, cx + dx] = True
    return struct


def postprocess_opening_closing(probs: np.ndarray) -> np.ndarray:
    """
    Post-processing with hysteresis + opening + closing.

    This method achieved the best validation score in our experiments:
    - Leaderboard: 0.6080 (vs 0.6010 for hysteresis-only)
    - TopoScore: 0.3423 (vs 0.3334 for hysteresis-only)

    Steps:
    1. 3D Hysteresis thresholding (t_high=0.85, t_low=0.30)
    2. Opening (remove small protrusions/noise)
    3. Anisotropic closing (fill small holes, z_radius=2, xy_radius=1)
    4. Dust removal (remove small connected components)

    Args:
        probs: Probability array with shape (num_classes, D, H, W)

    Returns:
        Binary prediction array with shape (D, H, W)
    """
    from skimage.morphology import remove_small_objects

    surface_prob = probs[1]  # Class 1 = surface

    # Step 1: 3D Hysteresis thresholding
    strong = surface_prob >= 0.85
    weak = surface_prob >= 0.30

    if not strong.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    # Step 2: Opening (remove small protrusions)
    struct_open = ndi.generate_binary_structure(3, 1)
    mask = ndi.binary_opening(mask, structure=struct_open)

    # Step 3: Closing (fill small holes)
    struct_close = build_anisotropic_struct(2, 1)
    mask = ndi.binary_closing(mask, structure=struct_close)

    # Step 4: Dust removal
    mask = remove_small_objects(mask.astype(bool), min_size=100)

    return mask.astype(np.uint8)

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

def setup_environment():
    """Set up nnUNet environment variables."""
    # Kaggle dataset extracts zip contents WITHOUT the parent folder name
    # Actual structure: /kaggle/input/vesvius-model-weights/fold_0/checkpoint_final.pth
    # nnUNet expects: {nnUNet_results}/Dataset100_VesuviusSurface/nnUNetTrainer__.../__config/fold_0/...
    # Solution: Create symlink at the trainer level

    # Use working directory for nnUNet_results so we can create symlinks
    results_dir = OUTPUT_DIR / "nnUNet_results"
    dataset_dir = results_dir / DATASET_NAME
    trainer_dir = dataset_dir / f"{TRAINER}__{PLANS}__{CONFIG}"

    # Create directory structure and symlink at trainer level
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if not trainer_dir.exists():
        trainer_dir.symlink_to(MODEL_DATASET_DIR)

    os.environ["nnUNet_raw"] = str(OUTPUT_DIR / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(OUTPUT_DIR / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(results_dir)

    print(f"nnUNet_results: {results_dir}")
    print(f"Symlink: {trainer_dir} -> {MODEL_DATASET_DIR}")

    # Verify all fold models exist
    model_dirs = {}
    for fold in FOLDS:
        model_dir = results_dir / DATASET_NAME / f"{TRAINER}__{PLANS}__{CONFIG}" / f"fold_{fold}"
        checkpoint = model_dir / "checkpoint_final.pth"

        if not checkpoint.exists():
            print(f"ERROR: Checkpoint not found: {checkpoint}")
            print(f"Available files in {MODEL_DATASET_DIR}:")
            for p in MODEL_DATASET_DIR.rglob("*"):
                print(f"  {p}")
            sys.exit(1)

        print(f"Model checkpoint (fold_{fold}): {checkpoint}")
        model_dirs[fold] = model_dir

    return model_dirs

# =============================================================================
# DATA PREPARATION
# =============================================================================

def create_spacing_json(output_path: Path, spacing: tuple = (1.0, 1.0, 1.0)):
    """Create JSON sidecar with spacing info for TIFF files."""
    json_data = {"spacing": list(spacing)}
    with open(output_path, "w") as f:
        json.dump(json_data, f)


def prepare_input_images(input_images: list, output_dir: Path) -> list:
    """Prepare input images for nnUNet inference."""
    output_dir.mkdir(parents=True, exist_ok=True)

    case_ids = []
    for img_path in input_images:
        case_id = img_path.stem
        dest_path = output_dir / f"{case_id}_0000.tif"
        json_path = output_dir / f"{case_id}_0000.json"

        # Symlink image
        if not dest_path.exists():
            dest_path.symlink_to(img_path.resolve())

        # Create JSON sidecar
        create_spacing_json(json_path)
        case_ids.append(case_id)

    return case_ids


def prepare_single_case(img_path: Path, case_input_dir: Path) -> str:
    """Prepare a single case for inference (for pipeline processing).

    Args:
        img_path: Path to input TIFF image
        case_input_dir: Directory to prepare input files

    Returns:
        case_id: The case identifier
    """
    case_id = img_path.stem

    # Clean and create input directory
    if case_input_dir.exists():
        shutil.rmtree(case_input_dir)
    case_input_dir.mkdir(parents=True)

    # Prepare input files
    dest_path = case_input_dir / f"{case_id}_0000.tif"
    json_path = case_input_dir / f"{case_id}_0000.json"
    dest_path.symlink_to(img_path.resolve())
    create_spacing_json(json_path)

    return case_id

# =============================================================================
# INFERENCE
# =============================================================================

def run_inference_single_case_fold(
    case_input_dir: Path,
    output_dir: Path,
    fold: str,
    gpu_id: int,
    result_queue: Queue
) -> None:
    """Run nnUNet inference on a single case with a specific fold and GPU.

    This function is designed to be run in a thread.
    Results are put into result_queue as (fold, success, error_msg).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} nnUNetv2_predict -d {DATASET_ID:03d} -c {CONFIG} -f {fold}"
    cmd += f" -i {case_input_dir} -o {output_dir} -p {PLANS} -tr {TRAINER}"
    cmd += " -npp 1 -nps 1 -step_size 0.3 --verbose"

    if DISABLE_TTA:
        cmd += " --disable_tta"

    if USE_HYSTERESIS:
        cmd += " --save_probabilities"

    print(f"[GPU {gpu_id}, fold_{fold}] Running: {cmd}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        error_msg = f"STDOUT:\n{result.stdout[-1000:]}\nSTDERR:\n{result.stderr[-1000:]}"
        result_queue.put((fold, False, error_msg))
    else:
        result_queue.put((fold, True, None))


def run_inference_parallel(case_input_dir: Path, output_dirs: dict) -> bool:
    """Run inference on all folds in parallel using multiple GPUs.

    Args:
        case_input_dir: Directory containing the input case
        output_dirs: Dict mapping fold -> output directory

    Returns:
        True if all folds succeeded, False otherwise
    """
    result_queue = Queue()
    threads = []

    for fold in FOLDS:
        gpu_id = GPU_FOLD_MAP[fold]
        output_dir = output_dirs[fold]

        t = threading.Thread(
            target=run_inference_single_case_fold,
            args=(case_input_dir, output_dir, fold, gpu_id, result_queue)
        )
        threads.append(t)
        t.start()

    # Wait for all threads to complete
    for t in threads:
        t.join()

    # Check results
    all_success = True
    while not result_queue.empty():
        fold, success, error_msg = result_queue.get()
        if not success:
            print(f"[fold_{fold}] Inference FAILED!")
            print(error_msg)
            all_success = False

    return all_success


def ensemble_and_convert_to_tiff(pred_dirs: dict, output_dir: Path, case_id: str) -> bool:
    """Ensemble predictions from multiple folds and convert to TIFF.

    Averages probabilities from all folds before applying hysteresis post-processing.

    Args:
        pred_dirs: Dict mapping fold -> prediction directory
        output_dir: Output directory for final TIFF
        case_id: Case identifier

    Returns:
        True if successful, False otherwise
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if USE_HYSTERESIS:
        # Load and average probabilities from all folds
        probs_list = []
        npz_paths = []

        for fold in FOLDS:
            npz_path = pred_dirs[fold] / f"{case_id}.npz"
            if not npz_path.exists():
                print(f"WARNING: NPZ not found for fold_{fold}: {npz_path}")
                return False

            data = np.load(npz_path)
            probs_list.append(data['probabilities'])
            npz_paths.append(npz_path)
            del data

        # Average probabilities
        probs_avg = np.mean(probs_list, axis=0)
        del probs_list

        # Opening+closing post-processing on averaged probabilities
        pred = postprocess_opening_closing(probs_avg)
        del probs_avg

        tifffile.imwrite(output_dir / f"{case_id}.tif", pred)
        del pred

        # Free disk space - delete all NPZ files
        for npz_path in npz_paths:
            npz_path.unlink()

        print(f"  Ensembled {len(FOLDS)} folds and converted: {case_id}")
        return True
    else:
        # Fallback: use argmax on averaged probabilities or majority vote on TIFFs
        preds_list = []
        tif_paths = []

        for fold in FOLDS:
            tif_path = pred_dirs[fold] / f"{case_id}.tif"
            if not tif_path.exists():
                print(f"WARNING: TIFF not found for fold_{fold}: {tif_path}")
                return False

            pred = tifffile.imread(str(tif_path))
            preds_list.append(pred)
            tif_paths.append(tif_path)

        # Majority vote (for binary: sum > len/2)
        pred_sum = np.sum(preds_list, axis=0)
        pred = (pred_sum > len(FOLDS) / 2).astype(np.uint8)
        del preds_list, pred_sum

        tifffile.imwrite(output_dir / f"{case_id}.tif", pred)
        del pred

        # Free disk space
        for tif_path in tif_paths:
            tif_path.unlink()

        print(f"  Ensembled {len(FOLDS)} folds (majority vote): {case_id}")
        return True

# =============================================================================
# SUBMISSION
# =============================================================================

def create_submission_zip(predictions_dir: Path, output_zip: Path) -> Path:
    """Create submission ZIP from TIFF predictions."""
    tif_files = sorted(predictions_dir.glob("*.tif"))

    if not tif_files:
        raise ValueError(f"No TIFF files found in {predictions_dir}")

    print(f"Creating submission ZIP with {len(tif_files)} files...")

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for tif_path in tif_files:
            zipf.write(tif_path, tif_path.name)

    zip_size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"Submission saved: {output_zip} ({zip_size_mb:.1f} MB)")

    return output_zip

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Vesuvius nnUNet Submission (2-GPU Ensemble)")
    print(f"Config: {CONFIG}, Folds: {', '.join(FOLDS)}")
    print(f"GPUs: {', '.join(f'fold_{f}->GPU{GPU_FOLD_MAP[f]}' for f in FOLDS)}")
    print(f"TTA: {'disabled' if DISABLE_TTA else 'enabled'}")
    print(f"Post-processing: {'opening+closing' if USE_HYSTERESIS else 'argmax'}")
    print(f"Ensemble: probability averaging")
    print("=" * 60)

    # Setup environment
    model_dirs = setup_environment()

    # Find test images
    test_images_dir = INPUT_DIR / "test_images"
    test_images = sorted(test_images_dir.glob("*.tif"))
    print(f"\nFound {len(test_images)} test images")

    if not test_images:
        print("ERROR: No test images found!")
        return 1

    # Prepare two sets of directories for double-buffering (avoid race condition)
    # While post-processing reads from pred_dirs_0, inference writes to pred_dirs_1
    pred_dirs_list = []
    for buf_idx in range(2):
        pred_dirs = {}
        for fold in FOLDS:
            pred_dirs[fold] = OUTPUT_DIR / f"predictions_fold{fold}_buf{buf_idx}"
            pred_dirs[fold].mkdir(parents=True, exist_ok=True)
        pred_dirs_list.append(pred_dirs)

    predictions_tiff_dir = OUTPUT_DIR / "predictions_tiff"
    predictions_tiff_dir.mkdir(parents=True, exist_ok=True)

    # Pipeline processing: overlap preparation and post-processing with inference
    # Use double-buffering for both input and output directories
    print(f"\nProcessing cases with pipelined inference on {len(FOLDS)} folds...")

    case_input_dirs = [OUTPUT_DIR / "case_input_0", OUTPUT_DIR / "case_input_1"]
    inference_failed = False

    # ThreadPoolExecutor for background tasks (preparation + post-processing)
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Prepare first case synchronously
        first_case_id = prepare_single_case(test_images[0], case_input_dirs[0])
        prepare_future: Optional[Future] = None
        # Track post-processing futures for each buffer separately
        postprocess_futures: list[Optional[tuple[str, Future]]] = [None, None]

        for i, img_path in enumerate(test_images):
            case_id = img_path.stem
            buf_idx = i % 2
            current_input_dir = case_input_dirs[buf_idx]
            current_pred_dirs = pred_dirs_list[buf_idx]

            # Wait for preparation if running in background (from previous iteration)
            if prepare_future is not None:
                prepare_future.result()
                prepare_future = None

            # Wait for post-processing using the SAME buffer to complete
            # This ensures we don't overwrite pred_dirs while they're being read
            if postprocess_futures[buf_idx] is not None:
                prev_case_id, future = postprocess_futures[buf_idx]
                result = future.result()
                if not result:
                    print(f"ERROR: Post-processing failed for {prev_case_id}!")
                    inference_failed = True
                    break
                postprocess_futures[buf_idx] = None

            print(f"\n[{i+1}/{len(test_images)}] Processing: {case_id}")

            # Start next case preparation in background (if there is a next case)
            if i + 1 < len(test_images):
                next_input_dir = case_input_dirs[(i + 1) % 2]
                prepare_future = executor.submit(
                    prepare_single_case, test_images[i + 1], next_input_dir
                )

            # Run inference on all folds in parallel (GPU-bound)
            success = run_inference_parallel(current_input_dir, current_pred_dirs)
            if not success:
                print(f"ERROR: Inference failed for {case_id}!")
                inference_failed = True
                break

            # Start post-processing in background (CPU-bound: NPZ load, ensemble, hysteresis)
            postprocess_futures[buf_idx] = (
                case_id,
                executor.submit(
                    ensemble_and_convert_to_tiff, current_pred_dirs, predictions_tiff_dir, case_id
                )
            )

        # Wait for all remaining post-processing to complete
        if not inference_failed:
            print("\nWaiting for post-processing to complete...")
            for buf_idx in range(2):
                if postprocess_futures[buf_idx] is not None:
                    prev_case_id, future = postprocess_futures[buf_idx]
                    result = future.result()
                    if not result:
                        print(f"ERROR: Post-processing failed for {prev_case_id}!")
                        inference_failed = True

    if inference_failed:
        return 1

    # Cleanup directories
    for input_dir in case_input_dirs:
        if input_dir.exists():
            shutil.rmtree(input_dir)
    for pred_dirs in pred_dirs_list:
        for fold_dir in pred_dirs.values():
            if fold_dir.exists():
                shutil.rmtree(fold_dir)

    # Create submission ZIP
    print(f"\nCreating submission ZIP...")
    submission_zip = OUTPUT_DIR / "submission.zip"
    create_submission_zip(predictions_tiff_dir, submission_zip)

    print("\n" + "=" * 60)
    print("SUBMISSION COMPLETE!")
    print(f"Output: {submission_zip}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

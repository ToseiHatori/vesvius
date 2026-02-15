#!/usr/bin/env python3
"""
Vesuvius Surface Detection - Kaggle Submission Script
nnUNet 3d_fullres, fold_all, Host Baseline (MedialSurfaceRecall), 1x T4 GPU, hysteresis post-processing
"""

import os
import subprocess
import sys

# =============================================================================
# INSTALL DEPENDENCIES
# =============================================================================

def install_packages():
    """Install required packages from offline source."""
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
# PATCH CHECKPOINT (convert custom trainer to standard nnUNetTrainer)
# =============================================================================

import torch

def patch_checkpoint_trainer(src_checkpoint, dst_checkpoint):
    """Patch checkpoint to use standard nnUNetTrainer instead of custom trainer."""
    print(f"Patching checkpoint: {src_checkpoint} -> {dst_checkpoint}")
    ckpt = torch.load(src_checkpoint, map_location='cpu', weights_only=False)
    original_trainer = ckpt.get('trainer_name', 'unknown')
    ckpt['trainer_name'] = 'nnUNetTrainer'
    torch.save(ckpt, dst_checkpoint)
    print(f"Patched trainer_name: {original_trainer} -> nnUNetTrainer")

# =============================================================================
# IMPORTS (after installation)
# =============================================================================

import json
import shutil
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

# Model configuration - Host Baseline fold_all with 3d_fullres
DATASET_ID = 100
DATASET_NAME = f"Dataset{DATASET_ID:03d}_VesuviusSurface"
ORIGINAL_TRAINER = "nnUNetTrainerMedialSurfaceRecall"  # Original custom trainer
TRAINER = "nnUNetTrainer"  # Standard trainer for inference
PLANS = "nnUNetHostBaselinePlans"
CONFIG = "3d_fullres"
FOLD = "all"

# Inference settings
DISABLE_TTA = True  # Faster inference
USE_HYSTERESIS = True  # Hysteresis post-processing

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


def postprocess_hysteresis(
    probs: np.ndarray,
    t_low: float = 0.3,
    t_high: float = 0.85,
    z_radius: int = 1,
    xy_radius: int = 0,
    dust_min_size: int = 100
) -> np.ndarray:
    """
    Post-processing with hysteresis thresholding.

    Steps:
    1. 3D Hysteresis thresholding: propagate from high confidence to low
    2. Anisotropic morphological closing (Z direction only)
    3. Dust removal (remove small connected components)

    Args:
        probs: Probability array with shape (num_classes, D, H, W)
        t_low: Low threshold for hysteresis (default: 0.3)
        t_high: High threshold for hysteresis (default: 0.85)
        z_radius: Radius in Z direction for closing (default: 1)
        xy_radius: Radius in XY plane for closing (default: 0)
        dust_min_size: Minimum size for connected components (default: 100)

    Returns:
        Binary prediction array with shape (D, H, W)
    """
    from skimage.morphology import remove_small_objects

    surface_prob = probs[1]  # Class 1 = surface

    # Step 1: 3D Hysteresis thresholding
    strong = surface_prob >= t_high
    weak = surface_prob >= t_low

    if not strong.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    # Step 2: Anisotropic closing
    if z_radius > 0 or xy_radius > 0:
        struct_close = build_anisotropic_struct(z_radius, xy_radius)
        if struct_close is not None:
            mask = ndi.binary_closing(mask, structure=struct_close)

    # Step 3: Dust removal
    if dust_min_size > 0:
        mask = remove_small_objects(mask.astype(bool), min_size=dust_min_size)

    return mask.astype(np.uint8)

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

def setup_environment():
    """Set up nnUNet environment variables and patch custom trainer checkpoint."""
    results_dir = OUTPUT_DIR / "nnUNet_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    os.environ["nnUNet_raw"] = str(OUTPUT_DIR / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(OUTPUT_DIR / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(results_dir)

    print(f"nnUNet_results: {results_dir}")

    # Source: original model with custom trainer
    src_trainer_dir = MODEL_DATASET_DIR / f"{ORIGINAL_TRAINER}__{PLANS}__{CONFIG}"
    src_checkpoint = src_trainer_dir / f"fold_{FOLD}" / "checkpoint_final.pth"

    if not src_checkpoint.exists():
        print(f"ERROR: Source checkpoint not found: {src_checkpoint}")
        print(f"Available files in {MODEL_DATASET_DIR}:")
        for p in MODEL_DATASET_DIR.rglob("*"):
            print(f"  {p}")
        sys.exit(1)

    # Destination: patched model with standard trainer
    dst_dataset_dir = results_dir / DATASET_NAME
    dst_trainer_dir = dst_dataset_dir / f"{TRAINER}__{PLANS}__{CONFIG}"
    dst_fold_dir = dst_trainer_dir / f"fold_{FOLD}"
    dst_fold_dir.mkdir(parents=True, exist_ok=True)

    # Copy plans.json and dataset.json
    shutil.copy(src_trainer_dir / "plans.json", dst_trainer_dir / "plans.json")
    shutil.copy(src_trainer_dir / "dataset.json", dst_trainer_dir / "dataset.json")

    # Patch and save checkpoint with standard trainer name
    dst_checkpoint = dst_fold_dir / "checkpoint_final.pth"
    if not dst_checkpoint.exists():
        patch_checkpoint_trainer(src_checkpoint, dst_checkpoint)

    print(f"Patched model directory: {dst_trainer_dir}")
    print(f"Model checkpoint: {dst_checkpoint}")

    return dst_trainer_dir / f"fold_{FOLD}"

# =============================================================================
# DATA PREPARATION
# =============================================================================

def create_spacing_json(output_path: Path, spacing: tuple = (1.0, 1.0, 1.0)):
    """Create JSON sidecar with spacing info for TIFF files."""
    json_data = {"spacing": list(spacing)}
    with open(output_path, "w") as f:
        json.dump(json_data, f)

# =============================================================================
# INFERENCE
# =============================================================================

def run_inference(case_input_dir: Path, output_dir: Path) -> bool:
    """Run nnUNet inference on a single case.

    Args:
        case_input_dir: Directory containing the input case
        output_dir: Output directory for predictions

    Returns:
        True if successful, False otherwise
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = f"nnUNetv2_predict -d {DATASET_ID:03d} -c {CONFIG} -f {FOLD}"
    cmd += f" -i {case_input_dir} -o {output_dir} -p {PLANS} -tr {TRAINER}"
    cmd += " -npp 2 -nps 2 --verbose"

    if DISABLE_TTA:
        cmd += " --disable_tta"

    if USE_HYSTERESIS:
        cmd += " --save_probabilities"

    print(f"Running: {cmd}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout[-2000:]}")
        print(f"STDERR:\n{result.stderr[-2000:]}")
        return False

    return True


def convert_to_tiff(pred_dir: Path, output_dir: Path, case_id: str) -> bool:
    """Convert prediction to TIFF with post-processing.

    Args:
        pred_dir: Directory containing nnUNet predictions
        output_dir: Output directory for final TIFF
        case_id: Case identifier

    Returns:
        True if successful, False otherwise
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if USE_HYSTERESIS:
        npz_path = pred_dir / f"{case_id}.npz"
        if not npz_path.exists():
            print(f"WARNING: NPZ not found: {npz_path}")
            return False

        data = np.load(npz_path)
        probs = data['probabilities']
        del data

        # Hysteresis post-processing
        pred = postprocess_hysteresis(probs)
        del probs

        tifffile.imwrite(output_dir / f"{case_id}.tif", pred)
        del pred

        # Free disk space
        npz_path.unlink()

        print(f"  Converted with hysteresis: {case_id}")
        return True
    else:
        # Fallback: just copy the TIFF
        tif_path = pred_dir / f"{case_id}.tif"
        if not tif_path.exists():
            print(f"WARNING: TIFF not found: {tif_path}")
            return False

        shutil.copy(tif_path, output_dir / f"{case_id}.tif")
        tif_path.unlink()

        print(f"  Copied: {case_id}")
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
    print("Vesuvius nnUNet Submission (fold_all, single model)")
    print(f"Config: {CONFIG}, Fold: {FOLD}")
    print(f"TTA: {'disabled' if DISABLE_TTA else 'enabled'}")
    print(f"Post-processing: {'hysteresis' if USE_HYSTERESIS else 'argmax'}")
    print("=" * 60)

    # Setup environment
    model_dir = setup_environment()

    # Find test images
    test_images_dir = INPUT_DIR / "test_images"
    test_images = sorted(test_images_dir.glob("*.tif"))
    print(f"\nFound {len(test_images)} test images")

    if not test_images:
        print("ERROR: No test images found!")
        return 1

    # Prepare directories
    pred_dir = OUTPUT_DIR / "predictions_raw"
    pred_dir.mkdir(parents=True, exist_ok=True)

    predictions_tiff_dir = OUTPUT_DIR / "predictions_tiff"
    predictions_tiff_dir.mkdir(parents=True, exist_ok=True)

    # Process each case
    print(f"\nProcessing cases...")
    for i, img_path in enumerate(test_images):
        case_id = img_path.stem
        print(f"\n[{i+1}/{len(test_images)}] Processing: {case_id}")

        # Create temp input directory for this case
        case_input_dir = OUTPUT_DIR / "case_input"
        if case_input_dir.exists():
            shutil.rmtree(case_input_dir)
        case_input_dir.mkdir(parents=True)

        # Prepare single input image
        dest_path = case_input_dir / f"{case_id}_0000.tif"
        json_path = case_input_dir / f"{case_id}_0000.json"
        dest_path.symlink_to(img_path.resolve())
        create_spacing_json(json_path)

        # Run inference
        success = run_inference(case_input_dir, pred_dir)
        if not success:
            print(f"ERROR: Inference failed for {case_id}!")
            return 1

        # Convert to TIFF with post-processing
        convert_to_tiff(pred_dir, predictions_tiff_dir, case_id)

        # Cleanup temp input
        shutil.rmtree(case_input_dir)

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

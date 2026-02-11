#!/usr/bin/env python3
"""
Vesuvius Surface Detection - Kaggle Submission Script
nnUNet 3d_lowres, fold_0, 1000 epochs, no TTA, argmax post-processing
"""

import os
import subprocess
import sys

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
FOLD = "0"

# Inference settings
DISABLE_TTA = True  # Faster inference
USE_HYSTERESIS = False  # Simple argmax

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

def setup_environment():
    """Set up nnUNet environment variables."""
    # Note: Kaggle dataset doesn't include the nnUNet_results prefix
    results_dir = MODEL_DATASET_DIR

    os.environ["nnUNet_raw"] = str(OUTPUT_DIR / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(OUTPUT_DIR / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(results_dir)

    print(f"nnUNet_results: {results_dir}")

    # Verify model exists
    model_dir = results_dir / DATASET_NAME / f"{TRAINER}__{PLANS}__{CONFIG}" / f"fold_{FOLD}"
    checkpoint = model_dir / "checkpoint_final.pth"

    if not checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        print(f"Available files in {MODEL_DATASET_DIR}:")
        for p in MODEL_DATASET_DIR.rglob("*"):
            print(f"  {p}")
        sys.exit(1)

    print(f"Model checkpoint: {checkpoint}")
    return model_dir

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

# =============================================================================
# INFERENCE
# =============================================================================

def run_inference(input_dir: Path, output_dir: Path) -> bool:
    """Run nnUNet inference."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = f"nnUNetv2_predict -d {DATASET_ID:03d} -c {CONFIG} -f {FOLD}"
    cmd += f" -i {input_dir} -o {output_dir} -p {PLANS} -tr {TRAINER}"
    cmd += " -npp 1 -nps 1 --verbose"

    if DISABLE_TTA:
        cmd += " --disable_tta"

    print(f"Running: {cmd}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Inference FAILED!")
        print(f"STDOUT:\n{result.stdout[-2000:]}")
        print(f"STDERR:\n{result.stderr[-2000:]}")
        return False

    print("Inference complete!")
    return True


def convert_predictions_to_tiff(pred_dir: Path, output_dir: Path):
    """Convert nnUNet predictions to TIFF format (argmax only)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    tif_files = list(pred_dir.glob("*.tif"))

    if tif_files:
        print(f"Copying {len(tif_files)} TIFF files...")
        for tif_path in tif_files:
            case_id = tif_path.stem
            pred = tifffile.imread(str(tif_path)).astype(np.uint8)
            tifffile.imwrite(output_dir / f"{case_id}.tif", pred)
    else:
        print(f"WARNING: No prediction files found in {pred_dir}")

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
    print("Vesuvius nnUNet Submission")
    print(f"Config: {CONFIG}, Fold: {FOLD}")
    print(f"TTA: {'disabled' if DISABLE_TTA else 'enabled'}")
    print(f"Post-processing: {'argmax' if not USE_HYSTERESIS else 'hysteresis'}")
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
    test_input_dir = OUTPUT_DIR / "test_input"
    predictions_dir = OUTPUT_DIR / "predictions"
    predictions_tiff_dir = OUTPUT_DIR / "predictions_tiff"

    # Prepare input images
    print(f"\nPreparing input images...")
    case_ids = prepare_input_images(test_images, test_input_dir)
    print(f"Prepared {len(case_ids)} cases")

    # Run inference
    print(f"\nRunning inference...")
    success = run_inference(test_input_dir, predictions_dir)

    if not success:
        print("ERROR: Inference failed!")
        return 1

    # Convert predictions to TIFF
    print(f"\nConverting predictions to TIFF...")
    convert_predictions_to_tiff(predictions_dir, predictions_tiff_dir)

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

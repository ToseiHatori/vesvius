#!/usr/bin/env python3
"""
TTA Speedup Factor Measurement Script
Measures the actual speedup factor between TTA and non-TTA inference.

Usage: Run as Kaggle notebook with GPU enabled.
"""

import os
import subprocess
import sys
import time

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
# IMPORTS (after installation)
# =============================================================================

from pathlib import Path

import numpy as np
import torch
import tifffile

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

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
CONFIG = "3d_lowres"  # Use one config for measurement
FOLD = "0"  # Use one fold for measurement

# Inference settings
TILE_STEP_SIZE = 0.3

# Measurement settings
NUM_SAMPLES = 3  # Number of samples to measure (keep small to save time)
NUM_WARMUP = 1   # Warmup runs (not counted)

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

def setup_environment():
    """Set up nnUNet environment variables."""
    results_dir = OUTPUT_DIR / "nnUNet_results"
    dataset_dir = results_dir / DATASET_NAME

    os.environ["nnUNet_raw"] = str(OUTPUT_DIR / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(OUTPUT_DIR / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(results_dir)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    trainer_dir = dataset_dir / f"{TRAINER}__{PLANS}__{CONFIG}"

    if not trainer_dir.exists():
        trainer_dir.symlink_to(MODEL_DATASET_DIR / CONFIG)

    print(f"Model: {trainer_dir}")
    return trainer_dir

# =============================================================================
# INFERENCE
# =============================================================================

class TiffReader:
    @staticmethod
    def read_images(image_fnames: list) -> tuple:
        images = []
        for fname in image_fnames:
            img = tifffile.imread(str(fname))
            images.append(img)
        image_array = np.stack(images, axis=0).astype(np.float32)
        properties = {'spacing': (1.0, 1.0, 1.0)}
        return image_array, properties


def create_predictor(use_tta: bool) -> nnUNetPredictor:
    """Create predictor with TTA on/off."""
    predictor = nnUNetPredictor(
        tile_step_size=TILE_STEP_SIZE,
        use_gaussian=True,
        use_mirroring=use_tta,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    return predictor


def load_model(predictor: nnUNetPredictor) -> None:
    """Load model into predictor."""
    model_folder = Path(os.environ["nnUNet_results"]) / DATASET_NAME / f"{TRAINER}__{PLANS}__{CONFIG}"
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=(int(FOLD),),
        checkpoint_name='checkpoint_final.pth',
    )


def predict_single_case(predictor: nnUNetPredictor, img_path: Path) -> np.ndarray:
    """Run inference on a single case."""
    image_array, properties = TiffReader.read_images([img_path])
    _seg, probs = predictor.predict_single_npy_array(
        image_array,
        properties,
        None,
        None,
        True
    )
    return probs


def measure_inference_time(predictor: nnUNetPredictor, images: list, label: str) -> list:
    """Measure inference time for multiple images."""
    times = []

    # Warmup
    print(f"\n[{label}] Warmup ({NUM_WARMUP} runs)...")
    for i in range(NUM_WARMUP):
        img = images[i % len(images)]
        predict_single_case(predictor, img)
        print(f"  Warmup {i+1}/{NUM_WARMUP} done")

    # Measurement
    print(f"[{label}] Measuring ({NUM_SAMPLES} runs)...")
    for i, img in enumerate(images[:NUM_SAMPLES]):
        torch.cuda.synchronize()
        start = time.perf_counter()

        predict_single_case(predictor, img)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Sample {i+1}/{NUM_SAMPLES}: {elapsed:.2f}s ({img.name})")

    return times

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("TTA Speedup Factor Measurement")
    print(f"Config: {CONFIG}, Fold: {FOLD}")
    print(f"Tile step size: {TILE_STEP_SIZE}")
    print(f"Samples: {NUM_SAMPLES}, Warmup: {NUM_WARMUP}")
    print("=" * 60)

    # Setup
    setup_environment()

    # Find train images (use train instead of test for measurement)
    train_images_dir = INPUT_DIR / "train_images"
    train_images = sorted(train_images_dir.glob("*.tif"))
    print(f"\nFound {len(train_images)} train images")

    if len(train_images) < NUM_SAMPLES + NUM_WARMUP:
        print(f"ERROR: Need at least {NUM_SAMPLES + NUM_WARMUP} images")
        return 1

    # Select samples
    sample_images = train_images[:NUM_SAMPLES + NUM_WARMUP]
    print(f"Using images: {[img.name for img in sample_images]}")

    # =========================================================================
    # Measure TTA OFF (faster, do first)
    # =========================================================================
    print("\n" + "=" * 60)
    print("MEASURING: TTA OFF (no mirroring)")
    print("=" * 60)

    predictor_no_tta = create_predictor(use_tta=False)
    load_model(predictor_no_tta)

    times_no_tta = measure_inference_time(predictor_no_tta, sample_images, "TTA OFF")

    del predictor_no_tta
    torch.cuda.empty_cache()

    # =========================================================================
    # Measure TTA ON (slower)
    # =========================================================================
    print("\n" + "=" * 60)
    print("MEASURING: TTA ON (with mirroring)")
    print("=" * 60)

    predictor_tta = create_predictor(use_tta=True)
    load_model(predictor_tta)

    times_tta = measure_inference_time(predictor_tta, sample_images, "TTA ON")

    del predictor_tta
    torch.cuda.empty_cache()

    # =========================================================================
    # Results
    # =========================================================================
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    avg_no_tta = np.mean(times_no_tta)
    avg_tta = np.mean(times_tta)
    speedup = avg_tta / avg_no_tta

    print(f"\nTTA OFF times: {times_no_tta}")
    print(f"TTA ON times:  {times_tta}")
    print(f"\nTTA OFF average: {avg_no_tta:.2f}s")
    print(f"TTA ON average:  {avg_tta:.2f}s")
    print(f"\n>>> TTA SPEEDUP FACTOR: {speedup:.2f}x <<<")
    print(f"\nRecommendation: Set TTA_SPEEDUP_FACTOR = {speedup:.1f} in submission.py")

    # Per-sample comparison
    print("\nPer-sample comparison:")
    print(f"{'Sample':<20} {'TTA OFF':>10} {'TTA ON':>10} {'Ratio':>10}")
    print("-" * 52)
    for i in range(NUM_SAMPLES):
        ratio = times_tta[i] / times_no_tta[i]
        print(f"{sample_images[i].name:<20} {times_no_tta[i]:>10.2f}s {times_tta[i]:>10.2f}s {ratio:>10.2f}x")

    print("\n" + "=" * 60)
    print("MEASUREMENT COMPLETE")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())

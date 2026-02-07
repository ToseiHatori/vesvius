#!/usr/bin/env python3
"""
nnUNet Preprocessing for Vesuvius Surface Detection (Docker/Kaggle Version)

This script runs inside the Kaggle Docker container with paths matching Kaggle environment.
"""

import os
import json
import shutil
import subprocess
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Tuple, List, Literal, Union

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

Epochs = Literal[1, 5, 10, 20, 50, 100, 250, 500, 750, 1000, 2000, 4000, 8000]

# =============================================================================
# CONFIGURATION - KAGGLE-COMPATIBLE PATHS (mounted via Docker)
# =============================================================================

# These paths match Kaggle environment and are mounted from host
INPUT_DIR = Path("/kaggle/input/vesuvius-challenge-surface-detection")
WORKING_DIR = Path("/kaggle/temp")
OUTPUT_DIR = Path("/kaggle/working")

# nnUNet directory structure
NNUNET_BASE = WORKING_DIR / "nnUNet_data"
NNUNET_RAW = NNUNET_BASE / "nnUNet_raw"
NNUNET_PREPROCESSED = OUTPUT_DIR / "nnUNet_preprocessed"
NNUNET_RESULTS = OUTPUT_DIR / "nnUNet_results"

# Dataset configuration
DATASET_ID = 100
DATASET_NAME = f"Dataset{DATASET_ID:03d}_VesuviusSurface"

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

FOLD: Union[int, str] = "all"
CONFIGURATION = "3d_fullres"
PLANNER = "nnUNetPlannerResEncM"
PLANS_NAME = "nnUNetResEncUNetMPlans"
NUM_WORKERS = os.cpu_count() or 4
EPOCHS: Optional[Epochs] = None
TRAINER: str = "nnUNetTrainer"
COMMAND_TIMEOUT: Optional[int] = None


def _get_gpu_count() -> int:
    """Get number of available CUDA GPUs."""
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        return 0


NUM_GPUS: int = _get_gpu_count()


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

def setup_environment():
    """Set up nnUNet environment variables and directories."""
    for d in [NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS, OUTPUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    os.environ["nnUNet_raw"] = str(NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    os.environ["nnUNet_results"] = str(NNUNET_RESULTS)
    os.environ["nnUNet_compile"] = "true"
    os.environ["nnUNet_USE_BLOSC2"] = "1"

    print(f"nnUNet_raw: {NNUNET_RAW}")
    print(f"nnUNet_preprocessed: {NNUNET_PREPROCESSED}")
    print(f"nnUNet_results: {NNUNET_RESULTS}")
    print(f"nnUNet_USE_BLOSC2: 1 (blosc2 compression enabled)")
    print(f"NUM_WORKERS: {NUM_WORKERS}")
    print(f"NUM_GPUS: {NUM_GPUS}")


# =============================================================================
# DATA UTILITIES
# =============================================================================

def create_spacing_json(output_path: Path, shape: tuple, spacing: tuple = (1.0, 1.0, 1.0)):
    """Create JSON sidecar with spacing info for TIFF files."""
    json_data = {"spacing": list(spacing)}
    with open(output_path, "w") as f:
        json.dump(json_data, f)


def create_dataset_json(output_dir: Path, num_training: int, file_ending: str = ".tif") -> dict:
    """Create dataset.json with ignore label support and 3D TIFF reader."""
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "surface": 1, "ignore": 2},
        "numTraining": num_training,
        "file_ending": file_ending,
        "overwrite_image_reader_writer": "SimpleTiffIO"
    }

    json_path = output_dir / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)

    print(f"Created {json_path}")
    print(f"  - {num_training} training cases")
    print(f"  - Labels: background(0), surface(1), ignore(2)")
    print(f"  - Reader: SimpleTiffIO (3D TIFF)")

    return dataset_json


# =============================================================================
# DATASET PREPARATION
# =============================================================================

def prepare_single_case(
    src_path: Path,
    dest_path: Path,
    json_path: Path,
    use_symlinks: bool = True
) -> bool:
    """Prepare a single TIFF file for nnUNet."""
    try:
        import tifffile
        with tifffile.TiffFile(src_path) as tif:
            shape = tif.pages[0].shape if len(tif.pages) == 1 else (len(tif.pages), *tif.pages[0].shape)

        if use_symlinks:
            if not dest_path.exists():
                dest_path.symlink_to(src_path.resolve())
        else:
            shutil.copy2(src_path, dest_path)

        create_spacing_json(json_path, shape)
        return True

    except Exception as e:
        print(f"Error processing {src_path.name}: {e}")
        return False


def _prepare_training_case(
    img_path: Path,
    train_labels_dir: Path,
    images_dir: Path,
    labels_dir: Path,
    use_symlinks: bool
) -> bool:
    """Worker function for parallel dataset preparation."""
    case_id = img_path.stem
    label_path = train_labels_dir / img_path.name

    if not label_path.exists():
        return False

    img_ok = prepare_single_case(
        img_path,
        images_dir / f"{case_id}_0000.tif",
        images_dir / f"{case_id}_0000.json",
        use_symlinks
    )

    label_ok = prepare_single_case(
        label_path,
        labels_dir / f"{case_id}.tif",
        labels_dir / f"{case_id}.json",
        use_symlinks
    )

    return img_ok and label_ok


def prepare_dataset(input_dir: Path, max_cases: Optional[int] = None, use_symlinks: bool = True):
    """Convert competition data to nnUNet format."""
    from tqdm.auto import tqdm

    dataset_dir = NNUNET_RAW / DATASET_NAME
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    train_images_dir = input_dir / "train_images"
    train_labels_dir = input_dir / "train_labels"

    if not train_images_dir.exists():
        print(f"ERROR: {train_images_dir} not found!")
        return None

    image_files = sorted(train_images_dir.glob("*.tif"))
    if max_cases:
        image_files = image_files[:max_cases]

    print(f"Found {len(image_files)} training cases")
    print(f"Using {'symlinks' if use_symlinks else 'copy'}")
    print(f"Processing with {NUM_WORKERS} workers...")

    worker = partial(
        _prepare_training_case,
        train_labels_dir=train_labels_dir,
        images_dir=images_dir,
        labels_dir=labels_dir,
        use_symlinks=use_symlinks
    )

    with Pool(NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(worker, image_files),
            total=len(image_files),
            desc="Preparing dataset"
        ))

    num_converted = sum(results)
    create_dataset_json(dataset_dir, num_converted, file_ending=".tif")

    print(f"\nDataset prepared: {num_converted} cases")
    print(f"Location: {dataset_dir}")

    return dataset_dir


# =============================================================================
# NNUNET COMMANDS
# =============================================================================

def _run_command(
    cmd: str,
    name: str = "Command",
    tail_lines: int = 20,
    timeout: Optional[int] = COMMAND_TIMEOUT
) -> bool:
    """
    Execute shell command and handle output parsing.

    This matches the original nnUNet notebook implementation.
    """
    print(f"Running: {cmd}")
    if timeout:
        print(f"Timeout: {timeout}s ({timeout/3600:.1f}h)")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
    except subprocess.TimeoutExpired:
        print(f"{name} TIMEOUT after {timeout}s!")
        return False

    if result.returncode != 0:
        print(f"{name} FAILED!")
        print(f"STDERR:\n{result.stderr[-3000:]}")
        return False

    print(f"{name} complete!")
    if result.stdout.strip():
        lines = result.stdout.strip().split('\n')
        print('\n'.join(lines[-tail_lines:]))

    return True


def run_preprocessing(
    dataset_id: int = DATASET_ID,
    planner: str = PLANNER,
    num_workers: int = NUM_WORKERS,
    configurations: Optional[List[str]] = None,
    timeout: Optional[int] = COMMAND_TIMEOUT
) -> bool:
    """Run nnUNet preprocessing."""
    if configurations is None:
        configurations = [CONFIGURATION]

    cmd = f"nnUNetv2_plan_and_preprocess -d {dataset_id:03d} -np {num_workers}"
    cmd += f" -pl {planner}"
    cmd += f" -c {' '.join(configurations)}"

    return _run_command(cmd, "Preprocessing", timeout=timeout)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("VESUVIUS PREPROCESSING - 3d_fullres (Docker/Kaggle)")
    print("=" * 60)

    # Check if input data exists
    if not INPUT_DIR.exists():
        print(f"\nERROR: Input data not found at {INPUT_DIR}")
        print("Make sure the data is mounted correctly.")
        return False

    train_images = INPUT_DIR / "train_images"
    if not train_images.exists():
        print(f"\nERROR: train_images not found at {train_images}")
        return False

    num_images = len(list(train_images.glob("*.tif")))
    print(f"\nFound {num_images} training images")

    # Step 1: Setup environment
    print("\n[0/2] Setting up environment...")
    setup_environment()

    # Step 2: Prepare dataset
    print("\n[1/2] Preparing dataset...")
    prepare_dataset(INPUT_DIR)

    # Step 3: Run preprocessing
    print("\n[2/2] Running nnUNet preprocessing...")
    success = run_preprocessing(
        planner=PLANNER,
        configurations=[CONFIGURATION]
    )

    print("\n" + "=" * 60)
    if success:
        print("DONE!")
        output_dir = NNUNET_PREPROCESSED / DATASET_NAME
        if output_dir.exists():
            print(f"\nOutput: {output_dir}")
            items = sorted(output_dir.iterdir())[:10]
            for item in items:
                print(f"  {item.name}")
    else:
        print("FAILED!")
    print("=" * 60)

    return success


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)

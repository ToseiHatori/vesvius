#!/usr/bin/env python3
"""
Vesuvius nnUNet Training & Inference - Local Version

This script runs nnUNet training and inference locally or in Docker.
Based on surface-nnunet-training-inference-with-2xt4.ipynb.

Usage:
    # Training only
    python surface-nnunet-training-local.py --train --epochs 50

    # Inference only (using existing model)
    python surface-nnunet-training-local.py --inference --model-path /path/to/checkpoint.pth

    # Full pipeline
    python surface-nnunet-training-local.py --train --inference --epochs 100

    # With custom configuration
    python surface-nnunet-training-local.py --train --config 3d_fullres --fold all --gpus 2
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Tuple, List, Literal, Union

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

Epochs = Literal[1, 5, 10, 20, 50, 100, 250, 500, 750, 1000, 2000, 4000, 8000]

# =============================================================================
# PATH CONFIGURATION - LOCAL
# =============================================================================

PROJECT_DIR = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_DIR / "data"
DATASET_DIR = DATA_DIR / "vesuvius-challenge-surface-detection"

# nnUNet directories
NNUNET_OUTPUT = PROJECT_DIR / "nnunet_output"
NNUNET_WORK = PROJECT_DIR / "nnunet_work"

NNUNET_RAW = NNUNET_WORK / "nnUNet_data" / "nnUNet_raw"
NNUNET_PREPROCESSED = NNUNET_OUTPUT / "nnUNet_preprocessed"
NNUNET_RESULTS = NNUNET_OUTPUT / "nnUNet_results"

# Pre-prepared preprocessed data (if available)
PREPARED_PREPROCESSED_PATH = NNUNET_PREPROCESSED

# Dataset configuration
DATASET_ID = 100
DATASET_NAME = f"Dataset{DATASET_ID:03d}_VesuviusSurface"


def get_num_workers() -> int:
    """Get number of CPU workers."""
    return os.cpu_count() or 4


def get_gpu_count() -> int:
    """Get number of available CUDA GPUs."""
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        # Try nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"], capture_output=True, text=True
            )
            if result.returncode == 0:
                return len([l for l in result.stdout.strip().split('\n') if l.startswith('GPU')])
        except FileNotFoundError:
            pass
        return 0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def setup_environment():
    """Set up nnUNet environment variables and directories."""
    for d in [NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS]:
        d.mkdir(parents=True, exist_ok=True)

    os.environ["nnUNet_raw"] = str(NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    os.environ["nnUNet_results"] = str(NNUNET_RESULTS)
    os.environ["nnUNet_compile"] = "true"
    os.environ["nnUNet_USE_BLOSC2"] = "1"

    print(f"nnUNet_raw: {NNUNET_RAW}")
    print(f"nnUNet_preprocessed: {NNUNET_PREPROCESSED}")
    print(f"nnUNet_results: {NNUNET_RESULTS}")
    print(f"NUM_WORKERS: {get_num_workers()}")
    print(f"NUM_GPUS: {get_gpu_count()}")


def get_trainer_name(epochs: Optional[int], trainer: str = "nnUNetTrainer") -> str:
    """Get trainer class name based on epochs."""
    if trainer != "nnUNetTrainer":
        return trainer
    if epochs is None or epochs == 1000:
        return "nnUNetTrainer"
    elif epochs == 1:
        return "nnUNetTrainer_1epoch"
    else:
        return f"nnUNetTrainer_{epochs}epochs"


def get_training_output_dir(
    epochs: Optional[int] = None,
    plans: str = "nnUNetResEncUNetMPlans",
    config: str = "3d_fullres",
    fold: Union[int, str] = "all",
    trainer: str = "nnUNetTrainer"
) -> Path:
    """Get the training output directory path."""
    trainer_name = get_trainer_name(epochs, trainer)
    return NNUNET_RESULTS / DATASET_NAME / f"{trainer_name}__{plans}__{config}" / f"fold_{fold}"


def run_command(
    cmd: str,
    name: str = "Command",
    timeout: Optional[int] = None
) -> bool:
    """Execute shell command and handle output."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Command: {cmd}")
    print('='*60)

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            timeout=timeout,
            env={**os.environ}
        )
        success = result.returncode == 0
        if success:
            print(f"\n{name} completed successfully!")
        else:
            print(f"\n{name} FAILED with exit code {result.returncode}")
        return success
    except subprocess.TimeoutExpired:
        print(f"\n{name} TIMEOUT after {timeout}s!")
        return False
    except Exception as e:
        print(f"\n{name} ERROR: {e}")
        return False


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def run_training(
    dataset_id: int = DATASET_ID,
    config: str = "3d_fullres",
    fold: Union[int, str] = "all",
    plans: str = "nnUNetResEncUNetMPlans",
    epochs: Optional[int] = None,
    trainer: str = "nnUNetTrainer",
    pretrained_weights: Optional[Path] = None,
    continue_training: bool = False,
    num_gpus: int = 1,
    timeout: Optional[int] = None
) -> bool:
    """Run nnUNet training."""
    trainer_name = get_trainer_name(epochs, trainer)

    cmd = f"nnUNetv2_train {dataset_id:03d} {config} {fold} -p {plans} -tr {trainer_name}"

    if pretrained_weights:
        cmd += f" -pretrained_weights {pretrained_weights}"
    if continue_training:
        cmd += " --c"
    if num_gpus > 1:
        cmd += f" -num_gpus {num_gpus}"

    epochs_str = epochs if epochs else 1000
    return run_command(cmd, f"Training ({epochs_str} epochs, {num_gpus} GPUs)", timeout=timeout)


def run_inference(
    input_dir: Path,
    output_dir: Path,
    dataset_id: int = DATASET_ID,
    config: str = "3d_fullres",
    fold: Union[int, str] = "all",
    plans: str = "nnUNetResEncUNetMPlans",
    epochs: Optional[int] = None,
    trainer: str = "nnUNetTrainer",
    save_probabilities: bool = True,
    timeout: Optional[int] = None
) -> bool:
    """Run inference with trained model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer_name = get_trainer_name(epochs, trainer)

    cmd = f"nnUNetv2_predict -d {dataset_id:03d} -c {config} -f {fold}"
    cmd += f" -i {input_dir} -o {output_dir} -p {plans} -tr {trainer_name}"
    cmd += " -npp 2 -nps 2 --verbose"

    if save_probabilities:
        cmd += " --save_probabilities"

    return run_command(cmd, "Inference", timeout=timeout)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def create_spacing_json(output_path: Path, shape: tuple, spacing: tuple = (1.0, 1.0, 1.0)):
    """Create JSON sidecar with spacing info for TIFF files."""
    json_data = {"spacing": list(spacing)}
    with open(output_path, "w") as f:
        json.dump(json_data, f)


def create_dataset_json(output_dir: Path, num_training: int, file_ending: str = ".tif") -> dict:
    """Create dataset.json with ignore label support."""
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
    return dataset_json


def prepare_single_case(args) -> bool:
    """Prepare a single TIFF file for nnUNet."""
    import tifffile

    src_path, dest_path, json_path, use_symlinks = args
    try:
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


def prepare_raw_dataset(input_dir: Path, max_cases: Optional[int] = None, use_symlinks: bool = True):
    """Convert competition data to nnUNet raw format."""
    from tqdm import tqdm

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

    # Prepare arguments for parallel processing
    tasks = []
    for img_path in image_files:
        case_id = img_path.stem
        label_path = train_labels_dir / img_path.name

        if not label_path.exists():
            continue

        tasks.append((
            img_path,
            images_dir / f"{case_id}_0000.tif",
            images_dir / f"{case_id}_0000.json",
            use_symlinks
        ))
        tasks.append((
            label_path,
            labels_dir / f"{case_id}.tif",
            labels_dir / f"{case_id}.json",
            use_symlinks
        ))

    # Process in parallel
    num_workers = get_num_workers()
    print(f"Processing with {num_workers} workers...")

    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(prepare_single_case, tasks),
            total=len(tasks),
            desc="Preparing dataset"
        ))

    num_converted = sum(results) // 2  # Each case has image + label
    create_dataset_json(dataset_dir, num_converted, file_ending=".tif")

    print(f"\nDataset prepared: {num_converted} cases")
    print(f"Location: {dataset_dir}")
    return dataset_dir


def prepare_test_data(input_dir: Path, output_dir: Path, use_symlinks: bool = True) -> Path:
    """Prepare test TIFF images for inference."""
    from tqdm import tqdm

    output_dir.mkdir(parents=True, exist_ok=True)
    test_images_dir = input_dir / "test_images"

    if not test_images_dir.exists():
        print(f"ERROR: {test_images_dir} not found!")
        return output_dir

    test_files = sorted(test_images_dir.glob("*.tif"))
    print(f"Found {len(test_files)} test cases")

    tasks = []
    for img_path in test_files:
        case_id = img_path.stem
        tasks.append((
            img_path,
            output_dir / f"{case_id}_0000.tif",
            output_dir / f"{case_id}_0000.json",
            use_symlinks
        ))

    with Pool(get_num_workers()) as pool:
        list(tqdm(
            pool.imap(prepare_single_case, tasks),
            total=len(tasks),
            desc="Preparing test data"
        ))

    return output_dir


# =============================================================================
# POST-PROCESSING
# =============================================================================

def predictions_to_tiff(pred_dir: Path, output_dir: Path):
    """Convert nnUNet predictions to TIFF files."""
    import numpy as np
    import tifffile
    from tqdm import tqdm

    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = list(pred_dir.glob("*.npz"))
    tif_files = list(pred_dir.glob("*.tif"))

    if npz_files:
        print(f"Converting {len(npz_files)} NPZ files to TIFF...")
        for npz_path in tqdm(npz_files, desc="Converting"):
            case_id = npz_path.stem
            data = np.load(npz_path)
            probs = data['probabilities']
            pred = np.argmax(probs, axis=0).astype(np.uint8)
            tifffile.imwrite(output_dir / f"{case_id}.tif", pred)
    elif tif_files:
        print(f"Copying {len(tif_files)} TIFF files...")
        for tif_path in tqdm(tif_files, desc="Copying"):
            case_id = tif_path.stem
            pred = tifffile.imread(str(tif_path)).astype(np.uint8)
            tifffile.imwrite(output_dir / f"{case_id}.tif", pred)
    else:
        print(f"No prediction files found in {pred_dir}")


def generate_submission(
    predictions_dir: Path,
    output_zip: Path,
    delete_after_zip: bool = False
) -> Optional[Path]:
    """Create submission ZIP from TIFF predictions."""
    import zipfile
    from tqdm import tqdm

    if not predictions_dir.exists():
        print(f"ERROR: Predictions directory not found: {predictions_dir}")
        return None

    tiff_files = sorted(predictions_dir.glob("*.tif"))
    if not tiff_files:
        print(f"No TIFF files found in {predictions_dir}")
        return None

    print(f"Creating submission ZIP with {len(tiff_files)} files...")

    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for tiff_path in tqdm(tiff_files, desc="Zipping"):
            zipf.write(tiff_path, tiff_path.name)
            if delete_after_zip:
                tiff_path.unlink()

    zip_size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"Submission saved: {output_zip} ({zip_size_mb:.1f} MB)")
    return output_zip


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def full_pipeline(
    do_prepare_raw: bool = True,
    do_train: bool = True,
    do_inference: bool = True,
    config: str = "3d_fullres",
    fold: Union[int, str] = "all",
    planner: str = "nnUNetPlannerResEncM",
    plans: str = "nnUNetResEncUNetMPlans",
    epochs: Optional[int] = None,
    trainer: str = "nnUNetTrainer",
    continue_training: bool = False,
    num_gpus: Optional[int] = None,
    save_probabilities: bool = True,
    timeout: Optional[int] = None,
    max_cases: Optional[int] = None,
) -> bool:
    """Run complete training/inference pipeline."""

    if num_gpus is None:
        num_gpus = get_gpu_count() or 1

    print("=" * 60)
    print("Vesuvius nnUNet Training & Inference - Local")
    print("=" * 60)
    print(f"Config: {config}")
    print(f"Fold: {fold}")
    print(f"Epochs: {epochs or 1000}")
    print(f"GPUs: {num_gpus}")
    print(f"Plans: {plans}")
    print(f"Trainer: {trainer}")
    print("=" * 60)

    # Setup
    print("\n[1/4] Setting up environment...")
    setup_environment()

    # Check if preprocessed data exists
    preprocessed_dataset = NNUNET_PREPROCESSED / DATASET_NAME
    if not preprocessed_dataset.exists():
        print(f"\nERROR: Preprocessed data not found at {preprocessed_dataset}")
        print("Run preprocessing first:")
        print("  ./run-preprocessing-nohup.sh")
        return False

    print(f"Using preprocessed data: {preprocessed_dataset}")

    # Prepare raw data (symlinks only - fast)
    if do_prepare_raw:
        print("\n[2/4] Preparing raw dataset (symlinks)...")
        raw_dataset = NNUNET_RAW / DATASET_NAME
        if not raw_dataset.exists():
            prepare_raw_dataset(DATASET_DIR, max_cases=max_cases)
        else:
            print(f"Raw dataset exists: {raw_dataset}")

    # Training
    if do_train:
        print("\n[3/4] Training...")
        success = run_training(
            config=config,
            fold=fold,
            plans=plans,
            epochs=epochs,
            trainer=trainer,
            continue_training=continue_training,
            num_gpus=num_gpus,
            timeout=timeout
        )
        if not success:
            print("Training failed!")
            return False
    else:
        print("\n[3/4] Skipping training...")

    # Inference
    if do_inference:
        print("\n[4/4] Running inference...")

        # Prepare test data
        test_input_dir = NNUNET_WORK / "test_input"
        prepare_test_data(DATASET_DIR, test_input_dir)

        # Run inference
        predictions_dir = NNUNET_WORK / "predictions"
        success = run_inference(
            test_input_dir,
            predictions_dir,
            config=config,
            fold=fold,
            plans=plans,
            epochs=epochs,
            trainer=trainer,
            save_probabilities=save_probabilities,
            timeout=timeout
        )
        if not success:
            print("Inference failed!")
            return False

        # Convert to TIFF
        tiff_output_dir = NNUNET_OUTPUT / "predictions_tiff"
        predictions_to_tiff(predictions_dir, tiff_output_dir)
        print(f"Predictions saved to: {tiff_output_dir}")

        # Generate submission
        submission_zip = NNUNET_OUTPUT / "submission.zip"
        generate_submission(tiff_output_dir, submission_zip)
    else:
        print("\n[4/4] Skipping inference...")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)

    # Show output location
    if do_train:
        model_dir = get_training_output_dir(epochs, plans, config, fold, trainer)
        print(f"\nModel saved to: {model_dir}")
        progress_img = model_dir / "progress.png"
        if progress_img.exists():
            print(f"Training progress: {progress_img}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Vesuvius nnUNet Training & Inference - Local",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training only (50 epochs)
  python surface-nnunet-training-local.py --train --epochs 50

  # Inference only
  python surface-nnunet-training-local.py --inference

  # Full pipeline with 2 GPUs
  python surface-nnunet-training-local.py --train --inference --gpus 2

  # Continue training from checkpoint
  python surface-nnunet-training-local.py --train --continue-training
"""
    )

    # Pipeline stages
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--inference", action="store_true", help="Run inference")
    parser.add_argument("--skip-raw-prep", action="store_true", help="Skip raw data preparation")

    # Training configuration
    parser.add_argument("--config", default="3d_fullres",
                        choices=["2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres"],
                        help="nnUNet configuration (default: 3d_fullres)")
    parser.add_argument("--fold", default="all",
                        help="Fold number (0-4) or 'all' (default: all)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (1,5,10,20,50,100,250,500,750,1000)")
    parser.add_argument("--gpus", type=int, default=None,
                        help="Number of GPUs (default: auto-detect)")
    parser.add_argument("--continue-training", action="store_true",
                        help="Continue from last checkpoint")

    # Advanced options
    parser.add_argument("--trainer", default="nnUNetTrainer",
                        help="Trainer class (default: nnUNetTrainer)")
    parser.add_argument("--plans", default="nnUNetResEncUNetMPlans",
                        help="Plans name (default: nnUNetResEncUNetMPlans)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Command timeout in seconds")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Limit number of training cases")

    args = parser.parse_args()

    # Default to training if neither specified
    if not args.train and not args.inference:
        print("No action specified. Use --train and/or --inference")
        parser.print_help()
        sys.exit(1)

    # Convert fold to int if numeric
    fold = args.fold
    if fold.isdigit():
        fold = int(fold)

    success = full_pipeline(
        do_prepare_raw=not args.skip_raw_prep,
        do_train=args.train,
        do_inference=args.inference,
        config=args.config,
        fold=fold,
        plans=args.plans,
        epochs=args.epochs,
        trainer=args.trainer,
        continue_training=args.continue_training,
        num_gpus=args.gpus,
        timeout=args.timeout,
        max_cases=args.max_cases,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Vesuvius Surface Detection - Kaggle Submission Script

This script generates predictions for Kaggle submission using a trained nnUNet model.

Usage:
    # Test mode - verify inference matches validation predictions
    python surface-nnunet-submission.py --test

    # Full inference on test data
    python surface-nnunet-submission.py --submit

    # Custom model path
    python surface-nnunet-submission.py --submit --model-dir /path/to/model/fold_0

Environment:
    - Docker container with nnUNet installed
    - Trained model in nnUNet_results directory
    - Test data in competition data directory
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional
import zipfile

import numpy as np
import tifffile
from scipy import ndimage as ndi


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
    t_low: float = 0.5,
    t_high: float = 0.9,
    z_radius: int = 1,
    xy_radius: int = 0,
    dust_min_size: int = 100
) -> np.ndarray:
    """
    Post-processing with hysteresis thresholding.

    This method achieved the best validation score in our experiments:
    - Leaderboard: 0.5886 (vs 0.5690 for argmax)
    - TopoScore: 0.3185 (vs 0.2436 for argmax)

    Steps:
    1. 3D Hysteresis thresholding: propagate from high confidence to low
    2. Anisotropic morphological closing (Z direction only)
    3. Dust removal (remove small connected components)

    Args:
        probs: Probability array with shape (num_classes, D, H, W)
        t_low: Low threshold for hysteresis (default: 0.5)
        t_high: High threshold for hysteresis (default: 0.9)
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
# CONFIGURATION
# =============================================================================

# Default paths (Docker container)
DEFAULT_INPUT_DIR = Path("/kaggle/input/vesuvius-challenge-surface-detection")
DEFAULT_OUTPUT_DIR = Path("/kaggle/working")
DEFAULT_RESULTS_DIR = Path("/kaggle/working/nnUNet_results")

# Dataset configuration
DATASET_ID = 100
DATASET_NAME = f"Dataset{DATASET_ID:03d}_VesuviusSurface"


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

def setup_environment(results_dir: Path = DEFAULT_RESULTS_DIR):
    """Set up nnUNet environment variables."""
    # nnUNet requires these environment variables
    os.environ["nnUNet_raw"] = str(results_dir.parent / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(results_dir.parent / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(results_dir)

    print(f"nnUNet_results: {results_dir}")


def parse_model_dir(model_dir: Path) -> dict:
    """
    Parse model directory path to extract nnUNet configuration.

    Expected format: .../DatasetXXX_Name/TrainerName__PlansName__Config/fold_X/

    Returns:
        dict with keys: trainer, plans, config, fold
    """
    parts = model_dir.parts

    # Find fold
    fold = None
    for part in reversed(parts):
        if part.startswith("fold_"):
            fold = part.replace("fold_", "")
            break

    # Find trainer__plans__config
    trainer = None
    plans = None
    config = None
    for part in parts:
        if "__" not in part:
            continue
        segments = part.split("__")
        if len(segments) >= 3:
            trainer = segments[0]
            plans = segments[1]
            config = segments[2]
            break

    return {
        "trainer": trainer,
        "plans": plans,
        "config": config,
        "fold": fold,
    }


# =============================================================================
# DATA PREPARATION
# =============================================================================

def create_spacing_json(output_path: Path, spacing: tuple = (1.0, 1.0, 1.0)):
    """Create JSON sidecar with spacing info for TIFF files."""
    json_data = {"spacing": list(spacing)}
    with open(output_path, "w") as f:
        json.dump(json_data, f)


def prepare_input_images(
    input_images: list,
    output_dir: Path,
    use_symlinks: bool = True
) -> list:
    """
    Prepare input images for nnUNet inference.

    nnUNet expects:
    - Files with _0000 suffix (channel indicator)
    - JSON sidecar files with spacing info

    Args:
        input_images: List of input image paths
        output_dir: Directory to write prepared images
        use_symlinks: Use symlinks instead of copying

    Returns:
        List of case IDs that were prepared
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    case_ids = []
    for img_path in input_images:
        case_id = img_path.stem

        # Destination paths
        dest_path = output_dir / f"{case_id}_0000.tif"
        json_path = output_dir / f"{case_id}_0000.json"

        # Link or copy file
        if use_symlinks:
            if not dest_path.exists():
                dest_path.symlink_to(img_path.resolve())
        else:
            shutil.copy2(img_path, dest_path)

        # Create JSON sidecar with default spacing
        create_spacing_json(json_path)
        case_ids.append(case_id)

    return case_ids


# =============================================================================
# INFERENCE
# =============================================================================

def run_inference(
    input_dir: Path,
    output_dir: Path,
    model_config: dict,
    dataset_id: int = DATASET_ID,
    save_probabilities: bool = False,
    disable_tta: bool = False,
    timeout: Optional[int] = None,
) -> bool:
    """
    Run nnUNet inference.

    Args:
        input_dir: Directory with prepared input images
        output_dir: Directory to save predictions
        model_config: Dict with trainer, plans, config, fold
        dataset_id: nnUNet dataset ID
        save_probabilities: Save probability maps
        disable_tta: Disable test-time augmentation (faster but less accurate)
        timeout: Command timeout in seconds

    Returns:
        True if inference succeeded
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = model_config["trainer"]
    plans = model_config["plans"]
    config = model_config["config"]
    fold = model_config["fold"]

    cmd = f"nnUNetv2_predict -d {dataset_id:03d} -c {config} -f {fold}"
    cmd += f" -i {input_dir} -o {output_dir} -p {plans} -tr {trainer}"
    cmd += " -npp 2 -nps 2 --verbose"

    if save_probabilities:
        cmd += " --save_probabilities"

    if disable_tta:
        cmd += " --disable_tta"

    print(f"Running: {cmd}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
    except subprocess.TimeoutExpired:
        print(f"Inference TIMEOUT after {timeout}s!")
        return False

    if result.returncode != 0:
        print(f"Inference FAILED!")
        print(f"STDERR:\n{result.stderr[-3000:]}")
        return False

    print("Inference complete!")
    return True


def convert_predictions_to_tiff(
    pred_dir: Path,
    output_dir: Path,
    use_hysteresis: bool = True
):
    """
    Convert nnUNet predictions to TIFF format with post-processing.

    Handles both NPZ (probability maps) and direct TIFF outputs.
    When NPZ files are available, applies hysteresis post-processing
    which achieves better TopoScore than simple argmax.

    Args:
        pred_dir: Directory with nnUNet predictions
        output_dir: Directory to save TIFF predictions
        use_hysteresis: Use hysteresis post-processing (default: True)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for different prediction formats
    npz_files = list(pred_dir.glob("*.npz"))
    tif_files = list(pred_dir.glob("*.tif"))

    if npz_files:
        postprocess_method = "hysteresis" if use_hysteresis else "argmax"
        print(f"Converting {len(npz_files)} NPZ files to TIFF (post-processing: {postprocess_method})...")
        for npz_path in npz_files:
            case_id = npz_path.stem
            # Load probabilities
            data = np.load(npz_path)
            probs = data['probabilities']

            if use_hysteresis:
                # Hysteresis post-processing (best validation score)
                pred = postprocess_hysteresis(probs)
            else:
                # Simple argmax (baseline)
                pred = np.argmax(probs, axis=0).astype(np.uint8)

            tifffile.imwrite(output_dir / f"{case_id}.tif", pred)
    elif tif_files:
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

def create_submission_zip(
    predictions_dir: Path,
    output_zip: Path,
) -> Path:
    """
    Create submission ZIP from TIFF predictions.

    Args:
        predictions_dir: Directory with TIFF predictions
        output_zip: Output ZIP file path

    Returns:
        Path to created ZIP file
    """
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
# TEST MODE
# =============================================================================

def run_test_mode(
    model_dir: Path,
    input_dir: Path = DEFAULT_INPUT_DIR,
    validation_dir: Optional[Path] = None,
    case_id: Optional[str] = None,
    disable_tta: bool = False,
) -> bool:
    """
    Test mode: verify inference matches validation predictions.

    This runs inference on a training sample and compares with
    the validation output generated during training.

    Args:
        model_dir: Path to model fold directory
        input_dir: Competition data directory
        validation_dir: Path to validation predictions (auto-detected if None)
        case_id: Specific case to test (auto-selected if None)
        disable_tta: Disable test-time augmentation

    Returns:
        True if test passed (predictions match)
    """
    print("=" * 60)
    print("TEST MODE - Verifying inference")
    print("=" * 60)

    if disable_tta:
        print("WARNING: TTA is disabled. Test may fail if validation used TTA.")

    # Parse model configuration
    model_config = parse_model_dir(model_dir)
    print(f"Model config: {model_config}")

    if None in model_config.values():
        print(f"ERROR: Could not parse model directory: {model_dir}")
        return False

    # Find validation directory
    if validation_dir is None:
        validation_dir = model_dir / "validation"

    if not validation_dir.exists():
        print(f"ERROR: Validation directory not found: {validation_dir}")
        return False

    # Get available validation cases
    val_files = sorted(validation_dir.glob("*.tif"))
    if not val_files:
        print(f"ERROR: No validation predictions found in {validation_dir}")
        return False

    print(f"Found {len(val_files)} validation predictions")

    # Select test case
    if case_id is None:
        # Use first validation case
        case_id = val_files[0].stem

    val_pred_path = validation_dir / f"{case_id}.tif"
    if not val_pred_path.exists():
        print(f"ERROR: Validation prediction not found: {val_pred_path}")
        return False

    print(f"Test case: {case_id}")

    # Find corresponding input image
    train_images_dir = input_dir / "train_images"
    input_image_path = train_images_dir / f"{case_id}.tif"

    if not input_image_path.exists():
        print(f"ERROR: Input image not found: {input_image_path}")
        return False

    # Create temp directory for test inference
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_input_dir = tmpdir / "input"
        test_output_dir = tmpdir / "output"

        # Prepare input image
        print(f"\nPreparing input image...")
        prepare_input_images([input_image_path], test_input_dir, use_symlinks=False)

        # Run inference
        print(f"\nRunning inference (TTA: {'disabled' if disable_tta else 'enabled'})...")
        success = run_inference(
            test_input_dir,
            test_output_dir,
            model_config,
            save_probabilities=False,
            disable_tta=disable_tta,
        )

        if not success:
            print("ERROR: Inference failed!")
            return False

        # Load predictions
        test_pred_path = test_output_dir / f"{case_id}.tif"
        if not test_pred_path.exists():
            print(f"ERROR: Test prediction not found: {test_pred_path}")
            return False

        test_pred = tifffile.imread(str(test_pred_path))
        val_pred = tifffile.imread(str(val_pred_path))

        # Compare predictions
        print(f"\nComparing predictions...")
        print(f"Test prediction shape: {test_pred.shape}, dtype: {test_pred.dtype}")
        print(f"Val prediction shape:  {val_pred.shape}, dtype: {val_pred.dtype}")

        if test_pred.shape != val_pred.shape:
            print(f"ERROR: Shape mismatch!")
            return False

        # Calculate match statistics
        match = (test_pred == val_pred)
        match_ratio = match.sum() / match.size
        diff_count = (~match).sum()

        print(f"\nMatch ratio: {match_ratio:.6f} ({match_ratio*100:.4f}%)")
        print(f"Different voxels: {diff_count} / {match.size}")

        # Check unique values
        test_unique = np.unique(test_pred)
        val_unique = np.unique(val_pred)
        print(f"\nTest unique values: {test_unique}")
        print(f"Val unique values:  {val_unique}")

        # Sum comparison
        test_sum = int(test_pred.sum())
        val_sum = int(val_pred.sum())
        print(f"\nTest sum (foreground voxels): {test_sum}")
        print(f"Val sum (foreground voxels):  {val_sum}")
        print(f"Difference: {abs(test_sum - val_sum)}")

        # Determine pass/fail
        # Allow small differences due to floating point / GPU non-determinism
        if match_ratio >= 0.9999:  # 99.99% match
            print("\n" + "=" * 60)
            print("TEST PASSED - Predictions match!")
            print("=" * 60)
            return True
        elif match_ratio >= 0.99:  # 99% match - likely GPU non-determinism
            print("\n" + "=" * 60)
            print("TEST PASSED (with minor differences)")
            print("Minor differences may be due to GPU non-determinism")
            print("=" * 60)
            return True
        else:
            print("\n" + "=" * 60)
            print("TEST FAILED - Predictions do not match!")
            print("=" * 60)
            return False


# =============================================================================
# SUBMIT MODE
# =============================================================================

def run_submit_mode(
    model_dir: Path,
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    disable_tta: bool = False,
    use_hysteresis: bool = True,
) -> Optional[Path]:
    """
    Submit mode: generate predictions for all test images.

    Args:
        model_dir: Path to model fold directory
        input_dir: Competition data directory
        output_dir: Output directory for predictions and submission
        disable_tta: Disable test-time augmentation
        use_hysteresis: Use hysteresis post-processing (default: True)

    Returns:
        Path to submission ZIP if successful, None otherwise
    """
    print("=" * 60)
    print("SUBMIT MODE - Generating predictions")
    print("=" * 60)

    # Parse model configuration
    model_config = parse_model_dir(model_dir)
    print(f"Model config: {model_config}")

    if None in model_config.values():
        print(f"ERROR: Could not parse model directory: {model_dir}")
        return None

    # Find test images
    test_images_dir = input_dir / "test_images"
    if not test_images_dir.exists():
        print(f"ERROR: Test images directory not found: {test_images_dir}")
        return None

    test_images = sorted(test_images_dir.glob("*.tif"))
    if not test_images:
        print(f"ERROR: No test images found in {test_images_dir}")
        return None

    print(f"Found {len(test_images)} test images")

    # Prepare directories
    test_input_dir = output_dir / "test_input"
    predictions_dir = output_dir / "predictions"
    predictions_tiff_dir = output_dir / "predictions_tiff"

    # Prepare input images
    print(f"\nPreparing input images...")
    case_ids = prepare_input_images(test_images, test_input_dir, use_symlinks=True)
    print(f"Prepared {len(case_ids)} cases")

    # Run inference
    print(f"\nRunning inference (TTA: {'disabled' if disable_tta else 'enabled'})...")
    success = run_inference(
        test_input_dir,
        predictions_dir,
        model_config,
        save_probabilities=True,
        disable_tta=disable_tta,
    )

    if not success:
        print("ERROR: Inference failed!")
        return None

    # Convert predictions to TIFF with post-processing
    print(f"\nConverting predictions to TIFF...")
    convert_predictions_to_tiff(predictions_dir, predictions_tiff_dir, use_hysteresis=use_hysteresis)

    # Create submission ZIP
    print(f"\nCreating submission ZIP...")
    submission_zip = output_dir / "submission.zip"
    create_submission_zip(predictions_tiff_dir, submission_zip)

    print("\n" + "=" * 60)
    print("SUBMISSION COMPLETE!")
    print(f"Output: {submission_zip}")
    print("=" * 60)

    return submission_zip


# =============================================================================
# MAIN
# =============================================================================

def find_default_model_dir(results_dir: Path = DEFAULT_RESULTS_DIR) -> Optional[Path]:
    """Find default model directory from nnUNet results."""
    # Look for Dataset100_VesuviusSurface
    dataset_dir = results_dir / DATASET_NAME
    if not dataset_dir.exists():
        return None

    # Find trainer directories
    trainer_dirs = [d for d in dataset_dir.iterdir() if d.is_dir() and "__" in d.name]
    if not trainer_dirs:
        return None

    # Use most recently modified
    trainer_dir = max(trainer_dirs, key=lambda d: d.stat().st_mtime)

    # Find fold directories
    fold_dirs = sorted([d for d in trainer_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    if not fold_dirs:
        return None

    # Prefer fold_0, then fold_all
    for preferred in ["fold_0", "fold_all"]:
        for fold_dir in fold_dirs:
            if fold_dir.name == preferred:
                return fold_dir

    return fold_dirs[0]


def main():
    parser = argparse.ArgumentParser(
        description="Vesuvius Surface Detection - Kaggle Submission Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--test", action="store_true",
        help="Test mode: verify inference matches validation predictions"
    )
    mode_group.add_argument(
        "--submit", action="store_true",
        help="Submit mode: generate predictions for all test images"
    )

    # Paths
    parser.add_argument(
        "--model-dir", type=Path, default=None,
        help="Path to model fold directory (auto-detected if not specified)"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help=f"Competition data directory (default: {DEFAULT_INPUT_DIR})"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help=f"nnUNet results directory (default: {DEFAULT_RESULTS_DIR})"
    )

    # Test mode options
    parser.add_argument(
        "--case-id", type=str, default=None,
        help="Specific case ID to test (test mode only)"
    )

    # Inference options
    parser.add_argument(
        "--disable-tta", action="store_true",
        help="Disable test-time augmentation (faster but less accurate). "
             "Note: Test mode will fail if validation was run with TTA enabled."
    )

    # Post-processing options
    parser.add_argument(
        "--no-hysteresis", action="store_true",
        help="Disable hysteresis post-processing (use simple argmax instead). "
             "Hysteresis achieves +0.02 better validation score."
    )

    args = parser.parse_args()

    # Setup environment
    setup_environment(args.results_dir)

    # Find model directory
    model_dir = args.model_dir
    if model_dir is None:
        model_dir = find_default_model_dir(args.results_dir)
        if model_dir is None:
            print("ERROR: Could not find model directory. Use --model-dir to specify.")
            return 1
        print(f"Auto-detected model directory: {model_dir}")

    # Verify model exists
    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        return 1

    # Check for checkpoint
    checkpoint_final = model_dir / "checkpoint_final.pth"
    checkpoint_best = model_dir / "checkpoint_best.pth"
    if not checkpoint_final.exists() and not checkpoint_best.exists():
        print(f"ERROR: No checkpoint found in {model_dir}")
        return 1

    # Run selected mode
    if args.test:
        success = run_test_mode(
            model_dir=model_dir,
            input_dir=args.input_dir,
            case_id=args.case_id,
            disable_tta=args.disable_tta,
        )
        return 0 if success else 1
    else:
        result = run_submit_mode(
            model_dir=model_dir,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            disable_tta=args.disable_tta,
            use_hysteresis=not args.no_hysteresis,
        )
        return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())

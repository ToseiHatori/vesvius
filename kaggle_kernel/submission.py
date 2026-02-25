#!/usr/bin/env python3
"""
Vesuvius Surface Detection - Kaggle Submission Script
nnUNet 3d_lowres + 3d_fullres, fold_0 + fold_1 ensemble (2x T4 GPU), TTA enabled, opening+closing post-processing

Optimized with Python API for faster inference:
- Model loaded once per (config, fold) combination
- Parallel inference on 2 GPUs
- Cumulative probability averaging across configs
"""

import os
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

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

import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tifffile
from scipy import ndimage as ndi

# nnUNet Python API
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
CONFIGS = ["3d_lowres", "3d_fullres"]  # Ensemble across configs
FOLDS = ["0", "1"]  # Ensemble of fold_0 and fold_1

# Inference settings
DISABLE_TTA = True  # Disable TTA for faster inference
TILE_STEP_SIZE = 0.3  # Sliding window step size (0.3 = 70% overlap)
BATCH_SIZE = 5  # Number of cases to process before ensemble/postprocess

# Post-processing settings
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


def postprocess_and_save(avg_probs: np.ndarray, output_dir: Path, case_id: str) -> str:
    """Run post-processing and save the result.

    This function is designed to be run in a separate thread/process
    to overlap with GPU inference.

    Args:
        avg_probs: Averaged probability array with shape (num_classes, D, H, W)
        output_dir: Output directory for TIFF file
        case_id: Case identifier for the output filename

    Returns:
        case_id for logging
    """
    pred = postprocess_opening_closing(avg_probs)
    tifffile.imwrite(output_dir / f"{case_id}.tif", pred)
    return case_id


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

def setup_environment():
    """Set up nnUNet environment variables."""
    # Kaggle dataset extracts zip contents WITHOUT the parent folder name
    # Actual structure: /kaggle/input/vesvius-model-weights/3d_lowres/fold_0/checkpoint_final.pth
    # nnUNet expects: {nnUNet_results}/Dataset100_VesuviusSurface/nnUNetTrainer__.../__config/fold_0/...
    # Solution: Create symlink at the trainer level for each config

    # Use working directory for nnUNet_results so we can create symlinks
    results_dir = OUTPUT_DIR / "nnUNet_results"
    dataset_dir = results_dir / DATASET_NAME

    os.environ["nnUNet_raw"] = str(OUTPUT_DIR / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(OUTPUT_DIR / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(results_dir)

    print(f"nnUNet_results: {results_dir}")

    # Create symlinks for all configs
    dataset_dir.mkdir(parents=True, exist_ok=True)
    model_dirs = {}

    for config in CONFIGS:
        trainer_dir = dataset_dir / f"{TRAINER}__{PLANS}__{config}"

        # New structure: MODEL_DATASET_DIR/3d_lowres/ or MODEL_DATASET_DIR/3d_fullres/
        if not trainer_dir.exists():
            trainer_dir.symlink_to(MODEL_DATASET_DIR / config)
        print(f"Symlink: {trainer_dir} -> {MODEL_DATASET_DIR / config}")

        # Verify all fold models exist for this config
        for fold in FOLDS:
            model_dir = trainer_dir / f"fold_{fold}"
            checkpoint = model_dir / "checkpoint_final.pth"

            if not checkpoint.exists():
                print(f"ERROR: Checkpoint not found: {checkpoint}")
                print(f"Available files in {MODEL_DATASET_DIR}:")
                for p in MODEL_DATASET_DIR.rglob("*"):
                    print(f"  {p}")
                sys.exit(1)

            print(f"Model checkpoint ({config}/fold_{fold}): {checkpoint}")
            model_dirs[(config, fold)] = model_dir

    return model_dirs

# =============================================================================
# INFERENCE (Python API - optimized)
# =============================================================================

class TiffReader:
    """Simple TIFF reader compatible with nnUNet's image reader interface."""

    @staticmethod
    def read_images(image_fnames: list) -> tuple:
        """Read TIFF images and return as numpy array with properties.

        Args:
            image_fnames: List of TIFF file paths (one per modality)

        Returns:
            Tuple of (image_array, properties_dict)
        """
        images = []
        for fname in image_fnames:
            img = tifffile.imread(str(fname))
            images.append(img)

        # Stack modalities: (C, D, H, W)
        image_array = np.stack(images, axis=0).astype(np.float32)

        # Properties dict (spacing is isotropic for this dataset)
        properties = {
            'spacing': (1.0, 1.0, 1.0),
        }

        return image_array, properties


def create_predictor(gpu_id: int) -> nnUNetPredictor:
    """Create an nnUNetPredictor for a specific GPU.

    Args:
        gpu_id: GPU device ID

    Returns:
        Initialized nnUNetPredictor (model not yet loaded)
    """
    predictor = nnUNetPredictor(
        tile_step_size=TILE_STEP_SIZE,
        use_gaussian=True,
        use_mirroring=not DISABLE_TTA,  # TTA = mirroring
        perform_everything_on_device=True,
        device=torch.device('cuda', gpu_id),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    return predictor


def load_model(predictor: nnUNetPredictor, config: str, fold: str) -> None:
    """Load a trained model into the predictor.

    Args:
        predictor: nnUNetPredictor instance
        config: Model configuration (e.g., '3d_lowres')
        fold: Fold number as string (e.g., '0')
    """
    model_folder = Path(os.environ["nnUNet_results"]) / DATASET_NAME / f"{TRAINER}__{PLANS}__{config}"
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=(int(fold),),
        checkpoint_name='checkpoint_final.pth',
    )
    print(f"  Loaded model: {config}/fold_{fold}")


def predict_single_case(
    predictor: nnUNetPredictor,
    img_path: Path,
) -> np.ndarray:
    """Run inference on a single case and return probabilities.

    Args:
        predictor: Initialized nnUNetPredictor with model loaded
        img_path: Path to input TIFF image

    Returns:
        Probability array with shape (num_classes, D, H, W)
    """
    # Read image
    image_array, properties = TiffReader.read_images([img_path])

    # Run inference with probability output
    _seg, probs = predictor.predict_single_npy_array(
        image_array,
        properties,
        None,  # segmentation_previous_stage
        None,  # output_file_truncated
        True   # save_or_return_probabilities
    )

    return probs


def run_inference_batch(
    batch_images: list,
    predictors: dict,
    output_dir: Path,
) -> bool:
    """Run inference on a batch of cases using all configs/folds.

    Pipelined GPU parallelization:
    - Each GPU (fold) processes cases and sends results as they complete
    - Post-processing runs in parallel threads as soon as all folds complete for a case
    - GPU inference and CPU post-processing overlap for maximum throughput

    Args:
        batch_images: List of input image paths for this batch
        predictors: Dict mapping (config, fold) -> nnUNetPredictor (already loaded)
        output_dir: Output directory for final TIFF files

    Returns:
        True if successful, False otherwise
    """
    n_models = len(CONFIGS) * len(FOLDS)

    # Shared state for collecting results
    case_results = {}  # case_id -> {fold: probs}
    case_locks = {img.stem: threading.Lock() for img in batch_images}
    errors = []
    errors_lock = threading.Lock()

    # Post-processing executor (runs in parallel with GPU inference)
    postprocess_executor = ThreadPoolExecutor(max_workers=4)
    postprocess_futures = []
    futures_lock = threading.Lock()

    def on_fold_result(case_id: str, fold_id: str, probs: np.ndarray) -> None:
        """Called when a fold completes inference for a case.

        If all folds are complete for this case, submits post-processing.
        """
        with case_locks[case_id]:
            if case_id not in case_results:
                case_results[case_id] = {}
            case_results[case_id][fold_id] = probs

            # Check if all folds are complete for this case
            if len(case_results[case_id]) == len(FOLDS):
                # Compute ensemble average
                total_probs = sum(case_results[case_id].values())
                avg_probs = total_probs / n_models

                # Release memory for fold results
                del case_results[case_id]

                # Submit post-processing to run in parallel
                future = postprocess_executor.submit(
                    postprocess_and_save, avg_probs, output_dir, case_id
                )
                with futures_lock:
                    postprocess_futures.append(future)

    def process_gpu(fold_id: str, imgs: list) -> None:
        """Process all cases for a single fold (GPU).

        Processes each case through all configs, then sends result immediately.
        """
        try:
            for img_path in imgs:
                case_id = img_path.stem
                cumulative_probs = None

                # Accumulate probabilities across all configs for this case
                for config in CONFIGS:
                    predictor = predictors[(config, fold_id)]
                    probs = predict_single_case(predictor, img_path)

                    if cumulative_probs is None:
                        cumulative_probs = probs
                    else:
                        cumulative_probs += probs

                # Send result immediately (enables pipelining)
                on_fold_result(case_id, fold_id, cumulative_probs)

        except Exception as e:
            error_msg = f"fold_{fold_id}: {e}\n{traceback.format_exc()}"
            with errors_lock:
                errors.append(error_msg)

    # Start GPU threads
    threads = []
    for fold in FOLDS:
        t = threading.Thread(target=process_gpu, args=(fold, batch_images))
        threads.append(t)
        t.start()

    # Wait for all GPU threads to complete
    for t in threads:
        t.join()

    # Check for errors
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        postprocess_executor.shutdown(wait=False)
        return False

    # Wait for all post-processing to complete
    for future in postprocess_futures:
        case_id = future.result()
        print(f"    {case_id}: saved")

    postprocess_executor.shutdown(wait=True)
    return True


def initialize_all_predictors() -> dict:
    """Initialize predictors for all config/fold combinations.

    Returns:
        Dict mapping (config, fold) -> nnUNetPredictor with model loaded
    """
    predictors = {}

    for config in CONFIGS:
        for fold in FOLDS:
            gpu_id = int(fold)  # fold 0 -> GPU 0, fold 1 -> GPU 1
            print(f"Loading model: {config}/fold_{fold} on GPU {gpu_id}")

            predictor = create_predictor(gpu_id)
            load_model(predictor, config, fold)
            predictors[(config, fold)] = predictor

    return predictors


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
    print("Vesuvius nnUNet Submission (Python API - Optimized)")
    print(f"Configs: {', '.join(CONFIGS)}")
    print(f"Folds: {', '.join(FOLDS)}")
    print(f"Total models: {len(CONFIGS) * len(FOLDS)}")
    print(f"TTA: {'disabled' if DISABLE_TTA else 'enabled'}")
    print(f"Tile step size: {TILE_STEP_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Post-processing: {'opening+closing' if USE_HYSTERESIS else 'argmax'}")
    print(f"Ensemble: probability averaging across all config/fold combinations")
    print(f"Optimization: Model loaded once per (config, fold), not per case")
    print("=" * 60)

    # Setup environment
    setup_environment()

    # Find test images
    test_images_dir = INPUT_DIR / "test_images"
    test_images = sorted(test_images_dir.glob("*.tif"))
    print(f"\nFound {len(test_images)} test images")

    if not test_images:
        print("ERROR: No test images found!")
        return 1

    # Prepare output directory
    predictions_tiff_dir = OUTPUT_DIR / "predictions_tiff"
    predictions_tiff_dir.mkdir(parents=True, exist_ok=True)

    # Initialize all predictors (models loaded once)
    n_models = len(CONFIGS) * len(FOLDS)
    print(f"\nLoading {n_models} models...")
    predictors = initialize_all_predictors()
    print(f"All models loaded!")

    # Process in batches
    n_batches = (len(test_images) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\nProcessing {len(test_images)} cases in {n_batches} batches of up to {BATCH_SIZE}...")

    for batch_idx in range(n_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(test_images))
        batch_images = test_images[start_idx:end_idx]

        print(f"\n[Batch {batch_idx + 1}/{n_batches}] Processing {len(batch_images)} cases...")

        success = run_inference_batch(
            batch_images=batch_images,
            predictors=predictors,
            output_dir=predictions_tiff_dir,
        )

        if not success:
            print(f"ERROR: Inference failed for batch {batch_idx + 1}!")
            return 1

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

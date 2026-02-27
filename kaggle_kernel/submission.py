#!/usr/bin/env python3
"""
Vesuvius Surface Detection - Kaggle Submission Script
nnUNet weighted ensemble: 4 models (fullres/lowres x fold0/fold1), 2x T4 GPU, opening+closing post-processing

Optimized with Python API for faster inference:
- Model loaded once per (config, fold) combination
- Parallel inference on 2 GPUs
- Weighted ensemble: configurable weights per (config, fold) combination
- TTA: configurable via ENABLE_TTA flag (adaptive or disabled)
"""

import os
import subprocess
import sys
import threading
import time
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

import zipfile
from pathlib import Path
from typing import List, Optional

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

# Model configurations with individual weights
# Each entry: (config, fold, weight)
# Weights are normalized automatically if they don't sum to 1.0
MODEL_CONFIGS = [
    ("3d_fullres", "0", 0.30),   # 2000epoch fullres fold0
    ("3d_fullres", "1", 0.30),   # 2000epoch fullres fold1
    ("3d_lowres", "0", 0.20),    # 2000epoch lowres fold0
    ("3d_lowres", "1", 0.20),    # 2000epoch lowres fold1
]

# Normalize weights to sum to 1.0
_weight_sum = sum(w for _, _, w in MODEL_CONFIGS)
MODEL_WEIGHTS = {(cfg, fold): w / _weight_sum for cfg, fold, w in MODEL_CONFIGS}
CONFIGS = list(set(cfg for cfg, _, _ in MODEL_CONFIGS))  # Unique configs for setup

# Inference settings
TILE_STEP_SIZE = 0.3  # Sliding window step size (0.3 = 70% overlap)
BATCH_SIZE = 5  # Number of cases to process before ensemble/postprocess

# TTA settings
ENABLE_TTA = False  # Set True to enable adaptive TTA, False for no TTA (faster)
KAGGLE_TIME_LIMIT_SECONDS = 9 * 60 * 60  # 9 hours
TTA_SPEEDUP_FACTOR = 8  # TTA is ~8x slower than non-TTA
SAFETY_MARGIN_SECONDS = 5 * 60  # 5 minutes safety buffer

# Post-processing settings
USE_HYSTERESIS = True  # Hysteresis post-processing (validation: 0.5886 vs 0.5690 for argmax)

# =============================================================================
# ADAPTIVE TTA CONTROLLER
# =============================================================================

class AdaptiveTTAController:
    """Time-aware TTA controller that switches off TTA when time is running out."""

    def __init__(self, total_cases: int, time_limit: float = KAGGLE_TIME_LIMIT_SECONDS):
        self.total_cases = total_cases
        self.time_limit = time_limit
        self.script_start_time = time.time()
        self.inference_start_time: Optional[float] = None
        self.processed_cases = 0
        self.tta_enabled = True
        self.tta_disabled_at_case: Optional[int] = None
        self.case_times: List[float] = []  # Processing time per case (with TTA)

    def mark_inference_start(self) -> None:
        """Mark the start of inference (after model loading)."""
        self.inference_start_time = time.time()
        setup_time = self.inference_start_time - self.script_start_time
        remaining = self.time_limit - setup_time - SAFETY_MARGIN_SECONDS
        print(f"[TTA Controller] Setup time: {setup_time:.1f}s")
        print(f"[TTA Controller] Available inference time: {remaining:.1f}s ({remaining/60:.1f}min)")
        print(f"[TTA Controller] Total cases: {self.total_cases}")

    def record_case_time(self, elapsed: float) -> None:
        """Record processing time for a single case."""
        self.case_times.append(elapsed)
        self.processed_cases += 1

    def get_avg_case_time(self) -> float:
        """Get average processing time per case (with TTA)."""
        if not self.case_times:
            return 0.0
        return sum(self.case_times) / len(self.case_times)

    def should_use_tta(self) -> bool:
        """Determine whether to use TTA based on remaining time and cases."""
        # Global TTA disable switch
        if not ENABLE_TTA:
            return False

        if self.inference_start_time is None:
            return True  # Default to TTA before inference starts

        if not self.tta_enabled:
            return False  # Already switched off

        # Need at least a few samples to estimate
        if self.processed_cases < 2:
            return True

        remaining_cases = self.total_cases - self.processed_cases
        if remaining_cases <= 0:
            return self.tta_enabled

        elapsed_since_inference = time.time() - self.inference_start_time
        setup_time = self.inference_start_time - self.script_start_time
        remaining_time = self.time_limit - setup_time - elapsed_since_inference - SAFETY_MARGIN_SECONDS

        avg_time_with_tta = self.get_avg_case_time()
        avg_time_without_tta = avg_time_with_tta / TTA_SPEEDUP_FACTOR

        time_needed_with_tta = remaining_cases * avg_time_with_tta
        time_needed_without_tta = remaining_cases * avg_time_without_tta

        # Key insight: Continue TTA as long as we have margin to switch to no-TTA later
        # We switch to no-TTA when remaining_time is just enough for no-TTA processing
        if time_needed_without_tta < remaining_time:
            # Still have margin - continue with TTA
            # Log progress periodically
            if self.processed_cases % 10 == 0:
                margin = remaining_time - time_needed_without_tta
                print(f"  [TTA] margin: {margin:.0f}s, can do ~{int(margin / avg_time_with_tta)} more with TTA")
            return True

        # No more margin - need to switch to no-TTA now
        self.tta_enabled = False
        self.tta_disabled_at_case = self.processed_cases

        if time_needed_without_tta <= remaining_time * 1.1:  # Within 10% margin
            print(f"\n{'='*60}")
            print(f"[TTA Controller] SWITCHING TTA OFF (optimal timing)")
            print(f"  Processed: {self.processed_cases}/{self.total_cases} cases")
            print(f"  Remaining: {remaining_cases} cases")
            print(f"  Remaining time: {remaining_time:.1f}s ({remaining_time/60:.1f}min)")
            print(f"  Avg time/case (TTA): {avg_time_with_tta:.1f}s")
            print(f"  Estimated time/case (no TTA): {avg_time_without_tta:.1f}s")
            print(f"  Time needed without TTA: {time_needed_without_tta:.1f}s")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"[TTA Controller] WARNING: TIME CRITICAL!")
            print(f"  Switching to no-TTA but may still be tight!")
            print(f"  Remaining time: {remaining_time:.1f}s ({remaining_time/60:.1f}min)")
            print(f"  Time needed (no TTA): {time_needed_without_tta:.1f}s")
            print(f"  Overage: {time_needed_without_tta - remaining_time:.1f}s")
            print(f"{'='*60}\n")

        return False

    def print_summary(self) -> None:
        """Print summary at the end of inference."""
        if self.inference_start_time is None:
            return

        total_time = time.time() - self.script_start_time
        inference_time = time.time() - self.inference_start_time

        print(f"\n[TTA Controller] Summary:")
        print(f"  Total script time: {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"  Inference time: {inference_time:.1f}s ({inference_time/60:.1f}min)")
        print(f"  Processed cases: {self.processed_cases}")
        if self.tta_disabled_at_case is not None:
            print(f"  TTA disabled at case: {self.tta_disabled_at_case}")
            print(f"  Cases with TTA: {self.tta_disabled_at_case}")
            print(f"  Cases without TTA: {self.processed_cases - self.tta_disabled_at_case}")
        else:
            print(f"  TTA: enabled throughout")


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

    # Create symlinks for configs used in MODEL_CONFIGS
    dataset_dir.mkdir(parents=True, exist_ok=True)
    model_dirs = {}

    for config in CONFIGS:
        trainer_dir = dataset_dir / f"{TRAINER}__{PLANS}__{config}"

        # New structure: MODEL_DATASET_DIR/3d_lowres/ or MODEL_DATASET_DIR/3d_fullres/
        if not trainer_dir.exists():
            trainer_dir.symlink_to(MODEL_DATASET_DIR / config)
        print(f"Symlink: {trainer_dir} -> {MODEL_DATASET_DIR / config}")

    # Verify only the specific config+fold combinations we're using
    for config, fold, _ in MODEL_CONFIGS:
        trainer_dir = dataset_dir / f"{TRAINER}__{PLANS}__{config}"
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


def create_predictor(gpu_id: int, use_tta: bool = True) -> nnUNetPredictor:
    """Create an nnUNetPredictor for a specific GPU.

    Args:
        gpu_id: GPU device ID
        use_tta: Whether to enable TTA (test-time augmentation via mirroring)

    Returns:
        Initialized nnUNetPredictor (model not yet loaded)
    """
    predictor = nnUNetPredictor(
        tile_step_size=TILE_STEP_SIZE,
        use_gaussian=True,
        use_mirroring=use_tta,  # TTA = mirroring
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
    tta_controller: AdaptiveTTAController,
) -> bool:
    """Run inference on a batch of cases using specified config/fold combinations.

    Processes a small batch of cases through all models in MODEL_CONFIGS,
    then ensembles and saves. Models run in parallel across GPUs.

    Args:
        batch_images: List of input image paths for this batch
        predictors: Dict mapping (config, fold) -> nnUNetPredictor (already loaded)
        output_dir: Output directory for final TIFF files
        tta_controller: Adaptive TTA controller for time-based TTA switching

    Returns:
        True if successful, False otherwise
    """
    n_cases = len(batch_images)

    # Store cumulative probabilities in memory for this batch
    # Key: case_id, Value: cumulative probability sum
    batch_cumulative = {img.stem: None for img in batch_images}

    # Track time per case for TTA controller
    case_start_times = {img.stem: time.time() for img in batch_images}

    # Check and update TTA setting
    use_tta = tta_controller.should_use_tta()
    for predictor in predictors.values():
        predictor.use_mirroring = use_tta

    tta_status = "TTA" if use_tta else "no-TTA"
    model_desc = ", ".join(f"{cfg}/{fold}" for cfg, fold, _ in MODEL_CONFIGS)
    print(f"  Models: {model_desc} ({tta_status})")

    # Run all model configs in parallel
    result_queues = {(config, fold): Queue() for config, fold, _ in MODEL_CONFIGS}
    threads = []

    for config, fold, _ in MODEL_CONFIGS:
        predictor = predictors[(config, fold)]

        def process_model(pred, imgs, cfg, fld, queue):
            try:
                for img_path in imgs:
                    case_id = img_path.stem
                    probs = predict_single_case(pred, img_path)
                    queue.put((case_id, probs, None))
            except Exception as e:
                import traceback
                queue.put((None, None, f"{cfg}/{fld}: {e}\n{traceback.format_exc()}"))

        t = threading.Thread(
            target=process_model,
            args=(predictor, batch_images, config, fold, result_queues[(config, fold)])
        )
        threads.append(t)
        t.start()

    # Collect results
    model_results = {(cfg, fold): {} for cfg, fold, _ in MODEL_CONFIGS}  # (config, fold) -> {case_id: probs}
    errors = []

    for config, fold, _ in MODEL_CONFIGS:
        for _ in range(n_cases):
            case_id, probs, error = result_queues[(config, fold)].get()
            if error:
                errors.append(error)
            else:
                model_results[(config, fold)][case_id] = probs

    for t in threads:
        t.join()

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return False

    # Accumulate weighted probabilities
    for img_path in batch_images:
        case_id = img_path.stem
        for config, fold, _ in MODEL_CONFIGS:
            model_weight = MODEL_WEIGHTS[(config, fold)]
            weighted_probs = model_results[(config, fold)][case_id] * model_weight

            if batch_cumulative[case_id] is None:
                batch_cumulative[case_id] = weighted_probs
            else:
                batch_cumulative[case_id] += weighted_probs

    # Free memory for model results
    del model_results

    # Finalize: post-process (weights already sum to 1.0)
    for img_path in batch_images:
        case_id = img_path.stem
        # Weighted average (weights normalized to sum to 1.0)
        avg_probs = batch_cumulative[case_id]
        pred = postprocess_opening_closing(avg_probs)
        tifffile.imwrite(output_dir / f"{case_id}.tif", pred)

        # Record case completion time
        case_elapsed = time.time() - case_start_times[case_id]
        tta_controller.record_case_time(case_elapsed)
        print(f"    {case_id}: saved ({case_elapsed:.1f}s)")

        # Free memory
        del batch_cumulative[case_id]

    return True


def initialize_all_predictors() -> dict:
    """Initialize predictors for specific config/fold combinations.

    Returns:
        Dict mapping (config, fold) -> nnUNetPredictor with model loaded
    """
    predictors = {}

    for idx, (config, fold, _) in enumerate(MODEL_CONFIGS):
        gpu_id = idx % 2  # Distribute across 2 GPUs
        print(f"Loading model: {config}/fold_{fold} on GPU {gpu_id}")

        # TTA setting controlled by ENABLE_TTA flag
        predictor = create_predictor(gpu_id, use_tta=ENABLE_TTA)
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
    print("Vesuvius nnUNet Submission (Python API - Weighted Ensemble)")
    model_desc = ", ".join(f"{cfg}/fold_{fold}" for cfg, fold, _ in MODEL_CONFIGS)
    print(f"Models: {model_desc}")
    print(f"Total models: {len(MODEL_CONFIGS)}")
    print(f"Model weights: {MODEL_WEIGHTS}")
    print(f"TTA: {'adaptive' if ENABLE_TTA else 'disabled'}")
    print(f"Time limit: {KAGGLE_TIME_LIMIT_SECONDS/3600:.1f}h (safety margin: {SAFETY_MARGIN_SECONDS/60:.0f}min)")
    print(f"Tile step size: {TILE_STEP_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Post-processing: {'opening+closing' if USE_HYSTERESIS else 'argmax'}")
    print(f"Ensemble: weighted probability averaging")
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

    # Initialize adaptive TTA controller
    tta_controller = AdaptiveTTAController(total_cases=len(test_images))

    # Prepare output directory
    predictions_tiff_dir = OUTPUT_DIR / "predictions_tiff"
    predictions_tiff_dir.mkdir(parents=True, exist_ok=True)

    # Initialize all predictors (models loaded once)
    n_models = len(MODEL_CONFIGS)
    print(f"\nLoading {n_models} models...")
    predictors = initialize_all_predictors()
    print(f"All models loaded!")

    # Mark inference start (after model loading)
    tta_controller.mark_inference_start()

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
            tta_controller=tta_controller,
        )

        if not success:
            print(f"ERROR: Inference failed for batch {batch_idx + 1}!")
            return 1

    # Print TTA controller summary
    tta_controller.print_summary()

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

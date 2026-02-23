#!/usr/bin/env python3
"""
Ensemble Post-processing Evaluation Script

Evaluates post-processing methods on ensemble of lowres + fullres models.

Ensemble:
    probs = α * lowres + (1-α) * fullres

Usage:
    # Evaluate fold_0 with α=0.5
    python evaluate_ensemble_postprocess.py \
        --fold 0 \
        --alpha 0.5 \
        --workers 6

    # Evaluate all combinations (α=0.5, 0.4, 0.3) for fold_0
    python evaluate_ensemble_postprocess.py \
        --fold 0 \
        --alpha all \
        --workers 6

    # Quick test with 5 cases
    python evaluate_ensemble_postprocess.py \
        --fold 0 \
        --alpha 0.5 \
        --max-cases 5 \
        --workers 4
"""

import argparse
import csv
import gc
import os
import sys
from collections import defaultdict

# Set Numba threading layer before importing numba (TBB may not be compatible)
os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter, median_filter
from skimage.filters import threshold_otsu
from skimage.morphology import area_opening, reconstruction, remove_small_objects
from skimage.restoration import denoise_bilateral

# Numba for performance-critical functions (required)
from numba import njit, prange

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_T_HIGH = 0.85
DEFAULT_T_LOW = 0.3
DEFAULT_DUST_MIN_SIZE = 100


# =============================================================================
# PATHS
# =============================================================================

# Support both local and Docker paths
_LOCAL_BASE = Path("/home/ben/Dev/vesvius")
_DOCKER_BASE = Path("/workspace")

# Auto-detect: use Docker path if /workspace exists, otherwise local
if _DOCKER_BASE.exists() and (_DOCKER_BASE / "nnunet_output").exists():
    BASE_DIR = _DOCKER_BASE
else:
    BASE_DIR = _LOCAL_BASE

RESULTS_DIR = BASE_DIR / "nnunet_output" / "nnUNet_results" / "Dataset100_VesuviusSurface"
GT_DIR = BASE_DIR / "nnunet_output" / "nnUNet_preprocessed" / "Dataset100_VesuviusSurface" / "gt_segmentations"

LOWRES_DIR = RESULTS_DIR / "nnUNetTrainer_2000epochs__nnUNetResEncUNetMPlans__3d_lowres"
FULLRES_DIR = RESULTS_DIR / "nnUNetTrainer_2000epochs__nnUNetResEncUNetMPlans__3d_fullres"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@njit(parallel=True, cache=True)
def _fill_z_gaps(mask: np.ndarray, max_gap: int = 4) -> np.ndarray:
    """Numba-optimized Z-direction gap filling with parallel processing."""
    result = mask.copy()
    depth, height, width = mask.shape

    for y in prange(height):
        for x in range(width):
            # Collect non-zero Z indices manually (np.nonzero is slow in numba)
            nonzero_indices = np.empty(depth, dtype=np.int64)
            count = 0
            for z in range(depth):
                if mask[z, y, x]:
                    nonzero_indices[count] = z
                    count += 1

            # Fill gaps
            if count > 1:
                for i in range(count - 1):
                    gap = nonzero_indices[i + 1] - nonzero_indices[i]
                    if 1 < gap <= max_gap:
                        for z in range(nonzero_indices[i], nonzero_indices[i + 1]):
                            result[z, y, x] = True

    return result


def _apply_hysteresis(
    surface_prob: np.ndarray,
    t_low: float = DEFAULT_T_LOW,
    t_high: float = DEFAULT_T_HIGH,
) -> Optional[np.ndarray]:
    """Apply hysteresis thresholding with binary propagation.

    Args:
        surface_prob: Surface probability array (already preprocessed if needed)
        t_low: Low threshold for weak mask
        t_high: High threshold for strong mask

    Returns:
        Binary mask after hysteresis, or None if no strong regions found
    """
    strong = surface_prob >= t_high
    weak = surface_prob >= t_low

    if not strong.any():
        return None

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return None

    return mask


def _finalize_mask(
    mask: np.ndarray,
    z_radius: int = 1,
    xy_radius: int = 0,
    dust_min_size: int = DEFAULT_DUST_MIN_SIZE,
) -> np.ndarray:
    """Apply closing and dust removal to finalize a binary mask.

    Args:
        mask: Binary mask to finalize
        z_radius: Z-radius for closing structure
        xy_radius: XY-radius for closing structure
        dust_min_size: Minimum size for connected components

    Returns:
        Finalized binary mask as uint8
    """
    if z_radius > 0 or xy_radius > 0:
        struct_close = build_anisotropic_struct(z_radius, xy_radius)
        if struct_close is not None:
            mask = ndi.binary_closing(mask, structure=struct_close)

    if dust_min_size > 0:
        mask = remove_small_objects(mask.astype(bool), min_size=dust_min_size)

    return mask.astype(np.uint8)


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


# =============================================================================
# POST-PROCESSING METHODS
# =============================================================================

def postprocess_none(probs: np.ndarray) -> np.ndarray:
    """Argmax - no post-processing (baseline)."""
    return np.argmax(probs, axis=0).astype(np.uint8)


def postprocess_threshold(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Simple threshold on surface class probability."""
    surface_prob = probs[1]
    return (surface_prob >= threshold).astype(np.uint8)


def postprocess_hysteresis(
    probs: np.ndarray,
    t_low: float = DEFAULT_T_LOW,
    t_high: float = DEFAULT_T_HIGH,
    z_radius: int = 1,
    xy_radius: int = 0,
    dust_min_size: int = DEFAULT_DUST_MIN_SIZE,
) -> np.ndarray:
    """Hysteresis thresholding with closing."""
    surface_prob = probs[1]
    mask = _apply_hysteresis(surface_prob, t_low, t_high)
    if mask is None:
        return np.zeros(surface_prob.shape, dtype=np.uint8)
    return _finalize_mask(mask, z_radius, xy_radius, dust_min_size)


def postprocess_opening_closing(probs: np.ndarray) -> np.ndarray:
    """Hysteresis + opening + closing (current best)."""
    surface_prob = probs[1]
    mask = _apply_hysteresis(surface_prob)
    if mask is None:
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    # Additional opening step before closing
    struct_open = ndi.generate_binary_structure(3, 1)
    mask = ndi.binary_opening(mask, structure=struct_open)

    return _finalize_mask(mask, z_radius=2, xy_radius=1)


# --- Morphological variants ---

def postprocess_morph_closing_z2(probs: np.ndarray) -> np.ndarray:
    return postprocess_hysteresis(probs, z_radius=2, xy_radius=0)


def postprocess_morph_closing_z3(probs: np.ndarray) -> np.ndarray:
    return postprocess_hysteresis(probs, z_radius=3, xy_radius=0)


def postprocess_morph_closing_xy(probs: np.ndarray) -> np.ndarray:
    return postprocess_hysteresis(probs, z_radius=1, xy_radius=1)


def postprocess_morph_closing_full(probs: np.ndarray) -> np.ndarray:
    return postprocess_hysteresis(probs, z_radius=2, xy_radius=1)


def postprocess_hole_filling(probs: np.ndarray) -> np.ndarray:
    """Hysteresis + Z-direction hole filling."""
    surface_prob = probs[1]
    mask = _apply_hysteresis(surface_prob)
    if mask is None:
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    # Fill holes in each direction
    filled = np.zeros_like(mask)
    for z in range(mask.shape[0]):
        filled[z] = ndi.binary_fill_holes(mask[z])
    for y in range(mask.shape[1]):
        filled[:, y, :] = ndi.binary_fill_holes(filled[:, y, :])
    for x in range(mask.shape[2]):
        filled[:, :, x] = ndi.binary_fill_holes(filled[:, :, x])

    return _finalize_mask(filled, z_radius=1, xy_radius=0)


# --- Probability preprocessing variants ---

def postprocess_smoothed_hysteresis(probs: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian smoothing before hysteresis."""
    surface_prob = probs[1]
    smoothed = gaussian_filter(surface_prob, sigma=sigma)
    mask = _apply_hysteresis(smoothed)
    if mask is None:
        return np.zeros(surface_prob.shape, dtype=np.uint8)
    return _finalize_mask(mask, z_radius=1, xy_radius=0)


def postprocess_smooth_sigma05(probs: np.ndarray) -> np.ndarray:
    return postprocess_smoothed_hysteresis(probs, sigma=0.5)


def postprocess_smooth_sigma10(probs: np.ndarray) -> np.ndarray:
    return postprocess_smoothed_hysteresis(probs, sigma=1.0)


def postprocess_smooth_sigma15(probs: np.ndarray) -> np.ndarray:
    return postprocess_smoothed_hysteresis(probs, sigma=1.5)


def postprocess_z_accumulated(probs: np.ndarray) -> np.ndarray:
    """Z-direction probability accumulation."""
    surface_prob = probs[1].copy()

    kernel_z = np.array([0.15, 0.25, 0.2, 0.25, 0.15]).reshape(5, 1, 1)
    accumulated = ndi.convolve(surface_prob, kernel_z, mode='reflect')
    accumulated = np.clip(accumulated, 0, 1)

    mask = _apply_hysteresis(accumulated)
    if mask is None:
        return np.zeros(surface_prob.shape, dtype=np.uint8)
    return _finalize_mask(mask, z_radius=1, xy_radius=0)


def postprocess_anisotropic_smooth(probs: np.ndarray) -> np.ndarray:
    """Anisotropic Gaussian (more smoothing in Z)."""
    surface_prob = probs[1]
    smoothed = gaussian_filter(surface_prob, sigma=(1.5, 0.5, 0.5))
    mask = _apply_hysteresis(smoothed)
    if mask is None:
        return np.zeros(surface_prob.shape, dtype=np.uint8)
    return _finalize_mask(mask, z_radius=1, xy_radius=0)


# --- Connected component variants ---

def postprocess_dust_aggressive(probs: np.ndarray) -> np.ndarray:
    return postprocess_hysteresis(probs, dust_min_size=500)


def postprocess_dust_very_aggressive(probs: np.ndarray) -> np.ndarray:
    return postprocess_hysteresis(probs, dust_min_size=1000)


def postprocess_largest_components(probs: np.ndarray, keep_n: int = 10) -> np.ndarray:
    """Keep only the N largest connected components."""
    surface_prob = probs[1]
    mask = _apply_hysteresis(surface_prob)
    if mask is None:
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    # Apply closing
    struct_close = build_anisotropic_struct(1, 0)
    mask = ndi.binary_closing(mask, structure=struct_close)

    # Keep only N largest components
    labeled, num_features = ndi.label(mask)
    if num_features == 0:
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    component_sizes = ndi.sum(mask, labeled, range(1, num_features + 1))
    sorted_indices = np.argsort(component_sizes)[::-1]
    keep_labels = sorted_indices[:keep_n] + 1

    return np.isin(labeled, keep_labels).astype(np.uint8)


def postprocess_largest_5(probs: np.ndarray) -> np.ndarray:
    return postprocess_largest_components(probs, keep_n=5)


def postprocess_largest_10(probs: np.ndarray) -> np.ndarray:
    return postprocess_largest_components(probs, keep_n=10)


def postprocess_largest_20(probs: np.ndarray) -> np.ndarray:
    return postprocess_largest_components(probs, keep_n=20)


def postprocess_component_bridging(probs: np.ndarray) -> np.ndarray:
    """Bridge nearby connected components using dilation."""
    surface_prob = probs[1]
    mask = _apply_hysteresis(surface_prob)
    if mask is None:
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    # Dilate then erode to bridge gaps
    struct_dilate = build_anisotropic_struct(2, 1)
    dilated = ndi.binary_dilation(mask, structure=struct_dilate)

    struct_erode = build_anisotropic_struct(1, 0)
    mask = ndi.binary_erosion(dilated, structure=struct_erode)

    return _finalize_mask(mask, z_radius=1, xy_radius=0)


# --- Hysteresis parameter variants ---

def postprocess_hyst_t_low_02(probs: np.ndarray) -> np.ndarray:
    return postprocess_hysteresis(probs, t_low=0.2, t_high=0.85)


def postprocess_hyst_t_low_04(probs: np.ndarray) -> np.ndarray:
    return postprocess_hysteresis(probs, t_low=0.4, t_high=0.85)


def postprocess_hyst_t_high_08(probs: np.ndarray) -> np.ndarray:
    return postprocess_hysteresis(probs, t_low=0.3, t_high=0.8)


def postprocess_hyst_t_high_09(probs: np.ndarray) -> np.ndarray:
    return postprocess_hysteresis(probs, t_low=0.3, t_high=0.9)


def postprocess_hyst_wide_range(probs: np.ndarray) -> np.ndarray:
    return postprocess_hysteresis(probs, t_low=0.2, t_high=0.9)


# --- Sheet structure enhancement ---

def postprocess_z_continuity(probs: np.ndarray) -> np.ndarray:
    """Enforce Z-direction continuity by filling gaps between slices.

    Uses Numba-optimized implementation if available (up to 72x faster).
    """
    surface_prob = probs[1]
    mask = _apply_hysteresis(surface_prob)
    if mask is None:
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    # Fill small gaps in Z direction (uses Numba if available)
    result = _fill_z_gaps(mask, max_gap=4)

    return _finalize_mask(result, z_radius=1, xy_radius=0)


# --- NEW: Additional methods ---

def postprocess_adaptive_threshold(probs: np.ndarray) -> np.ndarray:
    """Adaptive thresholding based on local statistics."""
    surface_prob = probs[1]

    # Adaptive threshold: above local mean + offset
    local_mean = gaussian_filter(surface_prob, sigma=5)
    threshold = np.clip(local_mean + 0.1, DEFAULT_T_LOW, 0.9)

    mask = surface_prob >= threshold
    if not mask.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    return _finalize_mask(mask, z_radius=2, xy_radius=1)


def postprocess_otsu_threshold(probs: np.ndarray) -> np.ndarray:
    """Otsu's method for automatic thresholding."""
    surface_prob = probs[1]

    # Only compute Otsu on non-zero regions
    nonzero = surface_prob[surface_prob > 0.1]
    if len(nonzero) < 100:
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    try:
        thresh = threshold_otsu(nonzero)
    except ValueError:
        thresh = 0.5
    thresh = np.clip(thresh, DEFAULT_T_LOW, 0.9)

    mask = surface_prob >= thresh
    if not mask.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    return _finalize_mask(mask, z_radius=1, xy_radius=0)


def postprocess_bilateral_smooth(probs: np.ndarray) -> np.ndarray:
    """Bilateral-like filtering (edge-preserving smoothing)."""
    surface_prob = probs[1]

    # Apply bilateral filter (2D per slice for speed)
    smoothed = np.zeros_like(surface_prob)
    for z in range(surface_prob.shape[0]):
        smoothed[z] = denoise_bilateral(
            surface_prob[z].astype(np.float64),
            sigma_color=0.1,
            sigma_spatial=2,
            channel_axis=None,
        )

    mask = _apply_hysteresis(smoothed)
    if mask is None:
        return np.zeros(surface_prob.shape, dtype=np.uint8)
    return _finalize_mask(mask, z_radius=1, xy_radius=0)


def postprocess_median_smooth(probs: np.ndarray) -> np.ndarray:
    """Median filtering before hysteresis."""
    surface_prob = probs[1]
    smoothed = median_filter(surface_prob, size=3)
    mask = _apply_hysteresis(smoothed)
    if mask is None:
        return np.zeros(surface_prob.shape, dtype=np.uint8)
    return _finalize_mask(mask, z_radius=1, xy_radius=0)


def postprocess_morphological_reconstruction(probs: np.ndarray) -> np.ndarray:
    """Morphological reconstruction from high-confidence seeds."""
    surface_prob = probs[1]

    # High-confidence seeds
    seed = (surface_prob >= 0.9).astype(np.float64)
    if not seed.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    # Mask: lower threshold
    mask_img = (surface_prob >= DEFAULT_T_LOW).astype(np.float64)

    # Reconstruction by dilation
    reconstructed = reconstruction(seed, mask_img, method='dilation')
    result = (reconstructed >= 0.5).astype(bool)

    return _finalize_mask(result, z_radius=1, xy_radius=0)


def postprocess_area_opening(probs: np.ndarray) -> np.ndarray:
    """Area opening (remove small connected regions)."""
    surface_prob = probs[1]
    mask = _apply_hysteresis(surface_prob)
    if mask is None:
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    # Area opening (3D) instead of dust removal
    mask = area_opening(mask.astype(np.uint8), area_threshold=200, connectivity=1)

    return _finalize_mask(mask.astype(bool), z_radius=2, xy_radius=1, dust_min_size=0)


def postprocess_z_sliding_avg(probs: np.ndarray) -> np.ndarray:
    """Z-direction sliding window average."""
    surface_prob = probs[1].copy()

    # Sliding average in Z direction (window=5)
    kernel = np.ones((5, 1, 1)) / 5
    smoothed = ndi.convolve(surface_prob, kernel, mode='reflect')

    mask = _apply_hysteresis(smoothed)
    if mask is None:
        return np.zeros(surface_prob.shape, dtype=np.uint8)
    return _finalize_mask(mask, z_radius=1, xy_radius=0)


# =============================================================================
# POSTPROCESS REGISTRY
# =============================================================================

POSTPROCESS_METHODS = {
    # Baseline
    "none": postprocess_none,
    "threshold_05": partial(postprocess_threshold, threshold=0.5),
    "threshold_075": partial(postprocess_threshold, threshold=0.75),

    # Current best methods
    "hysteresis": partial(postprocess_hysteresis, t_low=DEFAULT_T_LOW, t_high=DEFAULT_T_HIGH),
    "opening_closing": postprocess_opening_closing,

    # Morphological variants
    "morph_closing_z2": postprocess_morph_closing_z2,
    "morph_closing_z3": postprocess_morph_closing_z3,
    "morph_closing_xy": postprocess_morph_closing_xy,
    "morph_closing_full": postprocess_morph_closing_full,
    "hole_filling": postprocess_hole_filling,

    # Probability preprocessing
    "smooth_sigma05": postprocess_smooth_sigma05,
    "smooth_sigma10": postprocess_smooth_sigma10,
    "smooth_sigma15": postprocess_smooth_sigma15,
    "z_accumulated": postprocess_z_accumulated,
    "anisotropic_smooth": postprocess_anisotropic_smooth,

    # Connected component variants
    "dust_aggressive": postprocess_dust_aggressive,
    "dust_very_aggressive": postprocess_dust_very_aggressive,
    "largest_5": postprocess_largest_5,
    "largest_10": postprocess_largest_10,
    "largest_20": postprocess_largest_20,
    "component_bridging": postprocess_component_bridging,

    # Hysteresis parameter variants
    "hyst_t_low_02": postprocess_hyst_t_low_02,
    "hyst_t_low_04": postprocess_hyst_t_low_04,
    "hyst_t_high_08": postprocess_hyst_t_high_08,
    "hyst_t_high_09": postprocess_hyst_t_high_09,
    "hyst_wide_range": postprocess_hyst_wide_range,

    # Sheet structure
    "z_continuity": postprocess_z_continuity,

    # NEW methods
    "adaptive_threshold": postprocess_adaptive_threshold,
    "otsu_threshold": postprocess_otsu_threshold,
    "bilateral_smooth": postprocess_bilateral_smooth,
    "median_smooth": postprocess_median_smooth,
    "morph_reconstruction": postprocess_morphological_reconstruction,
    "area_opening": postprocess_area_opening,
    "z_sliding_avg": postprocess_z_sliding_avg,
}


# =============================================================================
# ENSEMBLE LOADING
# =============================================================================

def load_ensemble_probs(
    case_id: str,
    fold: int,
    alpha: float,
) -> Optional[np.ndarray]:
    """
    Load ensemble probabilities: α * lowres + (1-α) * fullres

    Args:
        case_id: Case identifier
        fold: Fold number (0 or 1)
        alpha: Weight for lowres model

    Returns:
        Ensemble probabilities or None if files not found
    """
    lowres_npz = LOWRES_DIR / f"fold_{fold}" / "validation_npz" / f"{case_id}.npz"
    fullres_npz = FULLRES_DIR / f"fold_{fold}" / "validation_npz" / f"{case_id}.npz"

    if not lowres_npz.exists() or not fullres_npz.exists():
        return None

    # Use context manager to properly close NpzFile and free memory
    with np.load(lowres_npz) as lowres_data:
        lowres_probs = lowres_data['probabilities'].copy()
    with np.load(fullres_npz) as fullres_data:
        fullres_probs = fullres_data['probabilities'].copy()

    # Ensemble: α * lowres + (1-α) * fullres
    ensemble_probs = alpha * lowres_probs + (1 - alpha) * fullres_probs

    # Explicitly free memory
    del lowres_probs, fullres_probs

    return ensemble_probs


def get_case_ids(fold: int) -> List[str]:
    """Get case IDs for a fold."""
    npz_dir = LOWRES_DIR / f"fold_{fold}" / "validation_npz"
    npz_files = sorted(npz_dir.glob("*.npz"))
    return [f.stem for f in npz_files]


# =============================================================================
# EVALUATION
# =============================================================================

def compute_single_case(
    case_id: str,
    fold: int,
    alpha: float,
    postprocess_fn: Callable,
    postprocess_name: str,
) -> dict:
    """Compute metrics for a single case."""
    from topometrics import compute_leaderboard_score

    gt_path = GT_DIR / f"{case_id}.tif"

    if not gt_path.exists():
        return {"case": case_id, "fold": fold, "alpha": alpha,
                "postprocess": postprocess_name, "error": "GT not found"}

    probs = None
    pred = None
    gt = None
    rep = None

    try:
        probs = load_ensemble_probs(case_id, fold, alpha)
        if probs is None:
            return {"case": case_id, "fold": fold, "alpha": alpha,
                    "postprocess": postprocess_name, "error": "NPZ not found"}

        pred = postprocess_fn(probs)

        # Free probs immediately after postprocessing
        del probs
        probs = None

        gt = tifffile.imread(gt_path)

        if pred.shape != gt.shape:
            error_msg = f"Shape mismatch: pred={pred.shape}, gt={gt.shape}"
            del pred, gt
            return {"case": case_id, "fold": fold, "alpha": alpha,
                    "postprocess": postprocess_name,
                    "error": error_msg}

        rep = compute_leaderboard_score(
            predictions=pred,
            labels=gt,
            dims=(0, 1, 2),
            spacing=(1.0, 1.0, 1.0),
            surface_tolerance=2.0,
            voi_connectivity=26,
            voi_transform="one_over_one_plus",
            voi_alpha=0.3,
            combine_weights=(0.3, 0.35, 0.35),
            fg_threshold=None,
            ignore_label=2,
            ignore_mask=None,
        )

        # Extract values before cleanup
        result = {
            "case": case_id,
            "fold": fold,
            "alpha": alpha,
            "postprocess": postprocess_name,
            "leaderboard": float(rep.score),
            "toposcore": float(rep.topo.toposcore) if rep.topo else np.nan,
            "surface_dice": float(rep.surface_dice),
            "voi_score": float(rep.voi.voi_score) if rep.voi else np.nan,
        }

        return result

    except Exception as e:
        return {"case": case_id, "fold": fold, "alpha": alpha,
                "postprocess": postprocess_name, "error": str(e)}

    finally:
        # Explicit cleanup to prevent memory leaks in worker processes
        del probs, pred, gt, rep
        gc.collect()


def _compute_wrapper(args):
    return compute_single_case(*args)


def evaluate_ensemble(
    fold: int,
    alphas: List[float],
    methods: List[str],
    workers: int = 4,
    max_cases: Optional[int] = None,
    output_csv: Optional[Path] = None,
) -> List[dict]:
    """Evaluate ensemble with multiple α values and post-processing methods."""
    from tqdm.auto import tqdm

    case_ids = get_case_ids(fold)
    if max_cases:
        case_ids = case_ids[:max_cases]

    print(f"Fold: {fold}")
    print(f"Cases: {len(case_ids)}")
    print(f"Alpha values: {alphas}")
    print(f"Methods: {len(methods)}")

    args_list = []
    for alpha in alphas:
        for method_name in methods:
            postprocess_fn = POSTPROCESS_METHODS[method_name]
            for case_id in case_ids:
                args_list.append((
                    case_id, fold, alpha,
                    postprocess_fn, method_name
                ))

    print(f"Total evaluations: {len(args_list)}")

    # Prepare streaming CSV output to avoid memory accumulation
    csv_file = None
    csv_writer = None
    fieldnames = ["fold", "alpha", "postprocess", "case", "leaderboard", "toposcore", "surface_dice", "voi_score", "error"]

    if output_csv:
        csv_file = open(output_csv, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    # Use running aggregation instead of storing all results
    # Key: (alpha, postprocess) -> list of scores
    aggregated: Dict[Tuple[float, str], Dict[str, list]] = defaultdict(
        lambda: {"leaderboard": [], "toposcore": [], "surface_dice": [], "voi_score": []}
    )

    try:
        if workers > 1:
            # Reduce maxtasksperchild to release memory more frequently
            with Pool(workers, maxtasksperchild=10) as pool:
                for i, result in enumerate(tqdm(
                    pool.imap_unordered(_compute_wrapper, args_list),
                    total=len(args_list),
                    desc=f"Evaluating fold_{fold}"
                )):
                    # Stream results to CSV immediately
                    if csv_writer:
                        row = {k: result.get(k, "") for k in fieldnames}
                        csv_writer.writerow(row)
                        # Flush periodically
                        if i % 100 == 0:
                            csv_file.flush()

                    # Aggregate scores instead of storing all results
                    if "error" not in result:
                        key = (result["alpha"], result["postprocess"])
                        aggregated[key]["leaderboard"].append(result["leaderboard"])
                        if not np.isnan(result.get("toposcore", np.nan)):
                            aggregated[key]["toposcore"].append(result["toposcore"])
                        if not np.isnan(result.get("surface_dice", np.nan)):
                            aggregated[key]["surface_dice"].append(result["surface_dice"])
                        if not np.isnan(result.get("voi_score", np.nan)):
                            aggregated[key]["voi_score"].append(result["voi_score"])

                    # Force garbage collection more frequently
                    if i % 100 == 0:
                        gc.collect()
        else:
            for i, args in enumerate(tqdm(args_list)):
                result = compute_single_case(*args)
                if csv_writer:
                    row = {k: result.get(k, "") for k in fieldnames}
                    csv_writer.writerow(row)
                if "error" not in result:
                    key = (result["alpha"], result["postprocess"])
                    aggregated[key]["leaderboard"].append(result["leaderboard"])
                    if not np.isnan(result.get("toposcore", np.nan)):
                        aggregated[key]["toposcore"].append(result["toposcore"])
                    if not np.isnan(result.get("surface_dice", np.nan)):
                        aggregated[key]["surface_dice"].append(result["surface_dice"])
                    if not np.isnan(result.get("voi_score", np.nan)):
                        aggregated[key]["voi_score"].append(result["voi_score"])
                if i % 50 == 0:
                    gc.collect()

    finally:
        if csv_file:
            csv_file.close()
            print(f"Results streamed to: {output_csv}")

    # Convert aggregated data to results format for compatibility
    results = []
    for (alpha, postprocess), scores in aggregated.items():
        results.append({
            "alpha": alpha,
            "postprocess": postprocess,
            "leaderboard": np.mean(scores["leaderboard"]) if scores["leaderboard"] else np.nan,
            "toposcore": np.mean(scores["toposcore"]) if scores["toposcore"] else np.nan,
            "surface_dice": np.mean(scores["surface_dice"]) if scores["surface_dice"] else np.nan,
            "voi_score": np.mean(scores["voi_score"]) if scores["voi_score"] else np.nan,
        })

    return results


def print_summary(results: List[dict]):
    """Print summary by alpha and method.

    Args:
        results: List of already-aggregated results with alpha, postprocess, and mean scores.
    """
    print(f"\n{'='*100}")
    print("ENSEMBLE POST-PROCESSING COMPARISON SUMMARY")
    print(f"{'='*100}")

    # Get unique alphas
    alphas = sorted(set(r["alpha"] for r in results))

    for alpha in alphas:
        print(f"\n{'─'*100}")
        print(f"α = {alpha} (lowres weight)")
        print(f"{'─'*100}")
        print(f"{'Method':<25} {'Leaderboard':>12} {'TopoScore':>12} {'SurfDice':>12} {'VOI':>12}")
        print("-" * 80)

        # Filter results for this alpha
        method_scores = []
        for r in results:
            if r["alpha"] == alpha:
                method_scores.append((
                    r["postprocess"],
                    r["leaderboard"],
                    r["toposcore"],
                    r["surface_dice"],
                    r["voi_score"],
                ))

        # Sort by leaderboard
        method_scores.sort(key=lambda x: x[1] if not np.isnan(x[1]) else -1, reverse=True)

        for method, lb, topo, sd, voi in method_scores:
            lb_str = f"{lb:.4f}" if not np.isnan(lb) else "N/A"
            topo_str = f"{topo:.4f}" if not np.isnan(topo) else "N/A"
            sd_str = f"{sd:.4f}" if not np.isnan(sd) else "N/A"
            voi_str = f"{voi:.4f}" if not np.isnan(voi) else "N/A"
            print(f"{method:<25} {lb_str:>12} {topo_str:>12} {sd_str:>12} {voi_str:>12}")

        print("\nTOP 5:")
        for i, (method, lb, topo, sd, voi) in enumerate(method_scores[:5], 1):
            lb_str = f"{lb:.4f}" if not np.isnan(lb) else "N/A"
            print(f"  {i}. {method}: {lb_str}")

    # Overall best
    print(f"\n{'='*100}")
    print("OVERALL BEST COMBINATIONS:")
    print(f"{'='*100}")

    all_scores = [(r["alpha"], r["postprocess"], r["leaderboard"], r["toposcore"]) for r in results]
    all_scores.sort(key=lambda x: x[2] if not np.isnan(x[2]) else -1, reverse=True)

    print(f"{'α':<6} {'Method':<25} {'Leaderboard':>12} {'TopoScore':>12}")
    print("-" * 60)
    for alpha, method, lb, topo in all_scores[:10]:
        lb_str = f"{lb:.4f}" if not np.isnan(lb) else "N/A"
        topo_str = f"{topo:.4f}" if not np.isnan(topo) else "N/A"
        print(f"{alpha:<6} {method:<25} {lb_str:>12} {topo_str:>12}")


def save_summary(results: List[dict], output_path: Path, fold: int):
    """Save aggregated summary to CSV.

    Args:
        results: List of already-aggregated result dicts with alpha, postprocess, and mean scores.
        output_path: Path to save CSV
        fold: Fold number to include in output
    """
    if not results:
        print("No results to summarize!")
        return

    # Results are already aggregated, just add fold info
    summary = []
    for r in results:
        summary.append({
            "fold": fold,
            "alpha": r["alpha"],
            "postprocess": r["postprocess"],
            "leaderboard": r["leaderboard"],
            "toposcore": r["toposcore"],
            "surface_dice": r["surface_dice"],
            "voi_score": r["voi_score"],
        })

    # Sort by leaderboard descending
    summary.sort(key=lambda x: -x["leaderboard"] if not np.isnan(x["leaderboard"]) else float('inf'))

    fieldnames = ["fold", "alpha", "postprocess", "leaderboard", "toposcore", "surface_dice", "voi_score"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    print(f"Summary saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ensemble post-processing")

    parser.add_argument("--fold", type=int, required=True, choices=[0, 1],
                        help="Fold to evaluate (0 or 1)")
    parser.add_argument("--alpha", type=str, default="all",
                        help="Alpha value(s): 'all' for [0.5, 0.4, 0.3] or comma-separated values")
    parser.add_argument("--methods", type=str, default="all",
                        help="Methods: 'all' or comma-separated names")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "docs" / "results")

    args = parser.parse_args()

    # Parse alpha
    if args.alpha == "all":
        alphas = [0.5, 0.4, 0.3]
    else:
        alphas = [float(a.strip()) for a in args.alpha.split(",")]

    # Parse methods
    if args.methods == "all":
        methods = list(POSTPROCESS_METHODS.keys())
    else:
        methods = [m.strip() for m in args.methods.split(",")]

    # Prepare output paths
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = args.output_dir / f"ensemble_fold{args.fold}_{timestamp}_detail.csv"
    summary_path = args.output_dir / f"ensemble_fold{args.fold}_{timestamp}_summary.csv"

    print(f"{'='*60}")
    print("ENSEMBLE POST-PROCESSING EVALUATION")
    print(f"{'='*60}")
    print(f"Fold: {args.fold}")
    print(f"Alpha values: {alphas}")
    print(f"Methods: {len(methods)}")
    print(f"Workers: {args.workers}")
    if args.max_cases:
        print(f"Max cases: {args.max_cases}")
    print(f"Output: {detail_path}")
    print(f"{'='*60}\n")

    # Run evaluation with streaming CSV output
    results = evaluate_ensemble(
        fold=args.fold,
        alphas=alphas,
        methods=methods,
        workers=args.workers,
        max_cases=args.max_cases,
        output_csv=detail_path,
    )

    # Print summary
    print_summary(results)

    # Save summary (aggregated by alpha/method)
    save_summary(results, summary_path, args.fold)

    return 0


if __name__ == "__main__":
    sys.exit(main())

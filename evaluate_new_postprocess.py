#!/usr/bin/env python3
"""
New Post-processing Methods Evaluation Script

Tests various new post-processing approaches for Vesuvius surface detection.
"""

import argparse
import csv
import sys
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Optional, List, Tuple, Callable
import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter


# =============================================================================
# HELPER FUNCTIONS
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


# =============================================================================
# BASELINE METHODS
# =============================================================================

def postprocess_none(probs: np.ndarray) -> np.ndarray:
    """Argmax - no post-processing (baseline)."""
    return np.argmax(probs, axis=0).astype(np.uint8)


def postprocess_hysteresis(
    probs: np.ndarray,
    t_low: float = 0.3,
    t_high: float = 0.85,
    z_radius: int = 1,
    xy_radius: int = 0,
    dust_min_size: int = 100
) -> np.ndarray:
    """Current best: hysteresis thresholding with closing."""
    from skimage.morphology import remove_small_objects

    surface_prob = probs[1]

    # Hysteresis thresholding
    strong = surface_prob >= t_high
    weak = surface_prob >= t_low

    if not strong.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    # Anisotropic closing
    if z_radius > 0 or xy_radius > 0:
        struct_close = build_anisotropic_struct(z_radius, xy_radius)
        if struct_close is not None:
            mask = ndi.binary_closing(mask, structure=struct_close)

    # Dust removal
    if dust_min_size > 0:
        mask = remove_small_objects(mask.astype(bool), min_size=dust_min_size)

    return mask.astype(np.uint8)


# =============================================================================
# 1. MORPHOLOGICAL OPERATIONS ENHANCEMENT
# =============================================================================

def postprocess_hysteresis_large_closing_z2(probs: np.ndarray) -> np.ndarray:
    """Hysteresis with larger Z closing (z_radius=2)."""
    return postprocess_hysteresis(probs, t_low=0.3, t_high=0.85, z_radius=2, xy_radius=0)


def postprocess_hysteresis_large_closing_z3(probs: np.ndarray) -> np.ndarray:
    """Hysteresis with larger Z closing (z_radius=3)."""
    return postprocess_hysteresis(probs, t_low=0.3, t_high=0.85, z_radius=3, xy_radius=0)


def postprocess_hysteresis_xy_closing(probs: np.ndarray) -> np.ndarray:
    """Hysteresis with XY closing added (z=1, xy=1)."""
    return postprocess_hysteresis(probs, t_low=0.3, t_high=0.85, z_radius=1, xy_radius=1)


def postprocess_hysteresis_full_closing(probs: np.ndarray) -> np.ndarray:
    """Hysteresis with both Z and XY closing (z=2, xy=1)."""
    return postprocess_hysteresis(probs, t_low=0.3, t_high=0.85, z_radius=2, xy_radius=1)


def postprocess_hole_filling(probs: np.ndarray) -> np.ndarray:
    """Hysteresis + Z-direction hole filling."""
    from skimage.morphology import remove_small_objects

    surface_prob = probs[1]

    # Hysteresis
    strong = surface_prob >= 0.85
    weak = surface_prob >= 0.3

    if not strong.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    # Fill holes in each XY slice
    filled = np.zeros_like(mask)
    for z in range(mask.shape[0]):
        filled[z] = ndi.binary_fill_holes(mask[z])

    # Also fill in YZ and XZ directions
    for y in range(mask.shape[1]):
        filled[:, y, :] = ndi.binary_fill_holes(filled[:, y, :])
    for x in range(mask.shape[2]):
        filled[:, :, x] = ndi.binary_fill_holes(filled[:, :, x])

    # Closing
    struct_close = build_anisotropic_struct(1, 0)
    mask = ndi.binary_closing(filled, structure=struct_close)

    # Dust removal
    mask = remove_small_objects(mask.astype(bool), min_size=100)

    return mask.astype(np.uint8)


def postprocess_opening_closing(probs: np.ndarray) -> np.ndarray:
    """Hysteresis + opening + closing to remove noise and fill gaps."""
    from skimage.morphology import remove_small_objects

    surface_prob = probs[1]

    # Hysteresis
    strong = surface_prob >= 0.85
    weak = surface_prob >= 0.3

    if not strong.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    # Opening (remove small protrusions)
    struct_open = ndi.generate_binary_structure(3, 1)
    mask = ndi.binary_opening(mask, structure=struct_open)

    # Closing (fill small holes)
    struct_close = build_anisotropic_struct(2, 1)
    mask = ndi.binary_closing(mask, structure=struct_close)

    # Dust removal
    mask = remove_small_objects(mask.astype(bool), min_size=100)

    return mask.astype(np.uint8)


# =============================================================================
# 2. PROBABILITY MAP PREPROCESSING
# =============================================================================

def postprocess_smoothed_hysteresis(probs: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian smoothing before hysteresis."""
    from skimage.morphology import remove_small_objects

    surface_prob = probs[1]

    # Gaussian smoothing
    smoothed = gaussian_filter(surface_prob, sigma=sigma)

    # Hysteresis
    strong = smoothed >= 0.85
    weak = smoothed >= 0.3

    if not strong.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    # Closing
    struct_close = build_anisotropic_struct(1, 0)
    mask = ndi.binary_closing(mask, structure=struct_close)

    # Dust removal
    mask = remove_small_objects(mask.astype(bool), min_size=100)

    return mask.astype(np.uint8)


def postprocess_smoothed_sigma05(probs: np.ndarray) -> np.ndarray:
    return postprocess_smoothed_hysteresis(probs, sigma=0.5)


def postprocess_smoothed_sigma10(probs: np.ndarray) -> np.ndarray:
    return postprocess_smoothed_hysteresis(probs, sigma=1.0)


def postprocess_smoothed_sigma15(probs: np.ndarray) -> np.ndarray:
    return postprocess_smoothed_hysteresis(probs, sigma=1.5)


def postprocess_z_accumulated(probs: np.ndarray) -> np.ndarray:
    """Z-direction probability accumulation for continuity."""
    from skimage.morphology import remove_small_objects

    surface_prob = probs[1].copy()

    # Accumulate probability from neighboring Z slices
    kernel_z = np.array([0.15, 0.25, 0.2, 0.25, 0.15]).reshape(5, 1, 1)
    accumulated = ndi.convolve(surface_prob, kernel_z, mode='reflect')

    # Normalize back to [0, 1]
    accumulated = np.clip(accumulated, 0, 1)

    # Hysteresis on accumulated
    strong = accumulated >= 0.85
    weak = accumulated >= 0.3

    if not strong.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    # Closing
    struct_close = build_anisotropic_struct(1, 0)
    mask = ndi.binary_closing(mask, structure=struct_close)

    # Dust removal
    mask = remove_small_objects(mask.astype(bool), min_size=100)

    return mask.astype(np.uint8)


def postprocess_anisotropic_smooth(probs: np.ndarray) -> np.ndarray:
    """Anisotropic Gaussian (more smoothing in Z)."""
    from skimage.morphology import remove_small_objects

    surface_prob = probs[1]

    # Anisotropic Gaussian (sigma_z=1.5, sigma_xy=0.5)
    smoothed = gaussian_filter(surface_prob, sigma=(1.5, 0.5, 0.5))

    # Hysteresis
    strong = smoothed >= 0.85
    weak = smoothed >= 0.3

    if not strong.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    # Closing
    struct_close = build_anisotropic_struct(1, 0)
    mask = ndi.binary_closing(mask, structure=struct_close)

    # Dust removal
    mask = remove_small_objects(mask.astype(bool), min_size=100)

    return mask.astype(np.uint8)


# =============================================================================
# 3. CONNECTED COMPONENT ANALYSIS IMPROVEMENTS
# =============================================================================

def postprocess_aggressive_dust(probs: np.ndarray) -> np.ndarray:
    """Hysteresis with more aggressive dust removal (min_size=500)."""
    return postprocess_hysteresis(probs, t_low=0.3, t_high=0.85, z_radius=1, xy_radius=0, dust_min_size=500)


def postprocess_very_aggressive_dust(probs: np.ndarray) -> np.ndarray:
    """Hysteresis with very aggressive dust removal (min_size=1000)."""
    return postprocess_hysteresis(probs, t_low=0.3, t_high=0.85, z_radius=1, xy_radius=0, dust_min_size=1000)


def postprocess_component_bridging(probs: np.ndarray) -> np.ndarray:
    """Bridge nearby connected components using dilation."""
    from skimage.morphology import remove_small_objects

    surface_prob = probs[1]

    # Hysteresis
    strong = surface_prob >= 0.85
    weak = surface_prob >= 0.3

    if not strong.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    # Dilate to bridge nearby components
    struct_dilate = build_anisotropic_struct(2, 1)
    dilated = ndi.binary_dilation(mask, structure=struct_dilate)

    # Erode back (but keeps bridges)
    struct_erode = build_anisotropic_struct(1, 0)
    mask = ndi.binary_erosion(dilated, structure=struct_erode)

    # Closing
    struct_close = build_anisotropic_struct(1, 0)
    mask = ndi.binary_closing(mask, structure=struct_close)

    # Dust removal
    mask = remove_small_objects(mask.astype(bool), min_size=100)

    return mask.astype(np.uint8)


def postprocess_largest_components(probs: np.ndarray, keep_n: int = 10) -> np.ndarray:
    """Keep only the N largest connected components."""
    from skimage.morphology import remove_small_objects

    surface_prob = probs[1]

    # Hysteresis
    strong = surface_prob >= 0.85
    weak = surface_prob >= 0.3

    if not strong.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    # Closing first
    struct_close = build_anisotropic_struct(1, 0)
    mask = ndi.binary_closing(mask, structure=struct_close)

    # Label connected components
    labeled, num_features = ndi.label(mask)
    if num_features == 0:
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    # Get component sizes
    component_sizes = ndi.sum(mask, labeled, range(1, num_features + 1))

    # Keep only largest N
    sorted_indices = np.argsort(component_sizes)[::-1]
    keep_labels = sorted_indices[:keep_n] + 1

    result = np.isin(labeled, keep_labels).astype(np.uint8)

    return result


def postprocess_largest_5(probs: np.ndarray) -> np.ndarray:
    return postprocess_largest_components(probs, keep_n=5)


def postprocess_largest_10(probs: np.ndarray) -> np.ndarray:
    return postprocess_largest_components(probs, keep_n=10)


def postprocess_largest_20(probs: np.ndarray) -> np.ndarray:
    return postprocess_largest_components(probs, keep_n=20)


# =============================================================================
# 4. FRANGI FILTER REVISITED
# =============================================================================

def compute_hessian_eigenvalues(volume: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute sorted eigenvalues of Hessian matrix."""
    # Smooth
    smoothed = gaussian_filter(volume.astype(np.float64), sigma=sigma)

    # Compute Hessian components
    Hzz = gaussian_filter(smoothed, sigma=sigma, order=[2, 0, 0])
    Hyy = gaussian_filter(smoothed, sigma=sigma, order=[0, 2, 0])
    Hxx = gaussian_filter(smoothed, sigma=sigma, order=[0, 0, 2])
    Hzy = gaussian_filter(smoothed, sigma=sigma, order=[1, 1, 0])
    Hzx = gaussian_filter(smoothed, sigma=sigma, order=[1, 0, 1])
    Hyx = gaussian_filter(smoothed, sigma=sigma, order=[0, 1, 1])

    # For each voxel, compute eigenvalues of 3x3 symmetric Hessian
    # Using analytical formula for eigenvalues
    shape = volume.shape
    lambda1 = np.zeros(shape, dtype=np.float64)
    lambda2 = np.zeros(shape, dtype=np.float64)
    lambda3 = np.zeros(shape, dtype=np.float64)

    # Compute eigenvalues using numpy's linalg (slower but accurate)
    # For speed, we'll use a simplified approach based on invariants
    # Trace
    trace = Hzz + Hyy + Hxx

    # For sheet-like structures, we expect:
    # - Two small eigenvalues (flat in two directions)
    # - One large eigenvalue (perpendicular to sheet)

    # Simplified: use directional second derivatives as proxy
    return Hzz, Hyy, Hxx


def postprocess_frangi_surfaceness(probs: np.ndarray) -> np.ndarray:
    """Improved Frangi-like surfaceness filter."""
    from skimage.morphology import remove_small_objects

    surface_prob = probs[1]

    # First, apply hysteresis to get initial mask
    strong = surface_prob >= 0.85
    weak = surface_prob >= 0.3

    if not strong.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    # Compute surfaceness response
    response = np.zeros_like(surface_prob, dtype=np.float64)

    for sigma in [1.0, 2.0]:
        # Hessian eigenvalues (simplified)
        Hzz, Hyy, Hxx = compute_hessian_eigenvalues(surface_prob, sigma)

        # Surfaceness: high response when one eigenvalue dominates
        # (sheet has one large eigenvalue, two small)
        abs_hessian = np.abs(Hzz) + np.abs(Hyy) + np.abs(Hxx)
        max_hessian = np.maximum(np.maximum(np.abs(Hzz), np.abs(Hyy)), np.abs(Hxx))

        # Ratio of max to total (high for sheets)
        with np.errstate(divide='ignore', invalid='ignore'):
            surfaceness = max_hessian / (abs_hessian + 1e-10)

        response = np.maximum(response, surfaceness * surface_prob)

    # Combine with probability
    combined = surface_prob * (1 + response)
    combined = np.clip(combined, 0, 1)

    # Threshold
    result = (combined >= 0.5) & mask

    # Closing
    struct_close = build_anisotropic_struct(1, 0)
    result = ndi.binary_closing(result, structure=struct_close)

    # Dust removal
    result = remove_small_objects(result.astype(bool), min_size=100)

    return result.astype(np.uint8)


def postprocess_frangi_then_hysteresis(probs: np.ndarray) -> np.ndarray:
    """Apply Frangi enhancement then hysteresis."""
    from skimage.morphology import remove_small_objects

    surface_prob = probs[1]

    # Frangi-like enhancement
    enhanced = np.zeros_like(surface_prob, dtype=np.float64)

    for sigma in [1.0, 1.5, 2.0]:
        smoothed = gaussian_filter(surface_prob, sigma=sigma)

        # Second derivatives
        Hzz = gaussian_filter(smoothed, sigma=sigma, order=[2, 0, 0])
        Hyy = gaussian_filter(smoothed, sigma=sigma, order=[0, 2, 0])
        Hxx = gaussian_filter(smoothed, sigma=sigma, order=[0, 0, 2])

        # Response: where curvature is high
        response = np.abs(Hzz) + np.abs(Hyy) + np.abs(Hxx)
        response = response * (surface_prob > 0.1)

        if response.max() > 0:
            response = response / response.max()

        enhanced = np.maximum(enhanced, response * surface_prob)

    # Boost original probability with enhancement
    boosted = surface_prob + 0.3 * enhanced
    boosted = np.clip(boosted, 0, 1)

    # Hysteresis on boosted
    strong = boosted >= 0.85
    weak = boosted >= 0.3

    if not strong.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    # Closing
    struct_close = build_anisotropic_struct(1, 0)
    mask = ndi.binary_closing(mask, structure=struct_close)

    # Dust removal
    mask = remove_small_objects(mask.astype(bool), min_size=100)

    return mask.astype(np.uint8)


# =============================================================================
# 5. SHEET STRUCTURE ENHANCEMENT
# =============================================================================

def postprocess_thinning(probs: np.ndarray) -> np.ndarray:
    """Apply morphological thinning to create thin sheets."""
    from skimage.morphology import remove_small_objects, thin

    surface_prob = probs[1]

    # Hysteresis
    strong = surface_prob >= 0.85
    weak = surface_prob >= 0.3

    if not strong.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    # Closing first
    struct_close = build_anisotropic_struct(1, 0)
    mask = ndi.binary_closing(mask, structure=struct_close)

    # Thin each XY slice (2D thinning)
    thinned = np.zeros_like(mask, dtype=bool)
    for z in range(mask.shape[0]):
        if mask[z].any():
            thinned[z] = thin(mask[z])

    # Slight dilation to recover some thickness
    struct_dilate = ndi.generate_binary_structure(3, 1)
    result = ndi.binary_dilation(thinned, structure=struct_dilate)

    # Dust removal
    result = remove_small_objects(result.astype(bool), min_size=50)

    return result.astype(np.uint8)


def postprocess_z_continuity(probs: np.ndarray) -> np.ndarray:
    """Enforce Z-direction continuity by filling gaps between slices."""
    from skimage.morphology import remove_small_objects

    surface_prob = probs[1]

    # Hysteresis
    strong = surface_prob >= 0.85
    weak = surface_prob >= 0.3

    if not strong.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    # For each XY position, fill Z gaps if surrounded by surface
    result = mask.copy()
    for y in range(mask.shape[1]):
        for x in range(mask.shape[2]):
            z_line = mask[:, y, x]
            if z_line.any():
                # Find first and last true
                nonzero = np.nonzero(z_line)[0]
                if len(nonzero) > 1:
                    # Fill gaps of size <= 3
                    for i in range(len(nonzero) - 1):
                        gap = nonzero[i + 1] - nonzero[i]
                        if 1 < gap <= 4:
                            result[nonzero[i]:nonzero[i + 1], y, x] = True

    # Closing
    struct_close = build_anisotropic_struct(1, 0)
    result = ndi.binary_closing(result, structure=struct_close)

    # Dust removal
    result = remove_small_objects(result.astype(bool), min_size=100)

    return result.astype(np.uint8)


# =============================================================================
# 6. HYSTERESIS PARAMETER VARIATIONS
# =============================================================================

def postprocess_hysteresis_t_low_02(probs: np.ndarray) -> np.ndarray:
    return postprocess_hysteresis(probs, t_low=0.2, t_high=0.85)


def postprocess_hysteresis_t_low_04(probs: np.ndarray) -> np.ndarray:
    return postprocess_hysteresis(probs, t_low=0.4, t_high=0.85)


def postprocess_hysteresis_t_high_08(probs: np.ndarray) -> np.ndarray:
    return postprocess_hysteresis(probs, t_low=0.3, t_high=0.8)


def postprocess_hysteresis_t_high_09(probs: np.ndarray) -> np.ndarray:
    return postprocess_hysteresis(probs, t_low=0.3, t_high=0.9)


def postprocess_hysteresis_wide_range(probs: np.ndarray) -> np.ndarray:
    """Wide hysteresis range: t_low=0.2, t_high=0.9."""
    return postprocess_hysteresis(probs, t_low=0.2, t_high=0.9)


# =============================================================================
# 7. ENSEMBLE APPROACHES
# =============================================================================

def postprocess_ensemble_union(probs: np.ndarray) -> np.ndarray:
    """Union of multiple methods."""
    m1 = postprocess_hysteresis(probs, t_low=0.3, t_high=0.85)
    m2 = postprocess_hysteresis_large_closing_z2(probs)
    m3 = postprocess_smoothed_sigma10(probs)

    result = (m1 | m2 | m3).astype(np.uint8)
    return result


def postprocess_ensemble_intersection(probs: np.ndarray) -> np.ndarray:
    """Intersection of multiple methods (conservative)."""
    m1 = postprocess_hysteresis(probs, t_low=0.3, t_high=0.85)
    m2 = postprocess_hysteresis(probs, t_low=0.2, t_high=0.9)
    m3 = postprocess_smoothed_sigma10(probs)

    result = (m1 & m2 & m3).astype(np.uint8)
    return result


def postprocess_ensemble_voting(probs: np.ndarray) -> np.ndarray:
    """Majority voting of multiple methods."""
    from skimage.morphology import remove_small_objects

    m1 = postprocess_hysteresis(probs, t_low=0.3, t_high=0.85)
    m2 = postprocess_hysteresis_large_closing_z2(probs)
    m3 = postprocess_smoothed_sigma10(probs)
    m4 = postprocess_hole_filling(probs)

    # Majority voting (at least 2 out of 4)
    votes = m1.astype(int) + m2.astype(int) + m3.astype(int) + m4.astype(int)
    result = (votes >= 2).astype(np.uint8)

    # Dust removal
    result = remove_small_objects(result.astype(bool), min_size=100).astype(np.uint8)

    return result


# =============================================================================
# POSTPROCESS REGISTRY
# =============================================================================

POSTPROCESS_METHODS = {
    # Baseline
    "none": postprocess_none,
    "hysteresis_baseline": partial(postprocess_hysteresis, t_low=0.3, t_high=0.85),

    # 1. Morphological enhancement
    "morph_closing_z2": postprocess_hysteresis_large_closing_z2,
    "morph_closing_z3": postprocess_hysteresis_large_closing_z3,
    "morph_closing_xy": postprocess_hysteresis_xy_closing,
    "morph_closing_full": postprocess_hysteresis_full_closing,
    "morph_hole_filling": postprocess_hole_filling,
    "morph_open_close": postprocess_opening_closing,

    # 2. Probability preprocessing
    "smooth_sigma05": postprocess_smoothed_sigma05,
    "smooth_sigma10": postprocess_smoothed_sigma10,
    "smooth_sigma15": postprocess_smoothed_sigma15,
    "z_accumulated": postprocess_z_accumulated,
    "anisotropic_smooth": postprocess_anisotropic_smooth,

    # 3. Connected component analysis
    "dust_aggressive": postprocess_aggressive_dust,
    "dust_very_aggressive": postprocess_very_aggressive_dust,
    "component_bridging": postprocess_component_bridging,
    "largest_5": postprocess_largest_5,
    "largest_10": postprocess_largest_10,
    "largest_20": postprocess_largest_20,

    # 4. Frangi
    "frangi_surfaceness": postprocess_frangi_surfaceness,
    "frangi_then_hysteresis": postprocess_frangi_then_hysteresis,

    # 5. Sheet structure
    "thinning": postprocess_thinning,
    "z_continuity": postprocess_z_continuity,

    # 6. Hysteresis variations
    "hyst_t_low_02": postprocess_hysteresis_t_low_02,
    "hyst_t_low_04": postprocess_hysteresis_t_low_04,
    "hyst_t_high_08": postprocess_hysteresis_t_high_08,
    "hyst_t_high_09": postprocess_hysteresis_t_high_09,
    "hyst_wide_range": postprocess_hysteresis_wide_range,

    # 7. Ensemble
    "ensemble_union": postprocess_ensemble_union,
    "ensemble_intersection": postprocess_ensemble_intersection,
    "ensemble_voting": postprocess_ensemble_voting,
}


# =============================================================================
# EVALUATION
# =============================================================================

def compute_single_case(
    case_id: str,
    npz_dir: Path,
    gt_dir: Path,
    postprocess_fn: Callable,
    postprocess_name: str,
    surface_tolerance: float = 2.0,
    ignore_label: int = 2,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict:
    """Compute metrics for a single case."""
    from topometrics import compute_leaderboard_score

    npz_path = npz_dir / f"{case_id}.npz"
    gt_path = gt_dir / f"{case_id}.tif"

    if not npz_path.exists():
        return {"case": case_id, "postprocess": postprocess_name, "error": f"NPZ not found"}
    if not gt_path.exists():
        return {"case": case_id, "postprocess": postprocess_name, "error": f"GT not found"}

    try:
        data = np.load(npz_path)
        probs = data['probabilities']
        pred = postprocess_fn(probs)
        gt = tifffile.imread(gt_path)

        if pred.shape != gt.shape:
            return {"case": case_id, "postprocess": postprocess_name,
                    "error": f"Shape mismatch: pred={pred.shape}, gt={gt.shape}"}

        rep = compute_leaderboard_score(
            predictions=pred,
            labels=gt,
            dims=(0, 1, 2),
            spacing=spacing,
            surface_tolerance=surface_tolerance,
            voi_connectivity=26,
            voi_transform="one_over_one_plus",
            voi_alpha=0.3,
            combine_weights=(0.3, 0.35, 0.35),
            fg_threshold=None,
            ignore_label=ignore_label,
            ignore_mask=None,
        )

        return {
            "case": case_id,
            "postprocess": postprocess_name,
            "leaderboard": rep.score,
            "toposcore": rep.topo.toposcore if rep.topo else np.nan,
            "surface_dice": rep.surface_dice,
            "voi_score": rep.voi.voi_score if rep.voi else np.nan,
        }

    except Exception as e:
        return {"case": case_id, "postprocess": postprocess_name, "error": str(e)}


def _compute_wrapper(args):
    return compute_single_case(*args)


def evaluate_methods(
    npz_dir: Path,
    gt_dir: Path,
    methods: List[str],
    workers: int = 4,
    max_cases: Optional[int] = None,
) -> List[dict]:
    """Evaluate multiple post-processing methods."""
    from tqdm.auto import tqdm

    npz_files = sorted(npz_dir.glob("*.npz"))
    if max_cases:
        npz_files = npz_files[:max_cases]

    case_ids = [f.stem for f in npz_files]
    print(f"Found {len(case_ids)} NPZ files")
    print(f"Methods to evaluate: {len(methods)}")

    args_list = []
    for method_name in methods:
        postprocess_fn = POSTPROCESS_METHODS[method_name]
        for case_id in case_ids:
            args_list.append((
                case_id, npz_dir, gt_dir,
                postprocess_fn, method_name,
                2.0, 2, (1.0, 1.0, 1.0)
            ))

    print(f"Total evaluations: {len(args_list)}")

    if workers > 1:
        with Pool(workers) as pool:
            results = list(tqdm(
                pool.imap(_compute_wrapper, args_list),
                total=len(args_list),
                desc="Evaluating"
            ))
    else:
        results = [compute_single_case(*args) for args in tqdm(args_list)]

    return results


def print_summary(results: List[dict]):
    """Print summary by method."""
    from collections import defaultdict

    by_method = defaultdict(list)
    for r in results:
        if "error" not in r:
            by_method[r["postprocess"]].append(r)

    print(f"\n{'='*90}")
    print("POST-PROCESSING COMPARISON SUMMARY")
    print(f"{'='*90}")

    print(f"\n{'Method':<30} {'Leaderboard':>12} {'TopoScore':>12} {'SurfDice':>12} {'VOI':>12}")
    print("-" * 90)

    method_scores = []
    for method in sorted(by_method.keys()):
        valid = by_method[method]
        if not valid:
            continue

        lb = np.mean([r["leaderboard"] for r in valid])
        topo = np.mean([r["toposcore"] for r in valid if not np.isnan(r["toposcore"])])
        sd = np.mean([r["surface_dice"] for r in valid])
        voi = np.mean([r["voi_score"] for r in valid if not np.isnan(r["voi_score"])])

        method_scores.append((method, lb, topo, sd, voi))
        print(f"{method:<30} {lb:>12.4f} {topo:>12.4f} {sd:>12.4f} {voi:>12.4f}")

    # Sort by leaderboard score
    method_scores.sort(key=lambda x: x[1], reverse=True)

    print("-" * 90)
    print("\nTOP 5 METHODS:")
    for i, (method, lb, topo, sd, voi) in enumerate(method_scores[:5], 1):
        print(f"  {i}. {method}: {lb:.4f}")

    print(f"{'='*90}")


def save_results(results: List[dict], output_path: Path):
    """Save results to CSV."""
    valid_results = [r for r in results if "error" not in r]
    if not valid_results:
        return

    fieldnames = ["case", "postprocess", "leaderboard", "toposcore", "surface_dice", "voi_score"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in valid_results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate new post-processing methods")

    parser.add_argument("--npz-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--methods", type=str, default="all",
                        help="Comma-separated method names or 'all'")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)

    args = parser.parse_args()

    if args.methods == "all":
        methods = list(POSTPROCESS_METHODS.keys())
    else:
        methods = [m.strip() for m in args.methods.split(",")]

    print(f"NPZ directory: {args.npz_dir}")
    print(f"GT directory: {args.gt_dir}")
    print(f"Methods: {len(methods)}")

    results = evaluate_methods(
        npz_dir=args.npz_dir,
        gt_dir=args.gt_dir,
        methods=methods,
        workers=args.workers,
        max_cases=args.max_cases,
    )

    print_summary(results)

    if args.output_csv:
        save_results(results, args.output_csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Vesuvius Competition Metric Evaluation Script with Post-processing

Computes the official leaderboard score using topometrics:
    Leaderboard = 0.3 * TopoScore + 0.35 * SurfaceDice@τ + 0.35 * VOI_score

Supports multiple post-processing methods:
    - none: argmax (baseline)
    - threshold: simple threshold on surface probability
    - host_baseline: threshold 0.75 + Frangi surfaceness filter
    - hysteresis: 3D hysteresis thresholding + closing + dust removal

Usage:
    # Evaluate with different post-processing methods
    python evaluate_metrics.py \
        --npz-dir /path/to/npz_predictions \
        --gt-dir /path/to/labels \
        --postprocess none \
        --workers 48

    # Compare all post-processing methods
    python evaluate_metrics.py \
        --npz-dir /path/to/npz_predictions \
        --gt-dir /path/to/labels \
        --postprocess all \
        --workers 48
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


# =============================================================================
# POST-PROCESSING METHODS
# =============================================================================

def postprocess_none(probs: np.ndarray) -> np.ndarray:
    """Argmax - no post-processing (baseline)."""
    return np.argmax(probs, axis=0).astype(np.uint8)


def postprocess_threshold(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Simple threshold on surface class probability."""
    surface_prob = probs[1]  # Class 1 = surface
    return (surface_prob >= threshold).astype(np.uint8)


def postprocess_threshold_05(probs: np.ndarray) -> np.ndarray:
    """Threshold at 0.5."""
    return postprocess_threshold(probs, threshold=0.5)


def postprocess_threshold_075(probs: np.ndarray) -> np.ndarray:
    """Threshold at 0.75 (Host Baseline step 1 only)."""
    return postprocess_threshold(probs, threshold=0.75)


def surfaceness_filter(volume: np.ndarray, sigmas: range = range(1, 4)) -> np.ndarray:
    """
    Frangi-like filter for sheet/surface structures.

    For sheet-like structures, we want:
    - Two small eigenvalues (flat in two directions)
    - One large eigenvalue (curved in one direction)
    """
    from scipy.ndimage import gaussian_filter

    result = np.zeros_like(volume, dtype=np.float64)

    for sigma in sigmas:
        # Compute Hessian matrix components
        smoothed = gaussian_filter(volume.astype(np.float64), sigma=sigma)

        # Second derivatives
        Hzz = gaussian_filter(smoothed, sigma=sigma, order=[2, 0, 0])
        Hyy = gaussian_filter(smoothed, sigma=sigma, order=[0, 2, 0])
        Hxx = gaussian_filter(smoothed, sigma=sigma, order=[0, 0, 2])
        Hzy = gaussian_filter(smoothed, sigma=sigma, order=[1, 1, 0])
        Hzx = gaussian_filter(smoothed, sigma=sigma, order=[1, 0, 1])
        Hyx = gaussian_filter(smoothed, sigma=sigma, order=[0, 1, 1])

        # Compute eigenvalues (vectorized for speed)
        # For 3x3 symmetric matrix, use analytical formula
        a = Hzz
        b = Hyy
        c = Hxx
        d = Hzy
        e = Hzx
        f = Hyx

        # Trace and other invariants
        I1 = a + b + c
        I2 = a*b + b*c + c*a - d*d - e*e - f*f
        I3 = a*b*c + 2*d*e*f - a*f*f - b*e*e - c*d*d

        # Eigenvalues via Cardano's formula (simplified)
        p = I1*I1 - 3*I2
        q = 2*I1*I1*I1 - 9*I1*I2 + 27*I3

        with np.errstate(invalid='ignore', divide='ignore'):
            discriminant = np.clip(q*q - 4*p*p*p, 0, None)
            sqrt_disc = np.sqrt(discriminant)

            # Approximate eigenvalues
            lambda1 = I1 / 3

        # Surfaceness response (simplified)
        # High response when one eigenvalue dominates
        response = np.abs(Hzz) + np.abs(Hyy) + np.abs(Hxx)
        response = response * (volume > 0)

        result = np.maximum(result, response)

    # Normalize
    if result.max() > 0:
        result = result / result.max()

    return result


def postprocess_host_baseline(
    probs: np.ndarray,
    threshold: float = 0.75,
    frangi_threshold: float = 0.5,
    min_component_size: int = 100
) -> np.ndarray:
    """
    Host Baseline post-processing.

    1. Threshold at 0.75
    2. Apply Frangi surfaceness filter
    3. Remove small connected components
    """
    surface_prob = probs[1]  # Class 1 = surface

    # Step 1: Threshold
    thresholded = surface_prob.copy()
    thresholded[thresholded < threshold] = 0

    # Step 2: Frangi surfaceness filter
    enhanced = surfaceness_filter(thresholded)

    # Threshold enhanced result
    result = (enhanced > frangi_threshold).astype(np.uint8)

    # Step 3: Remove small connected components
    if min_component_size > 0:
        labeled, num_features = ndi.label(result)
        if num_features > 0:
            sizes = ndi.sum(result, labeled, range(1, num_features + 1))
            mask = np.isin(labeled, np.where(np.array(sizes) >= min_component_size)[0] + 1)
            result = result * mask

    return result.astype(np.uint8)


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
    TransUNet-style post-processing with hysteresis thresholding.

    1. 3D Hysteresis thresholding
    2. Anisotropic closing
    3. Dust removal
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


def postprocess_opening_closing(probs: np.ndarray) -> np.ndarray:
    """
    Best post-processing: Hysteresis + opening + closing.

    1. Hysteresis thresholding (t_low=0.3, t_high=0.85)
    2. Opening (remove small protrusions/noise)
    3. Anisotropic closing (fill gaps, z_radius=2, xy_radius=1)
    4. Dust removal
    """
    from skimage.morphology import remove_small_objects

    surface_prob = probs[1]

    # Step 1: Hysteresis thresholding
    t_low, t_high = 0.3, 0.85
    strong = surface_prob >= t_high
    weak = surface_prob >= t_low

    if not strong.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    if not mask.any():
        return np.zeros(surface_prob.shape, dtype=np.uint8)

    # Step 2: Opening (remove small protrusions)
    struct_open = ndi.generate_binary_structure(3, 1)
    mask = ndi.binary_opening(mask, structure=struct_open)

    # Step 3: Anisotropic closing (z_radius=2, xy_radius=1)
    struct_close = build_anisotropic_struct(z_radius=2, xy_radius=1)
    mask = ndi.binary_closing(mask, structure=struct_close)

    # Step 4: Dust removal
    mask = remove_small_objects(mask.astype(bool), min_size=100)

    return mask.astype(np.uint8)


# Post-processing registry
POSTPROCESS_METHODS = {
    "none": postprocess_none,
    "threshold_05": postprocess_threshold_05,
    "threshold_075": postprocess_threshold_075,
    "host_baseline": postprocess_host_baseline,
    "hysteresis": postprocess_hysteresis,
    "opening_closing": postprocess_opening_closing,
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
    """Compute metrics for a single case with post-processing."""
    from topometrics import compute_leaderboard_score

    npz_path = npz_dir / f"{case_id}.npz"
    gt_path = gt_dir / f"{case_id}.tif"

    if not npz_path.exists():
        return {"case": case_id, "postprocess": postprocess_name, "error": f"NPZ not found: {npz_path}"}
    if not gt_path.exists():
        return {"case": case_id, "postprocess": postprocess_name, "error": f"GT not found: {gt_path}"}

    try:
        # Load probability map
        data = np.load(npz_path)
        probs = data['probabilities']

        # Apply post-processing
        pred = postprocess_fn(probs)

        # Load ground truth
        gt = tifffile.imread(gt_path)

        # Ensure same shape
        if pred.shape != gt.shape:
            return {"case": case_id, "postprocess": postprocess_name,
                    "error": f"Shape mismatch: pred={pred.shape}, gt={gt.shape}"}

        # Compute leaderboard score
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
            "topoF1_0": rep.topo.topoF1_by_dim.get(0, np.nan) if rep.topo else np.nan,
            "topoF1_1": rep.topo.topoF1_by_dim.get(1, np.nan) if rep.topo else np.nan,
            "topoF1_2": rep.topo.topoF1_by_dim.get(2, np.nan) if rep.topo else np.nan,
            "surface_dice": rep.surface_dice,
            "voi_score": rep.voi.voi_score if rep.voi else np.nan,
            "voi_total": rep.voi.voi_total if rep.voi else np.nan,
            "voi_split": rep.voi.voi_split if rep.voi else np.nan,
            "voi_merge": rep.voi.voi_merge if rep.voi else np.nan,
        }

    except Exception as e:
        import traceback
        return {"case": case_id, "postprocess": postprocess_name, "error": str(e)}


def _compute_wrapper(args):
    """Wrapper for multiprocessing."""
    return compute_single_case(*args)


def evaluate_with_postprocess(
    npz_dir: Path,
    gt_dir: Path,
    postprocess_methods: List[str],
    surface_tolerance: float = 2.0,
    ignore_label: int = 2,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    workers: int = 4,
    max_cases: Optional[int] = None,
    t_low: float = 0.5,
    t_high: float = 0.9,
) -> List[dict]:
    """Evaluate predictions with multiple post-processing methods."""
    from tqdm.auto import tqdm

    # Find NPZ files
    npz_files = sorted(npz_dir.glob("*.npz"))
    if max_cases:
        npz_files = npz_files[:max_cases]

    case_ids = [f.stem for f in npz_files]
    print(f"Found {len(case_ids)} NPZ files")
    print(f"Post-processing methods: {postprocess_methods}")

    # Prepare arguments for all combinations
    args_list = []
    for method_name in postprocess_methods:
        postprocess_fn = POSTPROCESS_METHODS[method_name]
        # Apply custom t_low/t_high for hysteresis
        if method_name == "hysteresis":
            postprocess_fn = partial(postprocess_fn, t_low=t_low, t_high=t_high)
        for case_id in case_ids:
            args_list.append((
                case_id, npz_dir, gt_dir,
                postprocess_fn, method_name,
                surface_tolerance, ignore_label, spacing
            ))

    print(f"Total evaluations: {len(args_list)}")

    # Run evaluation
    if workers > 1:
        with Pool(workers) as pool:
            results = list(tqdm(
                pool.imap(_compute_wrapper, args_list),
                total=len(args_list),
                desc="Evaluating"
            ))
    else:
        results = []
        for args in tqdm(args_list, desc="Evaluating"):
            results.append(compute_single_case(*args))

    return results


def print_comparison_summary(results: List[dict]):
    """Print comparison summary by post-processing method."""
    from collections import defaultdict

    # Group by method
    by_method = defaultdict(list)
    for r in results:
        if "error" not in r:
            by_method[r["postprocess"]].append(r)

    print(f"\n{'='*80}")
    print("POST-PROCESSING COMPARISON")
    print(f"{'='*80}")

    # Summary table header
    print(f"\n{'Method':<20} {'Leaderboard':>12} {'TopoScore':>12} {'SurfDice':>12} {'VOI':>12}")
    print("-" * 70)

    method_scores = {}
    for method in sorted(by_method.keys()):
        valid = by_method[method]
        if not valid:
            continue

        lb = np.mean([r["leaderboard"] for r in valid])
        topo = np.mean([r["toposcore"] for r in valid if not np.isnan(r["toposcore"])])
        sd = np.mean([r["surface_dice"] for r in valid])
        voi = np.mean([r["voi_score"] for r in valid if not np.isnan(r["voi_score"])])

        method_scores[method] = lb
        print(f"{method:<20} {lb:>12.4f} {topo:>12.4f} {sd:>12.4f} {voi:>12.4f}")

    # Find best method
    if method_scores:
        best_method = max(method_scores, key=method_scores.get)
        print("-" * 70)
        print(f"Best method: {best_method} (Leaderboard: {method_scores[best_method]:.4f})")

    print(f"{'='*80}")

    # Detailed per-method statistics
    for method in sorted(by_method.keys()):
        valid = by_method[method]
        if not valid:
            continue

        print(f"\n--- {method} ({len(valid)} cases) ---")

        metrics = ["leaderboard", "toposcore", "surface_dice", "voi_score"]
        for metric in metrics:
            values = [r[metric] for r in valid if not np.isnan(r.get(metric, np.nan))]
            if values:
                print(f"  {metric}: mean={np.mean(values):.4f}, std={np.std(values):.4f}, "
                      f"min={np.min(values):.4f}, max={np.max(values):.4f}")


def save_results(results: List[dict], output_path: Path):
    """Save results to CSV."""
    if not results:
        return

    valid_results = [r for r in results if "error" not in r]
    if not valid_results:
        return

    fieldnames = list(valid_results[0].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in valid_results:
            writer.writerow(r)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Vesuvius Metric Evaluation with Post-processing Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--npz-dir", type=Path, required=True,
        help="Directory containing NPZ probability files from nnUNet"
    )
    parser.add_argument(
        "--gt-dir", type=Path, required=True,
        help="Directory containing ground truth TIF files"
    )
    parser.add_argument(
        "--postprocess", type=str, default="all",
        choices=["all"] + list(POSTPROCESS_METHODS.keys()),
        help="Post-processing method (default: all)"
    )
    parser.add_argument(
        "--surface-tolerance", type=float, default=2.0,
        help="Surface dice tolerance in voxels (default: 2.0)"
    )
    parser.add_argument(
        "--ignore-label", type=int, default=2,
        help="Label value to ignore in GT (default: 2)"
    )
    parser.add_argument(
        "--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0],
        help="Voxel spacing (z, y, x) (default: 1 1 1)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--max-cases", type=int, default=None,
        help="Limit number of cases to evaluate"
    )
    parser.add_argument(
        "--output-csv", type=Path, default=None,
        help="Output CSV file path"
    )
    parser.add_argument(
        "--t-low", type=float, default=0.5,
        help="Hysteresis low threshold (default: 0.5)"
    )
    parser.add_argument(
        "--t-high", type=float, default=0.9,
        help="Hysteresis high threshold (default: 0.9)"
    )

    args = parser.parse_args()

    # Validate paths
    if not args.npz_dir.exists():
        print(f"ERROR: NPZ directory not found: {args.npz_dir}")
        return 1
    if not args.gt_dir.exists():
        print(f"ERROR: Ground truth directory not found: {args.gt_dir}")
        return 1

    # Determine methods to evaluate
    if args.postprocess == "all":
        methods = list(POSTPROCESS_METHODS.keys())
    else:
        methods = [args.postprocess]

    print(f"NPZ directory: {args.npz_dir}")
    print(f"Ground truth directory: {args.gt_dir}")
    print(f"Post-processing methods: {methods}")
    print(f"Surface tolerance: {args.surface_tolerance}")
    print(f"Workers: {args.workers}")

    # Run evaluation
    results = evaluate_with_postprocess(
        npz_dir=args.npz_dir,
        gt_dir=args.gt_dir,
        postprocess_methods=methods,
        surface_tolerance=args.surface_tolerance,
        ignore_label=args.ignore_label,
        spacing=tuple(args.spacing),
        workers=args.workers,
        max_cases=args.max_cases,
        t_low=args.t_low,
        t_high=args.t_high,
    )

    # Print comparison summary
    print_comparison_summary(results)

    # Save results
    if args.output_csv:
        save_results(results, args.output_csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())

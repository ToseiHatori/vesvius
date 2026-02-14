#!/usr/bin/env python3
"""
Hysteresis Post-processing Hyperparameter Grid Search

Searches over t_low and t_high parameters to find optimal settings.

Usage:
    python evaluate_hysteresis_grid.py \
        --npz-dir /path/to/npz_predictions \
        --gt-dir /path/to/labels \
        --workers 20
"""

import argparse
import csv
import sys
from pathlib import Path
from multiprocessing import Pool
from typing import Optional, List, Tuple
from functools import partial
import numpy as np
import tifffile
from scipy import ndimage as ndi
from itertools import product


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


def postprocess_hysteresis_params(
    probs: np.ndarray,
    t_low: float,
    t_high: float,
    z_radius: int = 1,
    xy_radius: int = 0,
    dust_min_size: int = 100
) -> np.ndarray:
    """Hysteresis thresholding with configurable parameters."""
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


def compute_single_case_with_params(
    case_id: str,
    npz_dir: Path,
    gt_dir: Path,
    t_low: float,
    t_high: float,
    z_radius: int,
    xy_radius: int,
    dust_min_size: int,
    surface_tolerance: float = 2.0,
    ignore_label: int = 2,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict:
    """Compute metrics for a single case with specific hysteresis parameters."""
    from topometrics import compute_leaderboard_score

    param_str = f"t_low={t_low}_t_high={t_high}_z={z_radius}_xy={xy_radius}_dust={dust_min_size}"

    npz_path = npz_dir / f"{case_id}.npz"
    gt_path = gt_dir / f"{case_id}.tif"

    if not npz_path.exists():
        return {"case": case_id, "params": param_str, "error": f"NPZ not found"}
    if not gt_path.exists():
        return {"case": case_id, "params": param_str, "error": f"GT not found"}

    try:
        # Load probability map
        data = np.load(npz_path)
        probs = data['probabilities']

        # Apply post-processing with specific parameters
        pred = postprocess_hysteresis_params(
            probs, t_low=t_low, t_high=t_high,
            z_radius=z_radius, xy_radius=xy_radius, dust_min_size=dust_min_size
        )

        # Load ground truth
        gt = tifffile.imread(gt_path)

        # Ensure same shape
        if pred.shape != gt.shape:
            return {"case": case_id, "params": param_str,
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
            "t_low": t_low,
            "t_high": t_high,
            "z_radius": z_radius,
            "xy_radius": xy_radius,
            "dust_min_size": dust_min_size,
            "leaderboard": rep.score,
            "toposcore": rep.topo.toposcore if rep.topo else np.nan,
            "surface_dice": rep.surface_dice,
            "voi_score": rep.voi.voi_score if rep.voi else np.nan,
        }

    except Exception as e:
        import traceback
        return {"case": case_id, "params": param_str, "error": str(e)}


def _compute_wrapper(args):
    """Wrapper for multiprocessing."""
    return compute_single_case_with_params(*args)


def run_grid_search(
    npz_dir: Path,
    gt_dir: Path,
    t_low_values: List[float],
    t_high_values: List[float],
    z_radius_values: List[int],
    xy_radius_values: List[int],
    dust_min_size_values: List[int],
    surface_tolerance: float = 2.0,
    ignore_label: int = 2,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    workers: int = 20,
    max_cases: Optional[int] = None,
) -> List[dict]:
    """Run grid search over hysteresis parameters."""
    from tqdm.auto import tqdm

    # Find NPZ files
    npz_files = sorted(npz_dir.glob("*.npz"))
    if max_cases:
        npz_files = npz_files[:max_cases]

    case_ids = [f.stem for f in npz_files]
    print(f"Found {len(case_ids)} NPZ files")

    # Generate parameter combinations
    param_combos = list(product(
        t_low_values, t_high_values,
        z_radius_values, xy_radius_values, dust_min_size_values
    ))

    # Filter invalid combinations (t_low must be < t_high)
    param_combos = [(tl, th, z, xy, dust) for tl, th, z, xy, dust in param_combos if tl < th]

    print(f"Parameter combinations: {len(param_combos)}")
    print(f"  t_low: {t_low_values}")
    print(f"  t_high: {t_high_values}")
    print(f"  z_radius: {z_radius_values}")
    print(f"  xy_radius: {xy_radius_values}")
    print(f"  dust_min_size: {dust_min_size_values}")

    # Prepare arguments for all combinations
    args_list = []
    for t_low, t_high, z_radius, xy_radius, dust_min_size in param_combos:
        for case_id in case_ids:
            args_list.append((
                case_id, npz_dir, gt_dir,
                t_low, t_high, z_radius, xy_radius, dust_min_size,
                surface_tolerance, ignore_label, spacing
            ))

    print(f"Total evaluations: {len(args_list)} ({len(param_combos)} params × {len(case_ids)} cases)")
    print(f"Workers: {workers}")

    # Run evaluation
    if workers > 1:
        with Pool(workers) as pool:
            results = list(tqdm(
                pool.imap(_compute_wrapper, args_list),
                total=len(args_list),
                desc="Grid Search"
            ))
    else:
        results = []
        for args in tqdm(args_list, desc="Grid Search"):
            results.append(compute_single_case_with_params(*args))

    return results


def summarize_results(results: List[dict]):
    """Summarize grid search results."""
    from collections import defaultdict

    # Group by parameters
    by_params = defaultdict(list)
    for r in results:
        if "error" not in r:
            key = (r["t_low"], r["t_high"], r["z_radius"], r["xy_radius"], r["dust_min_size"])
            by_params[key].append(r)

    print(f"\n{'='*100}")
    print("GRID SEARCH RESULTS")
    print(f"{'='*100}")

    # Summary table
    print(f"\n{'t_low':>6} {'t_high':>6} {'z_rad':>5} {'xy_rad':>6} {'dust':>5} | "
          f"{'Leaderboard':>11} {'TopoScore':>10} {'SurfDice':>10} {'VOI':>10} | {'N':>3}")
    print("-" * 100)

    param_scores = []
    for key in sorted(by_params.keys()):
        valid = by_params[key]
        if not valid:
            continue

        t_low, t_high, z_radius, xy_radius, dust_min_size = key
        lb = np.mean([r["leaderboard"] for r in valid])
        topo = np.nanmean([r["toposcore"] for r in valid])
        sd = np.mean([r["surface_dice"] for r in valid])
        voi = np.nanmean([r["voi_score"] for r in valid])

        param_scores.append((key, lb, topo, sd, voi, len(valid)))
        print(f"{t_low:>6.2f} {t_high:>6.2f} {z_radius:>5} {xy_radius:>6} {dust_min_size:>5} | "
              f"{lb:>11.4f} {topo:>10.4f} {sd:>10.4f} {voi:>10.4f} | {len(valid):>3}")

    # Find best parameters
    if param_scores:
        best = max(param_scores, key=lambda x: x[1])
        print("-" * 100)
        print(f"\nBEST PARAMETERS:")
        print(f"  t_low={best[0][0]}, t_high={best[0][1]}, z_radius={best[0][2]}, "
              f"xy_radius={best[0][3]}, dust_min_size={best[0][4]}")
        print(f"  Leaderboard: {best[1]:.4f}")
        print(f"  TopoScore:   {best[2]:.4f}")
        print(f"  SurfaceDice: {best[3]:.4f}")
        print(f"  VOI Score:   {best[4]:.4f}")

    print(f"{'='*100}")

    return param_scores


def save_results(results: List[dict], output_path: Path):
    """Save results to CSV."""
    valid_results = [r for r in results if "error" not in r]
    if not valid_results:
        print("No valid results to save")
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
        description="Hysteresis Post-processing Hyperparameter Grid Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--npz-dir", type=Path, required=True,
        help="Directory containing NPZ probability files"
    )
    parser.add_argument(
        "--gt-dir", type=Path, required=True,
        help="Directory containing ground truth TIF files"
    )
    parser.add_argument(
        "--t-low", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6],
        help="t_low values to search (default: 0.3 0.4 0.5 0.6)"
    )
    parser.add_argument(
        "--t-high", type=float, nargs="+", default=[0.75, 0.85, 0.9, 0.95],
        help="t_high values to search (default: 0.75 0.85 0.9 0.95)"
    )
    parser.add_argument(
        "--z-radius", type=int, nargs="+", default=[1],
        help="z_radius values to search (default: 1)"
    )
    parser.add_argument(
        "--xy-radius", type=int, nargs="+", default=[0],
        help="xy_radius values to search (default: 0)"
    )
    parser.add_argument(
        "--dust-min-size", type=int, nargs="+", default=[100],
        help="dust_min_size values to search (default: 100)"
    )
    parser.add_argument(
        "--surface-tolerance", type=float, default=2.0,
        help="Surface dice tolerance (default: 2.0)"
    )
    parser.add_argument(
        "--ignore-label", type=int, default=2,
        help="Label to ignore in GT (default: 2)"
    )
    parser.add_argument(
        "--workers", type=int, default=20,
        help="Number of parallel workers (default: 20)"
    )
    parser.add_argument(
        "--max-cases", type=int, default=None,
        help="Limit number of cases"
    )
    parser.add_argument(
        "--output-csv", type=Path, default=None,
        help="Output CSV file path"
    )

    args = parser.parse_args()

    # Validate paths
    if not args.npz_dir.exists():
        print(f"ERROR: NPZ directory not found: {args.npz_dir}")
        return 1
    if not args.gt_dir.exists():
        print(f"ERROR: Ground truth directory not found: {args.gt_dir}")
        return 1

    print(f"NPZ directory: {args.npz_dir}")
    print(f"Ground truth directory: {args.gt_dir}")

    # Run grid search
    results = run_grid_search(
        npz_dir=args.npz_dir,
        gt_dir=args.gt_dir,
        t_low_values=args.t_low,
        t_high_values=args.t_high,
        z_radius_values=args.z_radius,
        xy_radius_values=args.xy_radius,
        dust_min_size_values=args.dust_min_size,
        surface_tolerance=args.surface_tolerance,
        ignore_label=args.ignore_label,
        workers=args.workers,
        max_cases=args.max_cases,
    )

    # Summarize
    summarize_results(results)

    # Save results
    if args.output_csv:
        save_results(results, args.output_csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())

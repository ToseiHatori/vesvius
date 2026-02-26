#!/usr/bin/env python3
"""
Ensemble Weight Optimization via Log Loss

Optimizes the blending weights between lowres and fullres models
using binary cross entropy (log loss) which is much faster than
the full competition metrics.

Usage:
    # Grid search with default resolution
    python optimize_ensemble_weights.py

    # Fine-grained search
    python optimize_ensemble_weights.py --grid-resolution 0.02

    # Test with limited cases
    python optimize_ensemble_weights.py --max-cases 10
"""

import argparse
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile


# =============================================================================
# PATHS
# =============================================================================

_LOCAL_BASE = Path("/home/ben/Dev/vesvius")
_DOCKER_BASE = Path("/workspace")

if _DOCKER_BASE.exists() and (_DOCKER_BASE / "nnunet_output").exists():
    BASE_DIR = _DOCKER_BASE
else:
    BASE_DIR = _LOCAL_BASE

RESULTS_DIR = BASE_DIR / "nnunet_output" / "nnUNet_results" / "Dataset100_VesuviusSurface"
GT_DIR = BASE_DIR / "nnunet_output" / "nnUNet_preprocessed" / "Dataset100_VesuviusSurface" / "gt_segmentations"

LOWRES_DIR = RESULTS_DIR / "nnUNetTrainer_2000epochs__nnUNetResEncUNetMPlans__3d_lowres"
FULLRES_DIR = RESULTS_DIR / "nnUNetTrainer_2000epochs__nnUNetResEncUNetMPlans__3d_fullres"


# =============================================================================
# LOG LOSS COMPUTATION
# =============================================================================

def compute_logloss(probs: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    """
    Compute binary cross entropy (log loss) for surface class.

    Args:
        probs: Softmax probabilities, shape (2, D, H, W) or (D, H, W)
        gt: Ground truth labels (0=bg, 1=surface, 2=ignore)
        eps: Small value to avoid log(0)

    Returns:
        Mean log loss over valid voxels
    """
    # Get surface probability (class 1)
    if probs.ndim == 4 and probs.shape[0] == 2:
        p_surface = probs[1]
    else:
        p_surface = probs

    # Create valid mask (exclude ignore label)
    valid_mask = gt != 2

    # Get valid predictions and labels
    p = p_surface[valid_mask]
    y = gt[valid_mask].astype(np.float32)

    # Clip to avoid log(0)
    p = np.clip(p, eps, 1 - eps)

    # Binary cross entropy
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    return float(loss)


def compute_weighted_logloss(
    lowres_probs: np.ndarray,
    fullres_probs: np.ndarray,
    gt: np.ndarray,
    alpha: float,
) -> float:
    """
    Compute log loss for weighted ensemble.

    Args:
        lowres_probs: Lowres softmax probabilities
        fullres_probs: Fullres softmax probabilities
        gt: Ground truth
        alpha: Weight for lowres (ensemble = alpha*lowres + (1-alpha)*fullres)

    Returns:
        Log loss
    """
    ensemble = alpha * lowres_probs + (1 - alpha) * fullres_probs
    return compute_logloss(ensemble, gt)


# =============================================================================
# DATA LOADING
# =============================================================================

def get_validation_cases() -> List[Tuple[str, int]]:
    """
    Get all validation cases from both folds.

    Returns:
        List of (case_id, fold) tuples
    """
    cases = []

    for fold in [0, 1]:
        npz_dir = LOWRES_DIR / f"fold_{fold}" / "validation_npz"
        if not npz_dir.exists():
            continue

        for npz_file in sorted(npz_dir.glob("*.npz")):
            case_id = npz_file.stem
            # Check if both lowres and fullres exist
            fullres_npz = FULLRES_DIR / f"fold_{fold}" / "validation_npz" / f"{case_id}.npz"
            gt_path = GT_DIR / f"{case_id}.tif"

            if fullres_npz.exists() and gt_path.exists():
                cases.append((case_id, fold))

    return cases


def load_case_data(case_id: str, fold: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load lowres probs, fullres probs, and ground truth for a case.

    Returns:
        (lowres_probs, fullres_probs, gt) or None if loading fails
    """
    lowres_npz = LOWRES_DIR / f"fold_{fold}" / "validation_npz" / f"{case_id}.npz"
    fullres_npz = FULLRES_DIR / f"fold_{fold}" / "validation_npz" / f"{case_id}.npz"
    gt_path = GT_DIR / f"{case_id}.tif"

    try:
        with np.load(lowres_npz) as data:
            lowres_probs = data['probabilities'].copy()
        with np.load(fullres_npz) as data:
            fullres_probs = data['probabilities'].copy()
        gt = tifffile.imread(gt_path)

        return lowres_probs, fullres_probs, gt
    except Exception as e:
        print(f"  Error loading {case_id}: {e}")
        return None


# =============================================================================
# GRID SEARCH (PARALLEL)
# =============================================================================

def process_single_case(
    args: Tuple[str, int, List[float]]
) -> Optional[Dict[float, float]]:
    """
    Process a single case and compute logloss for all alpha values.

    Args:
        args: (case_id, fold, alpha_values)

    Returns:
        Dict mapping alpha -> loss, or None if loading fails
    """
    case_id, fold, alpha_values = args

    data = load_case_data(case_id, fold)
    if data is None:
        return None

    lowres_probs, fullres_probs, gt = data

    results = {}
    for alpha in alpha_values:
        loss = compute_weighted_logloss(lowres_probs, fullres_probs, gt, alpha)
        results[alpha] = loss

    # Clean up
    del lowres_probs, fullres_probs, gt
    gc.collect()

    return results


def grid_search(
    cases: List[Tuple[str, int]],
    alpha_values: List[float],
    n_workers: int = 8,
    verbose: bool = True,
) -> dict:
    """
    Parallel grid search over alpha values.

    Args:
        cases: List of (case_id, fold) tuples
        alpha_values: List of alpha values to try
        n_workers: Number of parallel workers
        verbose: Whether to print progress

    Returns:
        Dictionary with results per alpha
    """
    results = {alpha: [] for alpha in alpha_values}

    # Prepare arguments for workers
    work_items = [(case_id, fold, alpha_values) for case_id, fold in cases]

    completed = 0
    total = len(cases)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_case, item): item for item in work_items}

        for future in as_completed(futures):
            completed += 1
            case_id, fold, _ = futures[future]

            try:
                case_results = future.result()
                if case_results is not None:
                    for alpha, loss in case_results.items():
                        results[alpha].append(loss)
                    status = "OK"
                else:
                    status = "SKIP"
            except Exception as e:
                status = f"ERROR: {e}"

            if verbose:
                print(f"[{completed}/{total}] {case_id} (fold {fold}): {status}", flush=True)

    return results


def analyze_results(results: dict) -> Tuple[float, float, dict]:
    """
    Analyze grid search results.

    Returns:
        (best_alpha, best_loss, summary_dict)
    """
    summary = {}

    for alpha, losses in sorted(results.items()):
        if losses:
            mean_loss = np.mean(losses)
            std_loss = np.std(losses)
            summary[alpha] = {
                "mean": mean_loss,
                "std": std_loss,
                "n_cases": len(losses),
            }

    # Find best alpha
    best_alpha = min(summary.keys(), key=lambda a: summary[a]["mean"])
    best_loss = summary[best_alpha]["mean"]

    return best_alpha, best_loss, summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Optimize ensemble weights via log loss")
    parser.add_argument(
        "--grid-resolution", type=float, default=0.1,
        help="Step size for alpha grid (default: 0.1)"
    )
    parser.add_argument(
        "--alpha-min", type=float, default=0.3,
        help="Minimum alpha value (default: 0.3)"
    )
    parser.add_argument(
        "--alpha-max", type=float, default=0.7,
        help="Maximum alpha value (default: 0.7)"
    )
    parser.add_argument(
        "--max-cases", type=int, default=None,
        help="Maximum number of cases to use (for testing)"
    )
    parser.add_argument(
        "--fine-tune", action="store_true",
        help="Run fine-tuning around the best alpha found"
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel workers (default: 8)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Ensemble Weight Optimization via Log Loss")
    print("=" * 60)

    # Get validation cases
    cases = get_validation_cases()
    print(f"\nFound {len(cases)} validation cases (both folds)")

    if args.max_cases and len(cases) > args.max_cases:
        cases = cases[:args.max_cases]
        print(f"Using first {len(cases)} cases")

    # Define alpha grid
    alpha_values = list(np.arange(args.alpha_min, args.alpha_max + args.grid_resolution/2, args.grid_resolution))
    alpha_values = [round(a, 3) for a in alpha_values]
    print(f"\nSearching over {len(alpha_values)} alpha values: {alpha_values}")
    print("  alpha < 0.5 means favor fullres")
    print("  alpha = 0.5 means 50/50 (current)")
    print("  alpha > 0.5 means favor lowres")

    # Run grid search
    print("\n" + "-" * 60)
    print(f"Running grid search with {args.workers} workers...")
    print("-" * 60)

    results = grid_search(cases, alpha_values, n_workers=args.workers, verbose=True)

    # Analyze results
    print("\n" + "-" * 60)
    print("Results (sorted by log loss, lower is better)")
    print("-" * 60)

    best_alpha, best_loss, summary = analyze_results(results)

    # Sort by mean loss
    sorted_alphas = sorted(summary.keys(), key=lambda a: summary[a]["mean"])

    print(f"\n{'Alpha':<10} {'Mean Loss':<15} {'Std':<10} {'N':<5}")
    print("-" * 45)
    for alpha in sorted_alphas:
        s = summary[alpha]
        marker = " <-- BEST" if alpha == best_alpha else ""
        print(f"{alpha:<10.3f} {s['mean']:<15.6f} {s['std']:<10.6f} {s['n_cases']:<5}{marker}")

    # Fine-tuning
    if args.fine_tune and best_alpha > 0 and best_alpha < 1:
        print("\n" + "-" * 60)
        print("Fine-tuning around best alpha...")
        print("-" * 60)

        fine_step = args.grid_resolution / 5
        fine_alphas = list(np.arange(
            max(0, best_alpha - args.grid_resolution),
            min(1, best_alpha + args.grid_resolution) + fine_step/2,
            fine_step
        ))
        fine_alphas = [round(a, 4) for a in fine_alphas]

        fine_results = grid_search(cases, fine_alphas, n_workers=args.workers, verbose=False)
        fine_best, fine_loss, fine_summary = analyze_results(fine_results)

        print(f"\nFine-tuned best alpha: {fine_best:.4f} (loss: {fine_loss:.6f})")

        if fine_loss < best_loss:
            best_alpha = fine_best
            best_loss = fine_loss

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Best alpha: {best_alpha}")
    print(f"Best log loss: {best_loss:.6f}")
    print(f"\nEnsemble formula: {best_alpha:.2f} * lowres + {1-best_alpha:.2f} * fullres")

    if best_alpha == 0.5:
        print("\n=> Current 50/50 mixing is optimal!")
    elif best_alpha < 0.5:
        print(f"\n=> Favor fullres more ({(1-best_alpha)*100:.0f}% fullres)")
    else:
        print(f"\n=> Favor lowres more ({best_alpha*100:.0f}% lowres)")

    return best_alpha


if __name__ == "__main__":
    main()

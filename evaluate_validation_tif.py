#!/usr/bin/env python3
"""
Evaluate validation TIF predictions against ground truth using topometrics.
No post-processing (argmax results from nnUNet validation).
"""

import argparse
import sys
from pathlib import Path
from multiprocessing import Pool
from typing import List, Tuple
import numpy as np
import tifffile


def compute_single_case(args: Tuple[str, Path, Path, float, int]) -> dict:
    """Compute metrics for a single case."""
    from topometrics import compute_leaderboard_score

    case_id, pred_dir, gt_dir, surface_tolerance, ignore_label = args

    pred_path = pred_dir / f"{case_id}.tif"
    gt_path = gt_dir / f"{case_id}.tif"

    if not pred_path.exists():
        return {"case": case_id, "error": f"Pred not found: {pred_path}"}
    if not gt_path.exists():
        return {"case": case_id, "error": f"GT not found: {gt_path}"}

    try:
        pred = tifffile.imread(pred_path)
        gt = tifffile.imread(gt_path)

        if pred.shape != gt.shape:
            return {"case": case_id, "error": f"Shape mismatch: pred={pred.shape}, gt={gt.shape}"}

        rep = compute_leaderboard_score(
            predictions=pred,
            labels=gt,
            dims=(0, 1, 2),
            spacing=(1.0, 1.0, 1.0),
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
        return {"case": case_id, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate validation TIF predictions")
    parser.add_argument("--pred-dir", type=Path, required=True,
                        help="Directory containing prediction TIF files")
    parser.add_argument("--gt-dir", type=Path, required=True,
                        help="Directory containing ground truth TIF files")
    parser.add_argument("--surface-tolerance", type=float, default=2.0)
    parser.add_argument("--ignore-label", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-csv", type=Path, default=None)

    args = parser.parse_args()

    # Find prediction files
    pred_files = sorted(args.pred_dir.glob("*.tif"))
    # Filter out JSON
    case_ids = [f.stem for f in pred_files if f.suffix == ".tif"]

    print(f"Found {len(case_ids)} prediction files")
    print(f"Pred dir: {args.pred_dir}")
    print(f"GT dir: {args.gt_dir}")

    # Prepare args
    eval_args = [(cid, args.pred_dir, args.gt_dir, args.surface_tolerance, args.ignore_label)
                 for cid in case_ids]

    # Run evaluation
    from tqdm.auto import tqdm

    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = list(tqdm(pool.imap(compute_single_case, eval_args),
                              total=len(eval_args), desc="Evaluating"))
    else:
        results = [compute_single_case(a) for a in tqdm(eval_args, desc="Evaluating")]

    # Filter valid results
    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:5]:
            print(f"  {e['case']}: {e['error']}")

    if not valid:
        print("No valid results!")
        return 1

    # Compute statistics
    print(f"\n{'='*70}")
    print(f"RESULTS ({len(valid)} cases)")
    print(f"{'='*70}")

    metrics = ["leaderboard", "toposcore", "surface_dice", "voi_score"]
    for metric in metrics:
        values = [r[metric] for r in valid if not np.isnan(r.get(metric, np.nan))]
        if values:
            print(f"{metric:>15}: mean={np.mean(values):.4f}, std={np.std(values):.4f}, "
                  f"min={np.min(values):.4f}, max={np.max(values):.4f}")

    # Save CSV if requested
    if args.output_csv:
        import csv
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(valid[0].keys()))
            writer.writeheader()
            writer.writerows(valid)
        print(f"\nSaved to: {args.output_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

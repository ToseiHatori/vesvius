# Post-processing Comparison Results

## Evaluation Summary

**Model**: nnUNetTrainer__nnUNetResEncUNetMPlans__3d_lowres/fold_0 (1000 epochs)

**Validation Set**: 158 cases from fold_0

**Date**: 2026-02-11

## Results

| Method | Leaderboard | TopoScore | SurfaceDice | VOI |
|--------|-------------|-----------|-------------|-----|
| **hysteresis** | **0.5886** | **0.3185** | 0.8493 | 0.5595 |
| none (argmax) | 0.5690 | 0.2436 | 0.8605 | 0.5563 |
| threshold_05 | 0.5680 | 0.2401 | 0.8608 | 0.5562 |
| host_baseline | 0.5220 | 0.2315 | 0.7524 | 0.5406 |
| threshold_075 | 0.5075 | 0.1127 | 0.8033 | 0.5502 |

**Best Method**: hysteresis (Leaderboard: 0.5886)

## Post-processing Methods

### 1. none (Argmax)
- Simple argmax over probability channels
- Baseline method, no post-processing

### 2. threshold_05
- Threshold surface probability at 0.5
- Equivalent to argmax when background is < 0.5

### 3. threshold_075
- Threshold surface probability at 0.75
- More conservative, reduces false positives
- Significantly worse TopoScore (0.1127)

### 4. host_baseline
- From competition host solution (0.543 -> 0.562 on Public LB)
- Steps:
  1. Threshold at 0.75
  2. Frangi surfaceness filter
  3. Remove small connected components
- Lower SurfaceDice (0.7524) due to aggressive filtering

### 5. hysteresis (Best)
- From TransUNet notebook
- Parameters: t_low=0.5, t_high=0.9, z_radius=1, xy_radius=0, dust_min_size=100
- Steps:
  1. 3D Hysteresis thresholding (propagate from high confidence to low)
  2. Anisotropic morphological closing (Z direction only)
  3. Dust removal (remove_small_objects)

## Key Findings

1. **TopoScore is the differentiator**: Hysteresis achieves +0.075 better TopoScore than argmax, which is the main driver of the +0.02 Leaderboard improvement.

2. **SurfaceDice is relatively stable**: All methods except host_baseline achieve ~0.85 SurfaceDice. Aggressive thresholding hurts this metric.

3. **VOI is consistent**: All methods achieve ~0.55 VOI score with minimal variation.

4. **High threshold alone is harmful**: threshold_075 has the worst performance due to low TopoScore (0.1127), indicating many disconnected regions.

5. **Hysteresis combines benefits**: Uses low threshold (0.5) for coverage with high threshold (0.9) for seed regions, achieving better connectivity while maintaining coverage.

## Leaderboard Score Formula

```
Leaderboard = 0.3 * TopoScore + 0.35 * SurfaceDice + 0.35 * VOI_score
```

## Recommended Configuration

For submission, use hysteresis post-processing:

```python
def postprocess_hysteresis(
    probs: np.ndarray,
    t_low: float = 0.5,
    t_high: float = 0.9,
    z_radius: int = 1,
    xy_radius: int = 0,
    dust_min_size: int = 100
) -> np.ndarray
```

## Files

- `evaluate_metrics.py`: Evaluation script with all post-processing methods
- `surface-nnunet-submission.py`: Submission script with hysteresis post-processing
- `metrics_3d_lowres_fold0_1000ep.csv`: Raw evaluation results

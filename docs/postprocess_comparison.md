# Post-processing Comparison Results

## Evaluation Summary

**Model**: nnUNetTrainer__nnUNetResEncUNetMPlans__3d_lowres/fold_0 (1000 epochs)

**Validation Set**: 158 cases from fold_0

**Date**: 2026-02-11

## Results

| Method | Leaderboard | TopoScore | SurfaceDice | VOI |
|--------|-------------|-----------|-------------|-----|
| **opening_closing** | **0.6080** | **0.3423** | 0.8743 | 0.5641 |
| hysteresis | 0.5886 | 0.3185 | 0.8493 | 0.5595 |
| none (argmax) | 0.5690 | 0.2436 | 0.8605 | 0.5563 |
| threshold_05 | 0.5680 | 0.2401 | 0.8608 | 0.5562 |
| host_baseline | 0.5220 | 0.2315 | 0.7524 | 0.5406 |
| threshold_075 | 0.5075 | 0.1127 | 0.8033 | 0.5502 |

**Best Method**: opening_closing (Leaderboard: 0.6080)

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

### 5. hysteresis
- From TransUNet notebook
- Parameters: t_low=0.5, t_high=0.9, z_radius=1, xy_radius=0, dust_min_size=100
- Steps:
  1. 3D Hysteresis thresholding (propagate from high confidence to low)
  2. Anisotropic morphological closing (Z direction only)
  3. Dust removal (remove_small_objects)

### 6. opening_closing (Best)
- Hysteresisにopening処理を追加、closing構造要素を拡大
- Parameters: t_low=0.30, t_high=0.85
- Steps:
  1. 3D Hysteresis thresholding (t_high=0.85, t_low=0.30)
  2. **Opening** (構造要素: 3x3x3, connectivity=1) - 小さな突起・ノイズを除去
  3. **Closing** (構造要素: z_radius=2, xy_radius=1) - 穴埋め・断片の接続
  4. Dust removal (min_size=100)
- **TopoScore +0.024改善** (0.3185 → 0.3423): opening処理により連結性が向上

## Key Findings

1. **TopoScore is the differentiator**: opening_closing achieves +0.10 better TopoScore than argmax (0.34 vs 0.24), which is the main driver of the +0.04 Leaderboard improvement.

2. **Opening処理の効果**: ノイズを先に除去してからclosingで穴を埋めることで、よりクリーンな表面が得られる。TopoScoreが0.32→0.34に改善。

3. **SurfaceDice is relatively stable**: All methods except host_baseline achieve ~0.85 SurfaceDice. opening_closingは0.8743と最高。

4. **VOI is consistent**: All methods achieve ~0.55 VOI score with minimal variation.

5. **High threshold alone is harmful**: threshold_075 has the worst performance due to low TopoScore (0.1127), indicating many disconnected regions.

6. **Hysteresis combines benefits**: Uses low threshold for coverage with high threshold for seed regions, achieving better connectivity while maintaining coverage.

## Leaderboard Score Formula

```
Leaderboard = 0.3 * TopoScore + 0.35 * SurfaceDice + 0.35 * VOI_score
```

## Recommended Configuration

For submission, use opening_closing post-processing:

```python
def postprocess_opening_closing(probs: np.ndarray) -> np.ndarray:
    """Hysteresis + opening + closing to remove noise and fill gaps."""
    surface_prob = probs[1]

    # Hysteresis
    strong = surface_prob >= 0.85
    weak = surface_prob >= 0.30
    struct_hyst = ndi.generate_binary_structure(3, 3)
    mask = ndi.binary_propagation(strong, mask=weak, structure=struct_hyst)

    # Opening (remove small protrusions)
    struct_open = ndi.generate_binary_structure(3, 1)
    mask = ndi.binary_opening(mask, structure=struct_open)

    # Closing (fill small holes)
    struct_close = build_anisotropic_struct(2, 1)
    mask = ndi.binary_closing(mask, structure=struct_close)

    # Dust removal
    mask = remove_small_objects(mask.astype(bool), min_size=100)

    return mask.astype(np.uint8)
```

## Files

- `evaluate_metrics.py`: Evaluation script with all post-processing methods
- `surface-nnunet-submission.py`: Submission script with hysteresis post-processing
- `metrics_3d_lowres_fold0_1000ep.csv`: Raw evaluation results

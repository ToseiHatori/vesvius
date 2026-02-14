# Submission History

## 概要

Kaggle Vesuvius Challenge Surface Detection コンペティションへの提出履歴。

## 提出一覧

| # | 提出日時 (JST) | 経過時間 | Local CV | LB Score | 概要 |
|---|---------------|----------|----------|----------|------|
| 5 | 2026-02-14 04:20 | 4h 56min | - | 0.565 | 2-fold ensemble (fold_0+fold_1), 2xT4 parallel |
| 4 | 2026-02-13 14:38 | - | 0.601 | **0.565** | hysteresis optimized (t_low=0.3, t_high=0.85) |
| 3 | 2026-02-12 08:41 | 5h 20min | 0.589 | 0.549 | hysteresis, disk-saving mode |
| 2 | 2026-02-12 03:07 | - | 0.589 | - | hysteresis postprocess（失敗） |
| 1 | 2026-02-11 23:09 | 3h 39min | 0.569 | 0.530 | no TTA, argmax（ベースライン） |

※ Local CV は Leaderboard 計算式 (0.3×TopoScore + 0.35×SurfaceDice + 0.35×VOI) に基づく

## 詳細

### Submission #5 (2026-02-14)

- **Ref**: 50347563
- **Kernel**: v17
- **モデル**: nnUNetTrainer__nnUNetResEncUNetMPlans__3d_lowres fold_0 + fold_1
- **エポック数**: 1000
- **GPU**: 2x T4 (並列推論)
- **後処理**: Hysteresis thresholding (t_low=0.3, t_high=0.85)
- **Ensemble**: 確率の平均 (probability averaging)
- **経過時間**: 295.9 min (4h 56min)
- **結果**: LB 0.565 (前回と同じ、改善なし)
- **考察**: 2-fold ensemble でも LB スコアは変わらず。

### Submission #4 (2026-02-13)

- **Kernel**: v12
- **モデル**: nnUNetTrainer__nnUNetResEncUNetMPlans__3d_lowres fold_0
- **エポック数**: 1000
- **後処理**: Hysteresis thresholding (t_low=0.3, t_high=0.85)
- **改善点**: グリッドサーチでハイパーパラメータ最適化
  - fold_0, fold_1 両方で検証
  - t_low: 0.5 → 0.3（より広い領域をカバー）
  - t_high: 0.9 → 0.85（シード領域を緩和）
- **Local CV**: 0.601 (TopoScore: 0.322, SurfDice: 0.875, VOI: 0.566)
- **結果**: LB 0.549 → **0.565** (+0.016)

### Submission #3 (2026-02-12)

- **Ref**: 50319944
- **モデル**: nnUNetTrainer__nnUNetResEncUNetMPlans__3d_lowres fold_0
- **エポック数**: 1000
- **後処理**: Hysteresis thresholding (high=0.5, low=0.3)
- **改善点**: disk-saving mode で推論時のメモリ使用量を削減
- **結果**: LB 0.530 → 0.549 (+0.019)

### Submission #2 (2026-02-12)

- **Ref**: 50316602
- **モデル**: nnUNetTrainer__nnUNetResEncUNetMPlans__3d_lowres fold_0
- **エポック数**: 1000
- **後処理**: Hysteresis thresholding
- **結果**: 失敗（Kaggle Kernel メモリ不足）

### Submission #1 (2026-02-11)

- **Ref**: 50312721
- **モデル**: nnUNetTrainer__nnUNetResEncUNetMPlans__3d_lowres fold_0
- **エポック数**: 1000
- **後処理**: Simple argmax
- **結果**: LB 0.530（初回ベースライン）

## モデル情報

### nnUNet 3d_lowres fold_0 1000ep

- **学習データ**: Dataset100_VesuviusSurface
- **Configuration**: 3d_lowres
- **Trainer**: nnUNetTrainer (ResEncUNet)
- **Plans**: nnUNetResEncUNetMPlans
- **Fold**: 0
- **Epochs**: 1000
- **Local CV Dice**: 0.574

## 今後の改善案

- [x] 他の fold での学習・アンサンブル → fold_0 + fold_1 で試したが改善なし
- [ ] 3d_fullres での学習
- [ ] TTA (Test Time Augmentation) の追加
- [ ] より多くの fold でのアンサンブル (fold_2, fold_3, fold_4)
- [ ] 異なるモデルアーキテクチャとのアンサンブル

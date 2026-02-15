# 実験結果サマリー

Vesuvius Surface Detection の実験結果をまとめたドキュメント。

## モデル一覧

| モデル | Configuration | Fold | Epochs | Val Dice | 備考 |
|--------|---------------|------|--------|----------|------|
| nnUNetResEncUNet | 3d_lowres | 0 | 1000 | 0.574 | 完了 |
| nnUNetResEncUNet | 3d_lowres | 1 | 1000 | 0.579 | 完了 |
| nnUNetResEncUNet | 3d_lowres | 0 | 2000 | - | 完了 |
| nnUNetResEncUNet | 3d_lowres | 2-4 | - | - | 未学習 |
| nnUNetResEncUNet | 3d_fullres | all | 1000 | - | 完了、LB 0.565 |

## 評価結果

### 3d_lowres fold_0 / fold_1 (Postprocess: none)

| Fold | Cases | Leaderboard | TopoScore | Surface Dice | VOI Score |
|------|-------|-------------|-----------|--------------|-----------|
| fold_0 | 158 | 0.5690 | 0.2438 | 0.8605 | 0.5563 |
| fold_1 | 157 | 0.5674 | 0.2362 | 0.8627 | 0.5560 |

**Leaderboard計算式**: `0.3 * TopoScore + 0.35 * SurfaceDice + 0.35 * VOI_score`

### 後処理比較 (fold_0)

| Method | Leaderboard | TopoScore | SurfaceDice | VOI |
|--------|-------------|-----------|-------------|-----|
| **hysteresis** | **0.5886** | **0.3185** | 0.8493 | 0.5595 |
| none (argmax) | 0.5690 | 0.2436 | 0.8605 | 0.5563 |
| threshold_05 | 0.5680 | 0.2401 | 0.8608 | 0.5562 |
| host_baseline | 0.5220 | 0.2315 | 0.7524 | 0.5406 |
| threshold_075 | 0.5075 | 0.1127 | 0.8033 | 0.5502 |

**Best (default)**: hysteresis (t_low=0.5, t_high=0.9)

### Hysteresis グリッドサーチ結果

fold_0, fold_1 両方でグリッドサーチを実施し、最適パラメータを特定。

| Fold | Best t_low | Best t_high | Leaderboard |
|------|------------|-------------|-------------|
| fold_0 | 0.3 | 0.85 | **0.6010** |
| fold_1 | 0.3 | 0.75 | 0.5996 |

**最適パラメータ**: t_low=0.3, t_high=0.85（両foldで安定）

詳細: [hysteresis_grid_search.csv](results/hysteresis_grid_search.csv), [hysteresis_grid_search_fold1.csv](hysteresis_grid_search_fold1.csv)

詳細: [postprocess_comparison.md](postprocess_comparison.md)

### 2000エポック評価結果 (fold_0)

| Epochs | Postprocess | t_low | t_high | Leaderboard | TopoScore | SurfaceDice | VOI |
|--------|-------------|-------|--------|-------------|-----------|-------------|-----|
| 1000 | hysteresis | 0.3 | 0.85 | 0.6010 | - | - | - |
| 2000 | hysteresis | 0.5 | 0.9 | 0.5920 | 0.3225 | 0.8539 | 0.5610 |
| 2000 | hysteresis | 0.3 | 0.85 | **0.6052** | 0.3334 | 0.8769 | 0.5663 |

**2000エポックで+0.0042の改善** (0.6010 → 0.6052)

## Kaggle提出結果

詳細: [submission_history.md](submission_history.md)

## 知見・考察

### TopoScoreが改善の鍵
- Surface Dice (~0.86) と VOI Score (~0.56) は手法間で安定
- TopoScore (0.24 → 0.32) の改善がLeaderboard向上に直結
- Hysteresis後処理で連結性が改善し、TopoScoreが+0.08向上

### 後処理の効果
- 高い閾値単独 (0.75) は逆効果（TopoScore: 0.11）
- Hysteresisは低閾値でカバレッジ維持 + 高閾値でシード → 連結性向上

### Local CV vs LB
- Local CV: 0.569 → LB: 0.530（argmax）
- Local CV: 0.589 → LB: 0.549（hysteresis）
- 約0.04の差があるが、傾向は一致

### 3d_fullres vs 3d_lowres
- 3d_fullres fold_all (全データ学習): LB 0.565
- 3d_lowres 2-fold ensemble: LB 0.565
- 高解像度でも改善なし。データ量やモデルの表現力ではなく、別のボトルネックがある可能性

### 推論並列度 (npp, nps)
- npp=1, nps=1: 295.9 min
- npp=2, nps=2: 346.1 min (+50min、遅くなった)
- T4 GPU ではメモリ制約により並列化のオーバーヘッドが大きい。npp=1, nps=1 が最適。

### エポック数の影響
- 1000ep → 2000ep で Local CV が 0.6010 → 0.6052 (+0.0042) 改善
- 過学習の兆候なし。さらなるエポック数増加の余地あり

## 結果ファイル

CSVファイルは `docs/results/` に保存:
- `fold0_evaluation.csv` - fold 0評価結果
- `fold1_evaluation.csv` - fold 1評価結果
- `postprocess_comparison.csv` - 後処理比較結果
- `hysteresis_grid_search.csv` - ハイパーパラメータサーチ結果
- `2000ep_fold0_hysteresis.csv` - 2000エポック評価結果 (デフォルトパラメータ)
- `2000ep_fold0_hysteresis_optimal.csv` - 2000エポック評価結果 (最適パラメータ)

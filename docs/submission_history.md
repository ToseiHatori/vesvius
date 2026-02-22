# Submission History

## 概要

Kaggle Vesuvius Challenge Surface Detection コンペティションへの提出履歴。

## 提出一覧

| # | 提出日時 (JST) | 経過時間 | Local CV | LB Score | 概要 |
|---|---------------|----------|----------|----------|------|
| 17 | 2026-02-22 17:44 | 6h 45min | - | 0.576 | TTA有効 + step_size=0.5 (v59, 悪化) |
| 16 | 2026-02-22 | timeout | - | - | TTA有効 + step_size=0.3 (v56, timeout) |
| 15 | 2026-02-21 12:54 | 3h 28min | - | 0.580 | step_size=0.25 (v51, 悪化) |
| 14 | 2026-02-21 06:48 | 4h 05min | - | **0.582** | Python API + 4モデル (lowres+fullres × fold_0&1) |
| 13 | 2026-02-21 02:21 | - | - | pending | lowres+fullres ensemble (v45, timeout) |
| 12 | 2026-02-21 02:06 | - | - | 0.575 | **fullres 2000ep fold_0&1 (v43)** |
| 11 | 2026-02-17 14:18 | 4h 20min | - | 0.568 | 2000ep + opening_closing + fold_0&1 ensemble |
| 10 | 2026-02-16 17:59 | 5h 17min | - | 0.565 | パイプライン処理 ダブルバッファリング修正 |
| 9 | 2026-02-16 04:05 | 5h 13min | - | 0.565 | パイプライン処理（バグあり、遅くなった） |
| 8 | 2026-02-16 02:45 | 4h 10min | - | 0.565 | step_size=0.3 追加 |
| 7 | 2026-02-15 00:42 | 5h 46min | - | 0.565 | npp=2, nps=2 実験（遅くなった） |
| 6 | 2026-02-15 12:25 | 5h 10min | - | 0.565 | 3d_fullres fold_all (single model) |
| 5 | 2026-02-14 04:20 | 4h 56min | - | 0.565 | 2-fold ensemble (fold_0+fold_1), 2xT4 parallel |
| 4 | 2026-02-13 14:38 | - | 0.601 | **0.565** | hysteresis optimized (t_low=0.3, t_high=0.85) |
| 3 | 2026-02-12 08:41 | 5h 20min | 0.589 | 0.549 | hysteresis, disk-saving mode |
| 2 | 2026-02-12 03:07 | - | 0.589 | - | hysteresis postprocess（失敗） |
| 1 | 2026-02-11 23:09 | 3h 39min | 0.569 | 0.530 | no TTA, argmax（ベースライン） |

※ Local CV は Leaderboard 計算式 (0.3×TopoScore + 0.35×SurfaceDice + 0.35×VOI) に基づく

## 詳細

### Submission #17 (2026-02-22) - v59: TTA有効 + step_size=0.5

- **Ref**: 50504688
- **Kernel**: v59
- **モデル**: nnUNetTrainer_2000epochs (3d_lowres + 3d_fullres) × (fold_0 + fold_1) = 4モデル
- **エポック数**: 2000
- **GPU**: 2x T4 (並列推論)
- **後処理**: Opening/Closing
- **変更点**:
  - **TTA (Test Time Augmentation) 有効化**: 8方向ミラーリングで予測を平均化
  - **step_size = 0.5** (50% overlap): v56 (step=0.3) がtimeoutしたため拡大
- **経過時間**: 404.8 min (6h 45min)
- **結果**: LB **0.576** (-0.006 from best)
- **考察**:
  - TTAを有効にしたがスコアは悪化
  - step_size=0.5は粗すぎて、タイル境界での品質低下が原因と考えられる
  - TTAの恩恵よりstep_size拡大のデメリットが大きかった
  - 時間も6h 45minとベスト(4h 05min)より長い

### Submission #16 (2026-02-22) - v56: TTA有効 (timeout)

- **Kernel**: v56
- **モデル**: nnUNetTrainer_2000epochs (3d_lowres + 3d_fullres) × (fold_0 + fold_1) = 4モデル
- **エポック数**: 2000
- **GPU**: 2x T4 (並列推論)
- **後処理**: Opening/Closing
- **変更点**:
  - **TTA (Test Time Augmentation) 有効化**: 8方向ミラーリングで予測を平均化
  - **step_size = 0.3** (70% overlap)
- **結果**: **timeout** - 計算量が多すぎた

### Submission #15 (2026-02-21) - v51: step_size=0.25

- **Ref**: 50490393
- **Kernel**: v51
- **モデル**: nnUNetTrainer_2000epochs (3d_lowres + 3d_fullres) × (fold_0 + fold_1) = 4モデル
- **エポック数**: 2000
- **GPU**: 2x T4 (並列推論)
- **後処理**: Opening/Closing
- **変更点**: step_size を 0.3 → 0.25 に変更 (75% overlap)
- **経過時間**: 207.8 min (3h 28min)
- **結果**: LB **0.580** (-0.002)
- **考察**: step_size を小さくしてオーバーラップを増やしたが、スコアは悪化。推論時間も短くなったが、これは早期終了した可能性あり。v47 (step_size=0.3) が最適。

### Submission #14 (2026-02-21) - v47: Best Score

- **Ref**: 50484220
- **Kernel**: v47
- **モデル**: nnUNetTrainer_2000epochs (3d_lowres + 3d_fullres) × (fold_0 + fold_1) = 4モデル
- **エポック数**: 2000
- **GPU**: 2x T4 (並列推論)
- **後処理**: Opening/Closing
- **Ensemble**: 確率の平均 (probability averaging)
- **経過時間**: 245.1 min (4h 05min)
- **結果**: LB **0.582** (+0.014)
- **主な変更点**:
  - **Python API への移行**: CLI (`nnUNetv2_predict`) から `nnUNetPredictor` Python API へ
  - **4モデルアンサンブル**: 3d_lowres + 3d_fullres の両方を使用
  - **バッチ処理**: 5ケースずつ処理してメモリ効率を改善
  - モデルロード回数: 4回のみ（CLI版は 4 × N_cases 回）
- **考察**: Python API への移行と 3d_fullres 追加で大幅な精度改善。ローカルベンチマーク（RTX 3090）では 24.86s/case、T4では約4時間で完了。

### Submission #12 (2026-02-21) - v43: fullres 単独

- **Ref**: 50483817
- **Kernel**: v43
- **モデル**: 3d_fullres fold_0 + fold_1 (2モデル)
- **エポック数**: 2000
- **GPU**: 2x T4 (並列推論)
- **後処理**: Opening/Closing
- **Ensemble**: 確率の平均 (probability averaging)
- **結果**: LB **0.575** (+0.007 from lowres単独)
- **考察**:
  - fullres 2000ep fold_0&1 アンサンブルで LB 0.575
  - lowres 2000ep (LB 0.568) より +0.007 改善
  - **重要**: fullres + lowres の4モデル ensemble (v47) で更に +0.007 → LB 0.582
  - fullres と lowres は異なるスケールの特徴を捉えており、相補効果あり

### Submission #11 (2026-02-17)

- **Ref**: 50405194
- **Kernel**: v41
- **モデル**: nnUNetTrainer_2000epochs__nnUNetResEncUNetMPlans__3d_lowres fold_0 + fold_1
- **エポック数**: 2000
- **GPU**: 2x T4 (並列推論)
- **後処理**: Opening/Closing (disk r=1 open → close)
- **Ensemble**: 確率の平均 (probability averaging)
- **経過時間**: 259.5 min (4h 20min)
- **結果**: LB **0.568** (+0.003)
- **考察**: 2000エポック + opening_closing 後処理の組み合わせでスコア改善。hysteresis (0.565) より opening_closing が効果的。

### Submission #10 (2026-02-16)

- **Ref**: 50388506
- **Kernel**: v35
- **モデル**: nnUNetTrainer__nnUNetResEncUNetMPlans__3d_lowres fold_0 + fold_1
- **エポック数**: 1000
- **GPU**: 2x T4 (並列推論)
- **後処理**: Hysteresis thresholding (t_low=0.3, t_high=0.85)
- **変更点**: パイプライン処理のレースコンディションをダブルバッファリングで修正
  - `pred_dirs_buf0`, `pred_dirs_buf1` を交互使用
  - 同じバッファの後処理完了を待ってから次の推論を開始
- **経過時間**: 317.4 min (5h 17min)
- **結果**: LB 0.565 (スコア変わらず)
- **考察**: バグ修正後も速度改善なし。パイプライン処理自体のオーバーヘッドが大きい。

### Submission #9 (2026-02-16)

- **Ref**: 50379541
- **Kernel**: v31
- **モデル**: nnUNetTrainer__nnUNetResEncUNetMPlans__3d_lowres fold_0 + fold_1
- **エポック数**: 1000
- **GPU**: 2x T4 (並列推論)
- **後処理**: Hysteresis thresholding (t_low=0.3, t_high=0.85)
- **変更点**: パイプライン処理を追加（ThreadPoolExecutor で前処理・後処理を推論と並列化）
  - **バグあり**: `pred_dirs` が共有されており、後処理中に次の推論が上書きする可能性
- **経過時間**: 312.8 min (5h 13min)
- **結果**: LB 0.565 (スコア変わらず、小数点以下は悪化の可能性)
- **考察**: パイプライン処理を追加したが逆に遅くなった。レースコンディションにより精度も悪化した可能性。

### Submission #8 (2026-02-16)

- **Ref**: 50379312
- **Kernel**: v30
- **モデル**: nnUNetTrainer__nnUNetResEncUNetMPlans__3d_lowres fold_0 + fold_1
- **エポック数**: 1000
- **GPU**: 2x T4 (並列推論)
- **後処理**: Hysteresis thresholding (t_low=0.3, t_high=0.85)
- **変更点**: step_size=0.3 (70% overlap) を追加
- **経過時間**: 250.4 min (4h 10min)
- **結果**: LB 0.565 (スコア変わらず)
- **考察**: step_size=0.3 でオーバーラップを増やしたがスコア変わらず。この時点では最速。

### Submission #7 (2026-02-15)

- **Ref**: 50359320
- **Kernel**: v19
- **モデル**: nnUNetTrainer__nnUNetResEncUNetMPlans__3d_lowres fold_0 + fold_1
- **エポック数**: 1000
- **GPU**: 2x T4 (並列推論)
- **後処理**: Hysteresis thresholding (t_low=0.3, t_high=0.85)
- **変更点**: nnUNetv2_predict の並列度を変更 (npp=1, nps=1 → npp=2, nps=2)
- **経過時間**: 346.1 min (5h 46min)
- **結果**: LB 0.565 (スコア変わらず)
- **考察**: 並列度を上げたが逆に遅くなった (+50min)。T4のメモリ制約でオーバーヘッドが大きい可能性。npp=1, nps=1 に戻した。

### Submission #6 (2026-02-15)

- **Ref**: 50361657
- **Kernel**: v22
- **モデル**: nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres fold_all
- **エポック数**: 1000
- **GPU**: 1x T4
- **後処理**: Hysteresis thresholding (t_low=0.3, t_high=0.85)
- **経過時間**: 309.8 min (5h 10min)
- **結果**: LB 0.565 (前回と同じ、改善なし)
- **考察**: 3d_fullres (高解像度) でも 3d_lowres と同等のスコア。fold_all（全データ学習）も効果なし。

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
- [x] 3d_fullres での学習 → fold_all で試したが改善なし (LB 0.565)
- [x] パイプライン処理による高速化 → 逆に遅くなった、逐次処理が最適
- [x] opening_closing 後処理の提出検証 → LB 0.568 (hysteresis 0.565 より +0.003 改善)
- [x] **Python API + 4モデルアンサンブル** → LB **0.582** (+0.014 大幅改善！)
- [x] step_size=0.25 (75% overlap) → LB 0.580 (悪化、0.3が最適)
- [x] TTA (Test Time Augmentation) の追加 → step_size=0.3でtimeout、step_size=0.5でLB 0.576 (悪化)
- [ ] より多くの fold でのアンサンブル (fold_2, fold_3, fold_4)
- [ ] 異なるモデルアーキテクチャとのアンサンブル
- [ ] config-wise モデルローディング（v48で実装済み、メモリ効率改善）

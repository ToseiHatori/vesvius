# 実験結果サマリー

Vesuvius Surface Detection の実験結果をまとめたドキュメント。

## モデル一覧

| モデル | Configuration | Fold | Epochs | Val Dice | 備考 |
|--------|---------------|------|--------|----------|------|
| nnUNetResEncUNet | 3d_lowres | 0 | 1000 | 0.574 | 完了 |
| nnUNetResEncUNet | 3d_lowres | 1 | 1000 | 0.579 | 完了 |
| nnUNetResEncUNet | 3d_lowres | 0 | 2000 | 0.580 | 完了 |
| nnUNetResEncUNet | 3d_lowres | 1 | 2000 | 0.579 | 完了 |
| nnUNetResEncUNet | 3d_lowres | 2-4 | - | - | 未学習 |
| nnUNetResEncUNet | 3d_fullres | all | 1000 | - | 完了、LB 0.565 |
| nnUNetResEncUNet | 3d_fullres | 0 | 2000 | 0.603 | 完了 |
| nnUNetResEncUNet | 3d_fullres | 1 | 2000 | 0.603 | 完了 |
| nnUNetResEncUNet | 3d_fullres | 0 | 4000 | 0.606 | 完了 |
| nnUNetResEncUNet | 3d_lowres | 1 | 4000 | 0.603 | 完了 |

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
| **opening_closing** | **0.6080** | **0.3423** | 0.8743 | 0.5641 |
| hysteresis | 0.5886 | 0.3185 | 0.8493 | 0.5595 |
| none (argmax) | 0.5690 | 0.2436 | 0.8605 | 0.5563 |
| threshold_05 | 0.5680 | 0.2401 | 0.8608 | 0.5562 |
| host_baseline | 0.5220 | 0.2315 | 0.7524 | 0.5406 |
| threshold_075 | 0.5075 | 0.1127 | 0.8033 | 0.5502 |

**Best**: opening_closing (hysteresis + opening + closing)
- Opening処理（3x3x3, connectivity=1）で小さな突起・ノイズを除去
- Closing構造要素を拡大（z_radius=2, xy_radius=1）で穴埋め強化

### Hysteresis グリッドサーチ結果

fold_0, fold_1 両方でグリッドサーチを実施し、最適パラメータを特定。

| Fold | Best t_low | Best t_high | Leaderboard |
|------|------------|-------------|-------------|
| fold_0 | 0.3 | 0.85 | **0.6010** |
| fold_1 | 0.3 | 0.75 | 0.5996 |

**最適パラメータ**: t_low=0.3, t_high=0.85（両foldで安定）

詳細: [hysteresis_grid_search.csv](results/hysteresis_grid_search.csv), [hysteresis_grid_search_fold1.csv](hysteresis_grid_search_fold1.csv)

詳細: [postprocess_comparison.md](postprocess_comparison.md)

### 2000エポック評価結果 (fold_0 / fold_1 比較)

#### fold_0 / fold_1 完全比較

| Fold | Postprocess | Leaderboard | TopoScore | SurfaceDice | VOI |
|------|-------------|-------------|-----------|-------------|------|
| fold_0 | none | 0.5715 | 0.2464 | 0.8630 | 0.5586 |
| fold_0 | hysteresis | 0.6052 | 0.3334 | 0.8769 | 0.5663 |
| fold_0 | **opening_closing** | **0.6062** | **0.3423** | 0.8736 | 0.5650 |
| fold_1 | none | 0.5684 | 0.2394 | 0.8639 | 0.5548 |
| fold_1 | hysteresis | 0.5921 | 0.3030 | 0.8703 | 0.5618 |
| fold_1 | **opening_closing** | **0.5943** | **0.3144** | 0.8675 | 0.5610 |

#### 後処理効果の比較

| 比較 | Fold | Leaderboard改善 | TopoScore改善 |
|------|------|-----------------|---------------|
| none → hysteresis | fold_0 | +0.0337 (+5.9%) | +0.0870 (+35.3%) |
| none → hysteresis | fold_1 | +0.0237 (+4.2%) | +0.0636 (+26.6%) |
| hysteresis → opening_closing | fold_0 | +0.0010 (+0.2%) | +0.0089 (+2.7%) |
| hysteresis → opening_closing | fold_1 | +0.0022 (+0.4%) | +0.0114 (+3.8%) |

**結論**:
- hysteresisは大幅な改善効果（特にTopoScore +26〜35%）
- opening_closingはhysteresisに対してさらに微小だが一貫した改善（TopoScore +2.7〜3.8%）
- 両foldで一貫した傾向を確認

## Kaggle提出結果

### **v47: Python API + 4モデルアンサンブル (Best: LB 0.582)**

v47で大幅なスコア改善を達成。

| 項目 | 値 |
|------|-----|
| **LB Score** | **0.582** (+0.014 from v41) |
| 経過時間 | 4h 05min |
| モデル | 3d_lowres × fold_0&1 + 3d_fullres × fold_0&1 = 4モデル |
| Ensemble | 確率平均 (probability averaging) |
| 後処理 | Opening/Closing |

**主な技術改善:**
1. **CLI → Python API**: `nnUNetv2_predict` コマンドから `nnUNetPredictor` クラスへ
2. **モデルロード最適化**: 4回のみ（CLI版は 4 × N_cases 回）
3. **4モデルアンサンブル**: lowres + fullres の相補効果
4. **バッチ処理**: 5ケースずつ処理してメモリ効率改善

**なぜ速くなったか:**
- CLI版: 各ケースごとにモデルをロード → 巨大なオーバーヘッド
- Python API版: モデルを一度ロードして全ケースに適用 → ロード時間を償却

詳細: [submission_history.md](submission_history.md)

## 知見・考察

### TopoScoreが改善の鍵
- Surface Dice (~0.86) と VOI Score (~0.56) は手法間で安定
- TopoScore (0.24 → 0.32) の改善がLeaderboard向上に直結
- Hysteresis後処理で連結性が改善し、TopoScoreが+0.08向上

### 後処理の効果
- 高い閾値単独 (0.75) は逆効果（TopoScore: 0.11）
- Hysteresisは低閾値でカバレッジ維持 + 高閾値でシード → 連結性向上
- **opening_closing**: ノイズを先に除去 (opening) してからclosing → TopoScore +0.009改善

### Local CV vs LB
- Local CV: 0.569 → LB: 0.530（argmax）
- Local CV: 0.589 → LB: 0.549（hysteresis）
- 約0.04の差があるが、傾向は一致

### 3d_fullres vs 3d_lowres
- 3d_fullres fold_all (全データ学習): LB 0.565
- 3d_lowres 2-fold ensemble: LB 0.565
- 単独では改善なし。しかし **lowres + fullres の4モデルアンサンブルで LB 0.582** 達成
- lowres と fullres は異なるスケールの特徴を捉えており、相補効果がある

### 3d_fullres 2000エポック評価 (2026/02/19-21)

#### fold_0 / fold_1 完全比較

| Fold | Postprocess | Leaderboard | TopoScore | SurfaceDice | VOI |
|------|-------------|-------------|-----------|-------------|-----|
| fold_0 | none | 0.5683 | 0.2340 | 0.8644 | 0.5588 |
| fold_0 | hysteresis | 0.5886 | 0.3168 | 0.8473 | 0.5630 |
| fold_0 | **opening_closing** | **0.6029** | 0.3327 | 0.8728 | 0.5646 |
| fold_1 | none | 0.5686 | 0.2326 | 0.8677 | 0.5576 |
| fold_1 | hysteresis | 0.5898 | 0.3111 | 0.8579 | 0.5606 |
| fold_1 | **opening_closing** | **0.6030** | **0.3354** | 0.8724 | 0.5629 |

**結論**: 3d_fullres fold_0/fold_1 両方で Leaderboard 0.603 達成。3d_lowres (0.606/0.594) と比較して同等〜微減。
- fullresの高解像度による改善は見られず
- ensemble (lowres + fullres) での相補効果に期待

**Kaggle提出結果 (v43)**:
- 3d_fullres 2000ep fold_0&1 ensemble: **LB 0.575** (+0.007 from lowres単独の 0.568)
- さらに lowres + fullres の4モデル ensemble (v47): **LB 0.582** (+0.014)

### 3d_fullres 4000エポック評価 (2026/02/24)

4000エポックまで学習を延長した 3d_fullres fold_0 の評価結果。

| Postprocess | Leaderboard | TopoScore | SurfaceDice | VOI |
|-------------|-------------|-----------|-------------|------|
| none | 0.5764 | 0.2595 | 0.8658 | 0.5593 |
| hysteresis | 0.5944 | 0.3245 | 0.8584 | 0.5618 |
| **opening_closing** | **0.6063** | **0.3409** | **0.8758** | **0.5643** |

#### 2000ep → 4000ep 比較 (3d_fullres fold_0, opening_closing)

| Epochs | Leaderboard | TopoScore | SurfaceDice | VOI |
|--------|-------------|-----------|-------------|------|
| 2000 | 0.6029 | 0.3327 | 0.8728 | 0.5646 |
| 4000 | 0.6063 | 0.3409 | 0.8758 | 0.5643 |
| **差分** | **+0.0034** | **+0.0082** | **+0.0030** | **-0.0003** |

**結論**: 4000エポックで Leaderboard +0.003 の微改善。主にTopoScore (+0.008) とSurfaceDice (+0.003) の向上。VOIはほぼ横ばい。2000ep→4000epの改善幅は1000ep→2000ep (+0.004) と同程度で、過学習の兆候なし。

### 3d_lowres 4000エポック評価 (2026/02/27)

4000エポックまで学習を延長した 3d_lowres fold_1 の評価結果。

| Postprocess | Leaderboard | TopoScore | SurfaceDice | VOI |
|-------------|-------------|-----------|-------------|------|
| none | 0.5697 | 0.2413 | 0.8629 | 0.5575 |
| hysteresis | 0.5945 | 0.3212 | 0.8593 | 0.5633 |
| **opening_closing** | **0.6030** | **0.3354** | **0.8728** | **0.5625** |

#### 2000ep → 4000ep 比較 (3d_lowres fold_1, opening_closing)

| Epochs | Leaderboard | TopoScore | SurfaceDice | VOI |
|--------|-------------|-----------|-------------|------|
| 2000 | 0.5943 | 0.3144 | 0.8675 | 0.5610 |
| 4000 | 0.6030 | 0.3354 | 0.8728 | 0.5625 |
| **差分** | **+0.0087** | **+0.0210** | **+0.0053** | **+0.0015** |

**結論**: 4000エポックで Leaderboard **+0.0087** の改善（fullresの+0.003より大きい）。特にTopoScore (+0.021) の大幅な向上が特徴的。lowresは学習データサイズが小さいため、エポック数増加の恩恵が大きい可能性。

### 4000エポック総括 (2000ep vs 4000ep)

| Config | Fold | 2000ep | 4000ep | 差分 |
|--------|------|--------|--------|------|
| 3d_fullres | fold_0 | 0.6029 | 0.6063 | **+0.0034** |
| 3d_fullres | fold_1 | 0.6030 | - | - |
| 3d_lowres | fold_0 | 0.6062 | - | - |
| 3d_lowres | fold_1 | 0.5943 | 0.6030 | **+0.0087** |

**知見**:
- 4000エポックは2000エポックに対して一貫した改善効果
- lowres fold_1 の改善幅 (+0.0087) が最も大きく、fullres fold_0 (+0.0034) の約2.5倍
- lowres fold_1 は2000epでは0.5943と他より低かったが、4000epで0.6030まで改善し fullres 並みに
- 過学習の兆候は見られず、さらなるエポック増加も有効な可能性

### 推論並列度 (npp, nps)
- npp=1, nps=1: 295.9 min
- npp=2, nps=2: 346.1 min (+50min、遅くなった)
- T4 GPU ではメモリ制約により並列化のオーバーヘッドが大きい。npp=1, nps=1 が最適。

### パイプライン処理実験
推論と前処理・後処理の並列化を試行。

| 方式 | 経過時間 | LB Score | 備考 |
|------|----------|----------|------|
| 逐次処理 + step_size=0.3 | **250.4 min** | 0.565 | ベースライン |
| パイプライン（バグあり） | 312.8 min | 0.565 | pred_dirs共有によるレースコンディション |
| パイプライン（ダブルバッファリング） | 317.4 min | 0.565 | バグ修正後も改善なし |

**結論**: Kaggle T4環境ではパイプライン処理のオーバーヘッドが大きく、逐次処理の方が速い。
- ThreadPoolExecutor のスレッド管理コスト
- メモリ競合による性能低下
- 単純な逐次処理に戻すのが最適

### エポック数の影響
- 1000ep → 2000ep で Local CV が 0.6010 → 0.6052 (+0.0042) 改善
- 過学習の兆候なし。さらなるエポック数増加の余地あり

### Python API vs CLI (v47の教訓)

nnUNet の推論方法比較:

| 方式 | モデルロード | 経過時間 | 備考 |
|------|------------|----------|------|
| CLI (`nnUNetv2_predict`) | ケースごと | 4h 20min | シンプルだがオーバーヘッド大 |
| **Python API (`nnUNetPredictor`)** | **1回のみ** | **4h 05min** | 高速、4モデル同時ロード可能 |

**Python API の利点:**
- モデルロード時間の償却（数十秒 × N_cases → 数十秒 × 1回）
- 複数モデルの同時保持が可能
- 確率出力の直接取得（NPZファイル経由不要）
- メモリ管理の柔軟性（バッチサイズ調整）

## 結果ファイル

CSVファイルは `docs/results/` に保存:
- `fold0_evaluation.csv` - fold 0評価結果
- `fold1_evaluation.csv` - fold 1評価結果
- `postprocess_comparison.csv` - 後処理比較結果
- `hysteresis_grid_search.csv` - ハイパーパラメータサーチ結果
- `2000ep_fold0_hysteresis.csv` - 2000エポック評価結果 (デフォルトパラメータ)
- `2000ep_fold0_hysteresis_optimal.csv` - 2000エポック評価結果 (最適パラメータ)
- `eval_fold0_2000ep_comparison.csv` - fold_0 2000エポック none/opening_closing比較
- `eval_fold1_2000ep_none.csv` - fold_1 2000エポック none評価
- `eval_fold1_2000ep_open_close.csv` - fold_1 2000エポック opening_closing評価

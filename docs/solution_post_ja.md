# Vesuvius Challenge Surface Detection - Solution (Gold Medal)

## 1. Summary

nnUNetベースの3Dセグメンテーション。
3d_lowresと3d_fullresの4モデルアンサンブル + opening/closing後処理
- Public LB: 0.578
- Private LB: 0.613

---

## 2. Solution Overview

```
【パイプライン】

入力（3D CT volume）
    ↓
nnUNet 前処理（正規化、リサンプリング）
    ↓
4モデル推論（2x T4 GPU並列）
  ├─ 3d_lowres fold_0 (2000ep) × 0.2
  ├─ 3d_lowres fold_1 (4000ep) × 0.2
  ├─ 3d_fullres fold_0 (4000ep) × 0.3
  └─ 3d_fullres fold_1 (2000ep) × 0.3
    ↓
確率マップの重み付き平均
    ↓
後処理
  ├─ Hysteresis thresholding (t_low=0.3, t_high=0.85)
  ├─ Opening（ノイズ除去）
  └─ Closing（穴埋め）
    ↓
出力（3Dセグメンテーションマスク）
```

**ポイント**
- nnUNetをほぼデフォルト設定で使用
- lowres + fullres の異なるスケールを組み合わせ
- 後処理でTopoScore（トポロジー品質）を大幅改善

---

## 3. Model & Training

**Configuration比較**

|              | 3d_lowres | 3d_fullres |
|--------------|-----------|------------|
| Patch size   | 128³      | 128³       |
| Spacing      | 1.56 mm   | 1.0 mm     |
| 実効範囲     | ~200 mm³  | ~128 mm³   |
| Image size   | 205³      | 320³       |

※ 同じパッチサイズでもspacingの違いで捉える物理的範囲が異なる

**Fold戦略**
- nnUNetデフォルトの5-fold CVから2 foldのみ使用（fold_0, fold_1）
- 理由: 時間制約と、2 foldでも十分な多様性が得られた
- 学習データ: 628ケース、検証データ: 157-158ケース/fold

**エポック数**
- ベース: 2000 epochs
- 一部モデルは4000 epochsまで延長
- 全モデル4000 epochs学習したかったけど時間が足りなかった
- 最終的に使用したのは以下の4モデル

| Config     | Fold   | Epochs | CV (opening_closing) |
|------------|--------|--------|----------------------|
| 3d_lowres  | fold_0 | 2000   | 0.606                |
| 3d_lowres  | fold_1 | 4000   | 0.603                |
| 3d_fullres | fold_0 | 4000   | 0.606                |
| 3d_fullres | fold_1 | 2000   | 0.603                |

**エポック数の効果**
- 1000ep → 2000ep:   
  - 3d_lowres fold_0: 0.601 → 0.605 (+0.004)
- 2000ep → 4000ep:
  - 3d_fullres fold_0: 0.603 → 0.606 (+0.003)
  - 3d_lowres fold_1: 0.594 → 0.603 (+0.009)

---

## 4. Inference & Post-processing

**Sliding Window設定**
- step_size = 0.3（70% overlap）

**TTA (Test Time Augmentation)**
- 8方向ミラーリング
- TTAありで推論を行い、推論時間が9時間に近づいたらTTAなし推論に切り替え

**後処理パイプライン**
```
確率マップ → Hysteresis → Opening → Closing → Dust Removal → 最終マスク
```

**Step 1: 3D Hysteresis Thresholding**
- t_high = 0.85: 確信度の高い領域（シード）
- t_low = 0.30: 広い領域をカバー
- 構造要素: generate_binary_structure(3, 3) - 26連結
- 処理: binary_propagation で strong から weak へ伝播

**Step 2: Opening（ノイズ除去）**
- 構造要素: generate_binary_structure(3, 1) - 6連結
- 効果: 小さな突起・ノイズを除去

**Step 3: Anisotropic Closing（穴埋め）**
- 構造要素: z_radius=2, xy_radius=1（異方性）
- z方向を大きくした理由: スライス間の連結性を強化

**Step 4: Dust Removal**
- min_size = 100 voxels
- 効果: 小さな孤立領域を除去

**後処理の効果比較**（2000ep fold_0）

| Method | Competition Score | TopoScore | 改善幅 |
|--------|-------------|-----------|--------|
| なし (argmax) | 0.571 | 0.246 | - |
| Hysteresis のみ | 0.592 | 0.322 | +0.021 |
| + Opening/Closing | 0.606 | 0.342 | +0.035 |

---

## 5. Scores

**スコア推移**

| 段階 | 変更内容 | Public | Private |
|------|----------|--------|---------|
| ベースライン | 1000ep, 1モデル, argmax | 0.530 | 0.552 |
| +後処理 | hysteresis | 0.549 | 0.570 |
| +2-fold | fold_0+1 ensemble | 0.565 | 0.590 |
| +opening_closing | 後処理改善 | 0.565 | 0.593 |
| +2000ep | エポック増加 | 0.568 | 0.593 |
| +fullres | fullres 2モデルのみ | 0.575 | 0.598 |
| +4モデル | lowres+fullres ensemble | 0.582 | 0.606 |
| +重み調整 | fullres 60%, lowres 40% | 0.583 | 0.605 |
| +TTA | 8方向ミラーリング | 0.584 | 0.607 |
| +4000ep | 最終提出 | 0.578 | 0.613 |


---

## 6. What Worked / What Didn't Work

### What Worked

- **nnUNetのデフォルト設定**: ほぼ設定変更なしで強力なベースラインが得られた
- **4モデルアンサンブル (lowres + fullres)**: 単体では同等性能、組み合わせで+0.014改善
- **後処理 (Hysteresis + Opening/Closing)**: TopoScoreが0.24→0.34へ大幅改善、Leaderboardで+0.037
- **エポック数増加 (1000→2000→4000)**: 過学習なく一貫して改善
- **TTA (Test Time Augmentation)**: Publicでは効果見えず、Privateで+0.007。

### What Didn't Work

- **推論並列度の増加 (npp=2, nps=2)**: T4のメモリ制約で逆効果（+50min遅くなった）
- **step_size変更**: 0.3がベスト、0.2や0.5はpublicで悪化
# Vesuvius Competition Metric Evaluation

ローカル環境でVesuviusコンペの**公式評価指標**を計算するためのDocker環境。

公式の `topometrics` ライブラリ（Kaggleデータセット `sohier/vesuvius-metric-resources` に含まれる）をそのまま使用するため、**Kaggle上のスコアと完全に一致**します。

## 評価指標

コンペのスコアは以下の3つの指標の重み付き平均:

| 指標 | 重み | 説明 |
|------|------|------|
| Topological Score | 30% | Betti matching に基づくトポロジカルF1スコア |
| Surface Dice | 35% | 表面の一致度（tolerance=2.0ボクセル以内） |
| VOI Score | 35% | Variation of Information（1/(1 + α*VOI)） |

```
Leaderboard = 0.3 * TopoScore + 0.35 * SurfaceDice + 0.35 * VOI_score
```

---

## 初回セットアップ（1回だけ実行）

以下の手順は最初の1回だけ実行すればOKです。

### 1. メトリクスライブラリのダウンロード

```bash
cd eval
kaggle datasets download -d sohier/vesuvius-metric-resources -p ./data --unzip
```

### 2. TFRecords（検証データ）のダウンロード

```bash
kaggle datasets download -d ipythonx/vesuvius-tfrecords -p ./tfrecords --unzip
```

### 3. Dockerイメージのビルド

```bash
docker compose build
```

これにより以下が行われます:
- CUDA 12.4 + Python 3.11 環境の構築
- Betti-Matching-3D (C++拡張) のビルド
- topometrics パッケージのインストール
- JAX + Keras (nightly) + medicai のインストール

> 初回ビルドには時間がかかります（C++拡張のコンパイルのため）

---

## 評価の実行（毎回実行）

### 1. モデル重みのダウンロード

Kaggleで学習したモデルの重みをダウンロード:

```bash
cd /path/to/vesvius
kaggle kernels output toseihatori/train-vesuvius-surface-3d-detection-on-tpu -p .
# → model.weights.h5 がプロジェクトルートに保存される
```

### 2. 評価実行

```bash
cd eval
docker compose run --rm vesuvius-eval
```

カスタムオプションを指定する場合:

```bash
docker compose run --rm vesuvius-eval python evaluate.py \
    --weights /workspace/models/model.weights.h5 \
    --tfrecords /workspace/tfrecords \
    --output /workspace/output/results.json
```

---

## 出力例

```
============================================================
FINAL RESULTS (Official Competition Metric)
============================================================
  Samples evaluated: 6
  Mean Topo Score:     0.1452
  Mean Surface Dice:   0.7696
  Mean VOI Score:      0.5201
  Mean Combined Score: 0.4949 (+/- 0.0559)
============================================================
```

処理時間: 6サンプル × 320³ボクセル → 約7分（RTX 3090）

---

## ファイル構成

```
eval/
├── Dockerfile              # CUDA + topometrics ビルド環境
├── docker-compose.yml      # Docker Compose 設定
├── evaluate.py             # 評価スクリプト
├── README.md               # このファイル
├── data/                   # [初回DL] vesuvius-metric-resources
│   ├── topological-metrics-kaggle/  # 公式topometricsライブラリ
│   └── wheels/                      # 依存パッケージのwheels
├── tfrecords/              # [初回DL] 検証用TFRecords
└── output/                 # 結果出力先（results.json）
```

---

## コマンドラインオプション

```
options:
  --weights, -w       モデル重みファイル (.h5)
  --tfrecords, -t     TFRecordsディレクトリ
  --surface-tolerance Surface Diceのtolerance (default: 2.0)
  --overlap           Sliding window inferenceのoverlap (default: 0.5)
  --output, -o        結果出力ファイル (JSON)
```

---

## 注意事項

- validation set は学習ノートブック (v3) と同じく、**最後の TFRecord ファイル**（training_shard_130.tfrec）を使用
- 公式 `topometrics` ライブラリをそのまま使用 → **Kaggle上と完全に同じスコア**
- Keras nightly (3.12.0.dev2025100703) を使用（学習時と同じバージョン）

---

## トラブルシューティング

### GPUが認識されない

```bash
# NVIDIA Container Toolkit の確認
docker info | grep -i runtime

# インストール
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Betti-Matching-3D のビルドエラー

CMake や pybind11 関連のエラーが出る場合:

```bash
# コンテナ内で手動ビルド
docker compose run --rm vesuvius-eval bash
cd /workspace/topological-metrics-kaggle
./scripts/build_betti.sh
```

### モデル読み込みエラー

Kerasバージョンの不一致が原因の可能性があります。Dockerイメージを再ビルドしてください:

```bash
docker compose build --no-cache
```

# Vesuvius Training (PyTorch Multi-GPU)

ローカル環境でVesuviusモデルをPyTorchで学習するためのDocker環境。

`train-vesuvius-surface-3d-detection-on-tpu_v3.ipynb` をベースに、PyTorchのDistributedDataParallel (DDP) でマルチGPU学習に対応しています。

## モデル構成

| 項目 | 設定 |
|------|------|
| Model | TransUNet |
| Encoder | SEResNeXt50 |
| Input Shape | 160 × 160 × 160 |
| Num Classes | 3 |
| Loss | DiceCE + 0.2 × clDice (iters=10) |

---

## 初回セットアップ（1回だけ実行）

### 1. TFRecords（学習データ）の準備

評価用に既にダウンロード済みの場合はスキップ:

```bash
cd ../eval
kaggle datasets download -d ipythonx/vesuvius-tfrecords -p ./tfrecords --unzip
```

### 2. Dockerイメージのビルド

```bash
cd train
docker compose build
```

---

## 学習の実行

### YAML設定ファイルを使用（推奨）

```bash
# 2GPU学習
docker compose run --rm vesuvius-train \
    torchrun --nproc_per_node=2 train.py \
    --config /workspace/configs/exp001_baseline.yaml

# 1GPU学習
docker compose run --rm vesuvius-train-single \
    python train.py \
    --config /workspace/configs/exp001_baseline.yaml
```

### 出力先の変更

```bash
docker compose run --rm vesuvius-train \
    torchrun --nproc_per_node=2 train.py \
    --config /workspace/configs/exp001_baseline.yaml \
    --output /workspace/experiments
```

---

## 設定ファイル

実験設定は `configs/` ディレクトリのYAMLファイルで管理します。

### 設定例 (configs/exp001_baseline.yaml)

```yaml
experiment:
  name: exp001_baseline
  description: "Baseline TransUNet with SEResNeXt50 encoder"

model:
  architecture: transunet  # transunet or segformer
  encoder: seresnext50     # for transunet: seresnext50, for segformer: mit_b0

training:
  epochs: 200
  batch_size: 1            # per GPU
  learning_rate: 3e-4
  weight_decay: 1e-5
  warmup_ratio: 0.05

data:
  input_size: 160          # 160 or 128 (use 128 for less VRAM)
  val_fold: -1             # -1 means last TFRecord file
  num_classes: 3

loss:
  cldice_iters: 10
  cldice_weight: 0.2

evaluation:
  val_interval: 5          # validation every N epochs
  eval_interval: 100       # competition metric evaluation every N epochs
  overlap: 0.5             # sliding window overlap
  surface_tolerance: 2.0   # for surface dice
```

### 新しい実験の追加

```bash
cp configs/exp001_baseline.yaml configs/exp002_segformer.yaml
# 設定を編集
```

---

## 出力ディレクトリ構成

学習結果は `experiments/<experiment_name>/` に保存されます:

```
experiments/
└── exp001_baseline/
    ├── exp001_baseline.yaml    # 設定ファイルのコピー
    ├── exp001_baseline.log     # 学習ログ
    ├── model.weights.h5        # ベストモデル
    ├── history.json            # 学習履歴
    └── competition_metrics.json # 競技メトリクス履歴
```

---

## コマンドラインオプション

```
options:
  --config, -c         設定YAMLファイル (必須)
  --tfrecords, -t      TFRecordsディレクトリ (default: /workspace/tfrecords)
  --output, -o         出力ディレクトリ (default: /workspace/experiments)
```

---

## 競技メトリクス評価

学習中、以下のタイミングで公式競技メトリクス (topometrics) を評価します:

- `eval_interval` エポックごと（デフォルト: 100エポックごと）
- 学習終了時にベストモデルで最終評価

評価される指標:
- **Topo Score**: トポロジカルスコア
- **Surface Dice**: 表面ダイス係数
- **VOI Score**: Variation of Information スコア
- **Combined Score**: 加重平均スコア (0.3×Topo + 0.35×SurfDice + 0.35×VOI)

---

## ファイル構成

```
train/
├── Dockerfile              # PyTorch + CUDA 環境
├── docker-compose.yml      # Docker Compose 設定 (2GPU/1GPU)
├── train.py                # 学習スクリプト
├── README.md               # このファイル
├── configs/                # 実験設定ファイル
│   └── exp001_baseline.yaml
└── experiments/            # 出力先
    └── exp001_baseline/
        ├── model.weights.h5
        ├── exp001_baseline.log
        ├── history.json
        └── competition_metrics.json
```

---

## 注意事項

- **validation set**: 最後の TFRecord ファイル（training_shard_130.tfrec）を使用
- **Keras backend**: `torch` を使用
- **Keras version**: `3.12.0.dev2025100703`（Kaggle学習と同じ）
- **メモリ**: 160³ の3Dボリュームを扱うため、GPU 1枚あたり12GB以上推奨

---

## トラブルシューティング

### NCCL エラー（マルチGPU）

```bash
# 環境変数を設定
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
```

### メモリ不足

設定ファイルで input_size を小さくする:

```yaml
data:
  input_size: 128  # 160 → 128
```

または SegFormer を使用:

```yaml
model:
  architecture: segformer
  encoder: mit_b0
```

### GPUが認識されない

```bash
# NVIDIA Container Toolkit の確認
docker info | grep -i runtime

# インストール
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

# Kaggle 提出ガイド

## 概要

nnUNet モデルを Kaggle Code Competition に提出する手順。

## 必要なファイル

```
kaggle_kernel/
├── kernel-metadata.json    # Kernel 設定
└── submission.py           # 推論スクリプト

kaggle_packages/
├── dataset-metadata.json   # パッケージデータセット設定
├── nnunetv2-*.whl
├── acvl_utils-*.whl
├── batchgenerators-*.whl
├── batchgeneratorsv2-*.whl
├── connected_components_3d-*.whl
├── dynamic_network_architectures-*.whl
├── fft_conv_pytorch-*.whl
├── imagecodecs-*.whl
├── nibabel-*.whl
├── simpleitk-*.whl
└── tifffile-*.whl

kaggle_upload/
├── dataset-metadata.json   # モデルデータセット設定
└── nnUNet_results/         # モデルの重みファイル
```

## 手順

### 1. パッケージの準備

Docker コンテナ内で学習時と同じバージョンのパッケージを wheel 形式でダウンロード:

```bash
# バージョン確認
docker-compose exec nnunet pip freeze | grep -iE "nnunet|torch|acvl|connected|dynamic-network|batchgenerators"

# wheel ダウンロード（依存関係なし）
docker-compose exec nnunet pip download --no-deps \
  nnunetv2==2.6.4 \
  acvl_utils==0.2.5 \
  connected-components-3d==3.26.1 \
  batchgenerators==0.25.1 \
  batchgeneratorsv2==0.3.0 \
  nibabel==5.3.2 \
  tifffile==2025.10.16 \
  dynamic-network-architectures==0.4.3 \
  fft-conv-pytorch==1.2.0 \
  imagecodecs \
  SimpleITK \
  -d /workspace/kaggle_packages

# ソース配布(.tar.gz)を wheel に変換
docker-compose exec nnunet bash -c "
cd /workspace/kaggle_packages
pip wheel --no-deps nnunetv2-*.tar.gz -w .
pip wheel --no-deps acvl_utils-*.tar.gz -w .
pip wheel --no-deps batchgenerators-*.tar.gz -w .
pip wheel --no-deps batchgeneratorsv2-*.tar.gz -w .
pip wheel --no-deps dynamic_network_architectures-*.tar.gz -w .
rm *.tar.gz
"
```

### 2. Kaggle データセットのアップロード

#### パッケージデータセット

```bash
# kaggle_packages/dataset-metadata.json
{
  "title": "Vesvius nnUNet Packages",
  "id": "toseihatori/vesvius-nnunet-packages",
  "licenses": [{"name": "CC0-1.0"}]
}

# 新規作成
kaggle datasets create -p kaggle_packages --dir-mode zip

# 更新
kaggle datasets version -p kaggle_packages -m "説明" --dir-mode zip
```

#### モデルデータセット

```bash
# kaggle_upload にシンボリックリンク作成
mkdir -p kaggle_upload/nnUNet_results/Dataset100_VesuviusSurface/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_lowres/fold_0
cd kaggle_upload/nnUNet_results/Dataset100_VesuviusSurface/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_lowres
ln -sf /path/to/nnunet_output/.../plans.json .
ln -sf /path/to/nnunet_output/.../dataset.json .
ln -sf /path/to/nnunet_output/.../dataset_fingerprint.json .
cd fold_0
ln -sf /path/to/nnunet_output/.../fold_0/checkpoint_final.pth .

# kaggle_upload/dataset-metadata.json
{
  "title": "Vesvius nnUNet Model Weights",
  "id": "toseihatori/vesvius-model-weights",
  "licenses": [{"name": "CC0-1.0"}]
}

# アップロード
kaggle datasets version -p kaggle_upload -m "説明" --dir-mode zip
```

### 3. Kernel の設定

`kaggle_kernel/kernel-metadata.json`:

```json
{
  "id": "toseihatori/vesvius-nnunet-submission",
  "title": "Vesvius nnUNet Submission",
  "code_file": "submission.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": false,
  "dataset_sources": [
    "toseihatori/vesvius-model-weights",
    "toseihatori/vesvius-nnunet-packages"
  ],
  "competition_sources": [
    "vesuvius-challenge-surface-detection"
  ],
  "kernel_sources": []
}
```

### 4. Submission スクリプトのポイント

`kaggle_kernel/submission.py` で重要な点:

```python
# オフラインインストール（--no-deps で依存関係チェックをスキップ）
def install_packages():
    pkg_dir = "/kaggle/input/vesvius-nnunet-packages"
    packages = [
        "imagecodecs-*.whl",
        "fft_conv_pytorch-*.whl",
        "nnunetv2-*.whl",
        # ... 他のパッケージ
    ]
    for pkg in packages:
        cmd = f"pip install --no-deps -q {pkg_dir}/{pkg}"
        subprocess.run(cmd, shell=True, check=True)

# パス設定（Kaggle データセットの構造に注意）
# データセットアップロード時の構造によりパスが変わる
MODEL_DATASET_DIR = Path("/kaggle/input/vesvius-model-weights")
results_dir = MODEL_DATASET_DIR  # nnUNet_results は含まれない場合あり
```

### 5. Kernel の Push と Submit

```bash
# Kernel を push（実行される）
cd kaggle_kernel
kaggle kernels push

# ステータス確認
kaggle kernels status toseihatori/vesvius-nnunet-submission

# 出力取得
kaggle kernels output toseihatori/vesvius-nnunet-submission -p /tmp/output

# Competition に提出
kaggle competitions submit \
  -c vesuvius-challenge-surface-detection \
  -k toseihatori/vesvius-nnunet-submission \
  -f submission.zip \
  -v 9 \
  -m "説明"
```

### 6. 提出の監視

```bash
python monitor_submission.py \
  --competition vesuvius-challenge-surface-detection \
  --interval 300 \
  --log-dir logs
```

## トラブルシューティング

### パッケージが見つからない

- `--no-index -f /path` だと依存関係も含めて探すため、依存パッケージが不足するとエラー
- `--no-deps` で個別インストールすると Kaggle プリインストール済みパッケージを活用できる

### モデルパスが見つからない

- Kaggle データセットアップロード時のディレクトリ構造を確認
- `--dir-mode zip` の場合、ルートディレクトリが変わることがある
- Kernel ログで実際のファイルパスを確認

### TIFF 読み込みエラー (LZW compression)

- `imagecodecs` パッケージが必要
- バイナリ wheel (manylinux) を使用

### fft_conv_pytorch が見つからない

- `batchgeneratorsv2` の依存関係
- 明示的にインストールリストに追加

## 参考リンク

- [Kaggle API Wiki](https://github.com/Kaggle/kaggle-api/wiki)
- [nnUNet GitHub](https://github.com/MIC-DKFZ/nnUNet)

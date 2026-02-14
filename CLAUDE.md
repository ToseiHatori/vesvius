# Vesvius プロジェクト

Vesuvius巻物の3D表面検出プロジェクト。

## 注意事項

- `model.weights.h5` はバイナリファイルのため読み込まないこと
- ファイルの削除は権限の問題があるのでdocker経由で行うこと

## モデル学習の方法

Docker Composeを使用してnnUNet学習環境を構築・実行します。

### セットアップ

```bash
# イメージをビルド（初回のみ）
docker-compose build

# コンテナを起動
docker-compose up -d

# コンテナにログイン
docker-compose exec nnunet bash
```

### 前処理（初回のみ）

学習前にnnUNetの前処理を実行:

```bash
# 通常の前処理
python surface-nnunet-preprocessing.py

# ケース数を制限（デバッグ用）
python surface-nnunet-preprocessing.py --max-cases 5
```

前処理済みデータは `nnunet_output/nnUNet_preprocessed/` に保存されます。

### 学習の実行

コンテナ内で以下を実行:

```bash
# デバッグモード（動作確認用、1ケース・1エポック）
python surface-nnunet-training-local.py --debug --host-baseline --train

# Host Baseline（推奨設定）
python surface-nnunet-training-local.py --host-baseline --epochs 1000

# 通常の学習
python surface-nnunet-training-local.py --train --epochs 50

# 推論のみ
python surface-nnunet-training-local.py --inference
```

### コンテナの管理

```bash
# コンテナを停止
docker-compose down

# ログを確認
docker-compose logs -f

# コンテナの状態確認
docker-compose ps
```

### 不要リソースのクリーンアップ

```bash
# 停止中のコンテナを削除
docker container prune

# 未使用イメージを削除
docker image prune

# 全ての未使用リソースを削除
docker system prune
```


## Kaggle データセットへのアップロード手順

モデルの重みファイルをKaggleデータセットとしてアップロードする手順:

1. アップロード用ディレクトリを作成し、重みファイルをシンボリックリンクで配置する
2. `dataset-metadata.json` を作成する:
   ```json
   {
     "title": "Vesvius Model Weights",
     "id": "toseihatori/vesvius-model-weights",
     "licenses": [{"name": "CC0-1.0"}]
   }
   ```
3. Kaggle CLIでアップロードする:
   ```bash
   kaggle datasets create -p <ディレクトリパス> --dir-mode zip
   ```

アップロード先: https://www.kaggle.com/datasets/toseihatori/vesvius-model-weights

### バージョン更新

既存データセットを更新する場合:
```bash
kaggle datasets version -p <ディレクトリパス> -m "更新メッセージ" --dir-mode zip
```

## Kaggle 提出

詳細な手順は [docs/kaggle_submission_guide.md](docs/kaggle_submission_guide.md) を参照。

### クイックリファレンス

```bash
# Kernel を push
cd kaggle_kernel
kaggle kernels push

# Competition に提出
kaggle competitions submit \
  -c vesuvius-challenge-surface-detection \
  -k toseihatori/vesvius-nnunet-submission \
  -f submission.zip \
  -v <version番号> \
  -m "説明"

# 提出の監視（必ずnohupで実行すること！）
nohup python monitor_submission.py --competition vesuvius-challenge-surface-detection > logs/monitor_v<version>.log 2>&1 &
```

### 重要: 提出時の必須手順

**提出後は必ず `monitor_submission.py` を pipx の pythonで、 nohup で実行すること。**

これにより提出の経過時間が `logs/` に記録され、`docs/submission_history.md` の更新に必要な情報が残る。

## 実験管理

### ディレクトリ構成

```
docs/
├── EXPERIMENTS.md              # 実験結果サマリー（メイン）
├── results/                    # 評価結果CSV
│   ├── fold0_evaluation.csv
│   ├── fold1_evaluation.csv
│   └── ...
├── kaggle_submission_guide.md  # 提出ガイド
├── submission_history.md       # 提出履歴
└── postprocess_comparison.md   # 後処理比較分析
```

### 評価の実行

```bash
# Validation TIFの評価（postprocessなし）
docker-compose exec nnunet python /workspace/evaluate_validation_tif.py \
  --pred-dir /workspace/nnunet_output/nnUNet_results/<model>/fold_X/validation \
  --gt-dir /workspace/nnunet_output/nnUNet_preprocessed/Dataset100_VesuviusSurface/gt_segmentations \
  --workers 8 \
  --output-csv /workspace/docs/results/<name>.csv

# 後処理付き評価（NPZファイルが必要）
docker-compose exec nnunet python /workspace/evaluate_metrics.py \
  --npz-dir <npz_dir> \
  --gt-dir <gt_dir> \
  --postprocess all \
  --output-csv /workspace/docs/results/<name>.csv
```

### 結果の記録

1. CSVは `docs/results/` に保存
2. 主要な結果を `docs/EXPERIMENTS.md` に追記
3. Kaggle提出時は `docs/submission_history.md` を更新

### 評価指標

```
Leaderboard = 0.3 * TopoScore + 0.35 * SurfaceDice + 0.35 * VOI_score
```

- **TopoScore**: トポロジー保存（連結性）の評価
- **SurfaceDice**: 表面の一致度（tolerance=2.0）
- **VOI_score**: Variation of Information（セグメンテーション品質）

# Vesvius プロジェクト

Vesuvius巻物の3D表面検出プロジェクト。

## プロジェクト構成

- `train-vesuvius-surface-3d-detection-on-tpu.ipynb` - TPU上での学習ノートブック
- `train-vesuvius-surface-3d-detection-on-tpu.log` - 学習ログ
- `inference-vesuvius-surface-3d-detection.ipynb` - 推論ノートブック
- `vesuvius-2025-metric-demo.ipynb` - メトリクスデモノートブック
- `model.weights.h5` - 学習済みモデルの重み (803MB、バイナリ)

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

## 注意事項

- `model.weights.h5` はバイナリファイルのため読み込まないこと
- ファイルの削除は権限の問題があるのでdocker経由で行ってください
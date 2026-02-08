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

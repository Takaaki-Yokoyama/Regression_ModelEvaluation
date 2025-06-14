# 回帰モデル評価・可視化プロジェクト

## セットアップ

1. 必要なパッケージをインストール
```
pip install -r requirements.txt
```

2. 一括実行（前処理・学習・評価・可視化）
```
python main.py
```

---

個別に実行したい場合は以下の手順も利用できます：

- モデルの学習
```
python src/train.py
```
- モデルの評価（クロスバリデーションを実施し、評価指標をresults.jsonに保存）
```
python src/evaluate.py
```
- 結果の可視化
```
python src/visualize.py
```

- 各モデルの評価指標（r2, rmse, mae）が棒グラフで表示されます。
- 各モデルごとに予測値vs実測値の散布図が表示されます。

## ディレクトリ構成例
- data/data.csv : 入力データ
- config.yaml : 設定ファイル
- models/ : 学習済みモデル
- results.json : 評価結果
- src/ : スクリプト群
- main.py : 一括実行用スクリプト

## 補足
- config.yamlで使用モデルや評価指標を変更できます。
- `cv_folds` を変更することでクロスバリデーションの分割数を指定できます。
- データやモデルの形式に応じて`src/preprocess.py`を調整してください。

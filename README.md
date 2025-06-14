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
 - モデルの評価（クロスバリデーションを実施し、評価指標をresults.csvに保存。最良モデルはmodels/best_model.pklとして保存されます）
```
python src/evaluate.py
```
- 結果の可視化
```
python src/visualize.py
```

- 各モデルの評価指標（r2, rmse, mae）が棒グラフで表示されます。
- 各モデルごとに予測値vs実測値の散布図が表示されます。
- 最良モデル（評価指標が最も良いもの）は`models/best_model.pkl`として保存され、他プロジェクトでも再利用可能です。

## テスト

`pytest`で一連のパイプライン（学習・評価・保存）が正常に動作するか確認できます。

```
pytest
```

## ディレクトリ構成例
- data/data.csv : 入力データ
- config.yaml : 設定ファイル
- models/ : 学習済みモデル（best_model.pkl含む）
 - results.csv : 評価結果
- src/ : スクリプト群
- main.py : 一括実行用スクリプト
- tests/ : テストコード

## 補足
- config.yamlで使用モデルや評価指標を変更できます。
- `cv_folds` を変更することでクロスバリデーションの分割数を指定できます。
- データやモデルの形式に応じて`src/preprocess.py`を調整してください。
- **best_model.pklの選択基準**: `metrics`に複数の指標を指定した場合、config.yamlで最初に記載した指標（例: `metrics: [r2, rmse]`ならr2）が主指標となり、その値が最も良いモデルがbest modelとして保存されます。
- **マルチターゲット回帰対応**: `config.yaml` の `target_count` を2以上に設定すると、複数の目的変数に対する学習・評価が行われます。可視化では各ターゲットごとに `results/{モデル名}_target{インデックス}_pred_vs_true.png` のファイル名で散布図を保存します。

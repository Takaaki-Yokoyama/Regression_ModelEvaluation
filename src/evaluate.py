import os
import sys
import pickle
import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    make_scorer,
)
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# パスを追加してpreprocessをimport
sys.path.append(os.path.dirname(__file__))
from preprocess import load_data, load_config, preprocess_data
from train import MODEL_DICT

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'data', 'data.csv')
    config_path = os.path.join(base_dir, 'config.yaml')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    config = load_config(config_path)
    target_count = config.get('target_count', 1)
    model_names = config.get('regression_models', ['LinearRegression'])
    metrics = config.get('metrics', ['r2'])
    cv_folds = config.get('cv_folds', 5)

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df, target_count=target_count)
    # クロスバリデーション用に全データを結合
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    # 評価指標のscorer作成
    scorers = {
        'r2': make_scorer(r2_score),
        'rmse': make_scorer(
            lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), greater_is_better=False
        ),
        'mae': make_scorer(mean_absolute_error, greater_is_better=False),
    }

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    results = []
    for name in model_names:
        model_cls = MODEL_DICT.get(name)
        if model_cls is None:
            print(f"未対応モデル: {name}")
            continue
        model = model_cls()
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model),
        ])

        use_metrics = [m for m in metrics if m in scorers]
        cv_result = cross_validate(pipeline, X, y, cv=cv, scoring={m: scorers[m] for m in use_metrics})
        res = {'model': name}
        for metric in use_metrics:
            scores = cv_result[f'test_{metric}']
            if metric in ['rmse', 'mae']:
                res[metric] = (-scores).mean()
            else:
                res[metric] = scores.mean()
        results.append(res)
    # 結果表示
    for r in results:
        print(r)
    # 結果をJSONで保存
    import json
    results_path = os.path.join(base_dir, 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

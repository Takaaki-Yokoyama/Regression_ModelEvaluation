import os
import sys
import pickle
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
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

    # train.pyを呼び出してモデルを学習・保存
    try:
        from train import main as train_main
        train_main()
    except Exception as e:
        print(f"train実行エラー: {e}")

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

    # サポートする評価指標をフィルタリングし、主指標を決定
    scorers = {
        'r2': make_scorer(r2_score),
        'rmse': make_scorer(
            lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), greater_is_better=False
        ),
        'mae': make_scorer(mean_absolute_error, greater_is_better=False),
    }
    supported_metrics = [m for m in metrics if m in scorers]
    primary_metric = supported_metrics[0] if supported_metrics else None

    # クロスバリデーション用のscorersのみを使用
    cv_scorers = {m: scorers[m] for m in supported_metrics}

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    results = []
    if target_count > 1:
        for idx in range(target_count):
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
                # 各ターゲットごとにクロスバリデーション
                cv_result = cross_validate(pipeline, X, y[:, idx], cv=cv, scoring=cv_scorers)
                res = {'model': name, 'target': idx}
                for metric in supported_metrics:
                    scores = cv_result[f'test_{metric}']
                    if metric in ['rmse', 'mae']:
                        res[metric] = (-scores).mean()
                    else:
                        res[metric] = scores.mean()
                results.append(res)
    else:
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
            cv_result = cross_validate(pipeline, X, y, cv=cv, scoring=cv_scorers)
            res = {'model': name}
            for metric in supported_metrics:
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

    # --- ベストモデルのフルデータ再学習・保存（ターゲットごと） ---
    if primary_metric:
        if target_count > 1:
            for idx in range(target_count):
                # 各ターゲットごとに最良モデルを選択
                target_results = [r for r in results if r.get('target') == idx]
                if not target_results:
                    continue
                best = (min(target_results, key=lambda x: x[primary_metric])
                        if primary_metric in ['rmse', 'mae']
                        else max(target_results, key=lambda x: x[primary_metric]))
                best_name = best['model']
                model_cls = MODEL_DICT.get(best_name)
                if model_cls:
                    model = model_cls()
                    model.fit(X, y[:, idx])
                    best_model_dst = os.path.join(models_dir, f'best_model_target{idx}.pkl')
                    with open(best_model_dst, 'wb') as f:
                        pickle.dump(model, f)
                    print(f'ベストモデル: {best_name} (target{idx}) を保存しました: {best_model_dst}')
        else:
            # 1次元ターゲット
            best = (min(results, key=lambda x: x[primary_metric])
                    if primary_metric in ['rmse', 'mae']
                    else max(results, key=lambda x: x[primary_metric]))
            best_name = best['model']
            model_cls = MODEL_DICT.get(best_name)
            if model_cls:
                model = model_cls()
                model.fit(X, y)
                best_model_dst = os.path.join(models_dir, 'best_model.pkl')
                with open(best_model_dst, 'wb') as f:
                    pickle.dump(model, f)
                print(f'ベストモデル: {best_name} を保存しました: {best_model_dst}')

if __name__ == "__main__":
    main()

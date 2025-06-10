import os
import sys
import pickle
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# パスを追加してpreprocessをimport
sys.path.append(os.path.dirname(__file__))
from preprocess import load_data, load_config, preprocess_data

METRIC_FUNCS = {
    'r2': r2_score,
    'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
    'mae': mean_absolute_error
}

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

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df, target_count=target_count)

    results = []
    for name in model_names:
        model_path = os.path.join(models_dir, f'{name}.pkl')
        if not os.path.exists(model_path):
            print(f"モデルファイルがありません: {model_path}")
            continue
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        y_pred = model.predict(X_test)
        res = {'model': name}
        for metric in metrics:
            func = METRIC_FUNCS.get(metric)
            if func:
                score = func(y_test, y_pred)
                res[metric] = score
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

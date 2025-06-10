# モデル学習用

import os
import pickle
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from src.preprocess import load_data, load_config, preprocess_data

MODEL_DICT = {
    'LinearRegression': LinearRegression,
    'ElasticNet': ElasticNet,
    'DecisionTreeRegressor': DecisionTreeRegressor,
    'RandomForestRegressor': RandomForestRegressor,
    'GradientBoostingRegressor': GradientBoostingRegressor,
    'SVR': SVR
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

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df, target_count=target_count)

    for name in model_names:
        model_cls = MODEL_DICT.get(name)
        if model_cls is None:
            print(f"未対応モデル: {name}")
            continue
        model = model_cls()
        model.fit(X_train, y_train)
        model_path = os.path.join(models_dir, f'{name}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"{name} を保存しました: {model_path}")

if __name__ == "__main__":
    main()

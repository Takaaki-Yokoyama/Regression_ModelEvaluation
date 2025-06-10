import os
import sys
import json
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 日本語フォントを明示的に指定
matplotlib.rcParams['font.family'] = 'MS Gothic'  # Windowsの場合。LinuxならIPAexGothicなどに変更
matplotlib.rcParams['axes.unicode_minus'] = False

# パスを追加してpreprocessをimport
sys.path.append(os.path.dirname(__file__))
from preprocess import load_data, load_config, preprocess_data

# 評価指標の棒グラフ
def plot_metrics(results_path):
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    df = pd.DataFrame(results)
    metrics = [col for col in df.columns if col != 'model']
    df.set_index('model', inplace=True)
    df[metrics].plot(kind='bar', figsize=(10, 6))
    plt.title('各モデルの評価指標')
    plt.ylabel('スコア')
    plt.xlabel('モデル')
    plt.tight_layout()
    plt.show()

# 予測値vs実測値の散布図
import pickle

def plot_pred_vs_true():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'data', 'data.csv')
    config_path = os.path.join(base_dir, 'config.yaml')
    models_dir = os.path.join(base_dir, 'models')
    config = load_config(config_path)
    target_count = config.get('target_count', 1)
    model_names = config.get('regression_models', ['LinearRegression'])
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df, target_count=target_count)
    for name in model_names:
        model_path = os.path.join(models_dir, f'{name}.pkl')
        if not os.path.exists(model_path):
            continue
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        y_pred = model.predict(X_test)
        plt.figure(figsize=(6,6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.xlabel('実測値')
        plt.ylabel('予測値')
        plt.title(f'{name}: 予測値 vs 実測値')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    results_path = os.path.join(base_dir, 'results.json')
    plot_metrics(results_path)
    plot_pred_vs_true()

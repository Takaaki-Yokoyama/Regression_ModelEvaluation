import os
import sys
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 日本語フォントを明示的に指定
matplotlib.rcParams['font.family'] = 'MS Gothic'  # Windowsの場合。LinuxならIPAexGothicなどに変更
matplotlib.rcParams['axes.unicode_minus'] = False

# パスを追加してpreprocessをimport
sys.path.append(os.path.dirname(__file__))
from preprocess import load_data, load_config, preprocess_data

# 評価指標の棒グラフ

def plot_metrics(results_path):
    import os
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    df = pd.DataFrame(results)
    # 複数ターゲット対応: model, targetごとに指標をプロット
    if 'target' in df.columns:
        metrics = [col for col in df.columns if col not in ['model', 'target']]
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            sns.barplot(
                data=df,
                x='model',
                y=metric,
                hue='target',
                palette='Set2'
            )
            plt.title(f'各モデル・ターゲットの{metric}指標')
            plt.ylabel(metric)
            plt.xlabel('モデル')
            plt.tight_layout()
            results_dir = os.path.join(os.path.dirname(results_path), 'results')
            os.makedirs(results_dir, exist_ok=True)
            out_path = os.path.join(results_dir, f'metrics_bar_{metric}.png')
            plt.savefig(out_path)
            print(f'評価指標グラフ({metric})を保存しました: {out_path}')
            plt.close()
    else:
        # 旧来の単一ターゲット用
        metrics = [col for col in df.columns if col != 'model']
        df.set_index('model', inplace=True)
        df[metrics].plot(kind='bar', figsize=(10, 6))
        plt.title('各モデルの評価指標')
        plt.ylabel('スコア')
        plt.xlabel('モデル')
        plt.tight_layout()
        results_dir = os.path.join(os.path.dirname(results_path), 'results')
        os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, 'metrics_bar.png')
        plt.savefig(out_path)
        print(f'評価指標グラフを保存しました: {out_path}')
        plt.close()

# 予測値vs実測値の散布図
import pickle

def plot_pred_vs_true():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'data', 'data.csv')
    config_path = os.path.join(base_dir, 'config.yaml')
    models_dir = os.path.join(base_dir, 'models')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    config = load_config(config_path)
    target_count = config.get('target_count', 1)
    model_names = config.get('regression_models', ['LinearRegression'])
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df, target_count=target_count)
    # 多次元出力対応: y_test, y_predを2D配列に変換
    y_true = np.array(y_test)
    for name in model_names:
        model_path = os.path.join(models_dir, f'{name}.pkl')
        if not os.path.exists(model_path):
            continue
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        y_pred = model.predict(X_test)
        y_pred = np.array(y_pred)
        # 1次元配列を2次元配列に
        if y_true.ndim == 1:
            y_true2 = y_true.reshape(-1, 1)
        else:
            y_true2 = y_true
        if y_pred.ndim == 1:
            y_pred2 = y_pred.reshape(-1, 1)
        else:
            y_pred2 = y_pred
        n_targets = y_true2.shape[1]
        for idx in range(n_targets):
            true_vals = y_true2[:, idx]
            pred_vals = y_pred2[:, idx]
            plt.figure(figsize=(6,6))
            plt.scatter(true_vals, pred_vals, alpha=0.5)
            plt.xlabel('実測値')
            plt.ylabel('予測値')
            title = f'{name} target_{idx}: 予測値 vs 実測値'
            plt.title(title)
            plt.plot([true_vals.min(), true_vals.max()], [true_vals.min(), true_vals.max()], 'r--')
            plt.tight_layout()
            out_path = os.path.join(results_dir, f'{name}_target{idx}_pred_vs_true.png')
            plt.savefig(out_path)
            print(f'{title} グラフを保存しました: {out_path}')
            plt.close()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    results_path = os.path.join(base_dir, 'results.json')
    plot_metrics(results_path)
    plot_pred_vs_true()

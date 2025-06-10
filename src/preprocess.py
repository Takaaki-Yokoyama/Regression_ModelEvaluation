import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import yaml

# データ読み込み関数
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# 前処理関数
def preprocess_data(df, target_count=1, standardize=True, test_size=0.2, random_state=42):
    # 欠損値除去
    df = df.dropna()
    # 目的変数の分割（最後のtarget_count列）
    if target_count == 1:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    else:
        X = df.iloc[:, :-target_count]
        y = df.iloc[:, -target_count:]
    # 標準化
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    # 訓練・テスト分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # サンプル実行
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, "data", "data.csv")
    config_path = os.path.join(base_dir, "config.yaml")
    config = load_config(config_path)
    target_count = config.get('target_count', 1)
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = preprocess_data(df, target_count=target_count)
    print(f"train shape: {X_train.shape}, test shape: {X_test.shape}")

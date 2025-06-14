import os
import subprocess
import sys
import pickle
import yaml

def test_main_pipeline_runs_and_saves_models(tmp_path, monkeypatch):
    # テスト用にカレントディレクトリをプロジェクトルートに設定
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    monkeypatch.chdir(project_root)

    # 既存のmodelsディレクトリがあれば削除してクリーンアップ
    models_dir = os.path.join(project_root, 'models')
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            os.remove(os.path.join(models_dir, f))
    else:
        os.makedirs(models_dir)

    # main.pyを実行
    result = subprocess.run([sys.executable, 'main.py'], capture_output=True, text=True)
    assert result.returncode == 0, f"main.py 実行エラー: {result.stderr}"

    # trainによる各モデルの保存確認
    expected_models = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    assert expected_models, "モデルファイルが一つも保存されていません"

    # config.yamlからtarget_count取得
    config_path = os.path.join(project_root, 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    target_count = config.get('target_count', 1)

    if target_count > 1:
        # 各ターゲットごとにbest_model_target{idx}.pklの存在とロード・predict確認
        for idx in range(target_count):
            best_model_path = os.path.join(models_dir, f'best_model_target{idx}.pkl')
            assert os.path.exists(best_model_path), f"best_model_target{idx}.pkl が存在しません"
            with open(best_model_path, 'rb') as f:
                model = pickle.load(f)
            assert hasattr(model, 'predict'), f"best_model_target{idx}.pkl にpredict属性がありません"
    else:
        # best_model.pklの存在確認
        best_model_path = os.path.join(models_dir, 'best_model.pkl')
        assert os.path.exists(best_model_path), "best_model.pkl が存在しません"
        with open(best_model_path, 'rb') as f:
            model = pickle.load(f)
        assert hasattr(model, 'predict'), "best_model.pkl にpredict属性がありません"

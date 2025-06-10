import os
import sys
import subprocess

# プロジェクトのルートディレクトリを取得
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')

# スクリプト実行用関数
def run_script(script_name):
    script_path = os.path.join(SRC_DIR, script_name)
    print(f'==== {script_name} 実行開始 ===')
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    print(f'==== {script_name} 実行終了 ===\n')

if __name__ == '__main__':
    # train.py → evaluate.py → visualize.py の順に実行
    run_script('train.py')
    run_script('evaluate.py')
    run_script('visualize.py')

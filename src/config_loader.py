import json
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_FILE_PATH = os.path.join(PROJECT_ROOT, 'config', 'config.json')


def load_config(config_path=CONFIG_FILE_PATH):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}.")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


_config = load_config()

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, _config['data']['raw_path'])
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, _config['data']['processed_path'])
FINAL_DATA_DIR = os.path.join(PROJECT_ROOT, _config['data']['final_dir'])

MODEL_DIR = os.path.join(PROJECT_ROOT, _config['models']['model_dir'])
MODEL_PATH = os.path.join(PROJECT_ROOT, _config['models']['model_path'])
SCALER_PATH = os.path.join(PROJECT_ROOT, _config['models']['scaler_path'])

FIGURE_DIR = os.path.join(PROJECT_ROOT, _config['reports']['figure_dir'])
PLOT_PATH = os.path.join(PROJECT_ROOT, _config['reports']['plot_path'])


TEST_YEAR = _config['params']['test_year']
WINDOW_SIZE = _config['params']['window_size']
EPOCHS = _config['params']['epochs']
BATCH_SIZE = _config['params']['batch_size']

if __name__ == '__main__':
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Raw Data Path: {RAW_DATA_PATH}")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Window Size: {WINDOW_SIZE}")
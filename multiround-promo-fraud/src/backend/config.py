import os

class Config:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    SRC_DIR = os.path.join(BASE_DIR, 'src')
    DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
    RESULT_DIR = os.path.join(BASE_DIR, 'result')
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoint')
    SCRIPT_DIR = os.path.join(BASE_DIR, 'scripts')

    MAX_CONCURRENT_EXPERIMENTS = 1
    MAX_HISTORY_PER_EXPERIMENT = 50

    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = False

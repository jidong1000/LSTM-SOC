from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATASETS_DIR = ROOT_DIR / 'datasets'
SRC_DIR = ROOT_DIR / 'src'
MODELS_DIR = ROOT_DIR / 'models'

HIDDEN_SIZE = 64
NUM_LAYERS = 2
WIN_SIZE = 300
import sys
from pathlib import Path


V4_DIR = Path(__file__).resolve().parents[1]
for path in (V4_DIR / 'model', V4_DIR / 'scripts'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

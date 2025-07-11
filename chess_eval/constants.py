# %%
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.resolve()
SCALING_PATH = BASE_DIR / "models/scaling.json"
MODEL_PATH = BASE_DIR / "models/chess_model.pt"

if sys.platform == "win32":
    SF_PATH = BASE_DIR / "stockfish_/stockfish-windows-x86-64-avx2.exe"
else:
    raise OSError("No specified path for stockfish for this system")

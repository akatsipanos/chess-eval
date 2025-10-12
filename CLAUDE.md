# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a chess position evaluation project that uses machine learning to predict game outcomes (White win, Draw, Black win) based on board positions, player ratings, time remaining, and Stockfish evaluations. The project consists of:

1. **Python backend**: PyTorch neural network model + FastAPI server
2. **React frontend**: Interactive chessboard UI for real-time position evaluation
3. **Data processing pipeline**: Converts PGN game files into training data with Stockfish evaluations

## Development Setup

### Python Environment

This project uses `uv` for Python dependency management. The project requires Python >= 3.11.

**Install dependencies:**
```bash
uv sync
```

**Activate virtual environment:**
```bash
source .venv/bin/activate  # macOS/Linux
```

### Frontend Setup

The React frontend is located in `my-app/`.

```bash
cd my-app
npm install
npm run dev  # Starts Vite dev server on port 5173
```

## Common Commands

### Running the Application

**Start FastAPI backend:**
```bash
uvicorn chess_eval.app_fastapi:app --reload --host localhost --port 8000
```

The backend serves two endpoints:
- `/predict` (form data) - for template rendering
- `/api/predict` (JSON) - for the React frontend

**Start React frontend:**
```bash
cd my-app
npm run dev
```

### Data Processing

Process PGN chess game files into training data:

```bash
python chess_eval/data_processing.py --config windows-train
```

Configuration is specified in `data_processing_config.yml`:
- `sf_depth`: Stockfish evaluation depth
- `raw_data_file_name`: Input PGN file
- `data_type`: train/val/test

This generates `.npy` files in `data/processed/` with 71 features per position:
- 64 board squares (piece values)
- White/black remaining time ratios
- Stockfish evaluation
- Turn indicator
- White/black ratings
- Result label

### Training

Train the model using configurations from `ml_config.yml`:

```bash
python chess_eval/train.py --config test --save-model --log-interval 1000
```

Arguments:
- `--config`: Config name from `ml_config.yml` (e.g., "test", "old_run")
- `--save-model`: Save trained model to `models/`
- `--log-interval`: Batch logging frequency

Training configurations include: epochs, learning rate, gamma, scheduler type, batch size, step size.

### Hyperparameter Tuning

Run Optuna-based hyperparameter search with MLflow tracking:

```bash
python chess_eval/hp_tuning_lr.py
```

This performs 50 trials optimizing validation accuracy, logging to `./mlruns`.

### Testing

```bash
pytest
```

Run a single test:
```bash
pytest tests/unit/test_a.py::test_function_name
```

### Code Quality

**Type checking:**
```bash
uv run mypy --explicit-package-bases chess_eval/
uv run pyrefly check chess_eval/
```

Note: mypy is slow; pyrefly is being evaluated as a faster alternative.

**Linting and formatting:**
```bash
uv run ruff check
uv run ruff format
uv run isort --profile black chess_eval/
```

**Pre-commit hooks:**
```bash
pre-commit run --all-files
```

Pre-commit runs: ruff, mypy, pyrefly, isort, pyupgrade, bandit, detect-secrets, nbstripout

## Architecture

### Neural Network Models

The main model is `Network` (2 hidden layer architecture) in `chess_eval/networks.py`:
- Input: 70 features (board + metadata, excluding result)
- Hidden layers: 32 → 16 nodes with ReLU, BatchNorm, Dropout(0.5)
- Output: 3-class softmax (White win, Draw, Black win)

Alternative architectures available: `Network_1h`, `Network_3h`, `Conv` (experimental CNN).

### Data Flow

1. **Input** → `InputData` schema (FEN, ratings, time, turn)
2. **Preprocessing** → `create_input()` converts FEN to matrix + normalizes ratings
3. **Stockfish evaluation** → Calls local Stockfish binary for position evaluation
4. **Model inference** → PyTorch model outputs probability distribution
5. **Output** → Predictions returned as JSON or HTML template

### Key Modules

- `app_fastapi.py`: FastAPI server with CORS for local frontend, model caching
- `train.py`: Training loop with validation, learning rate schedulers, plotting
- `data_processing.py`: PGN → processed data pipeline with DataProcessing class
- `networks.py`: PyTorch model definitions
- `utils.py`: FEN parsing, Stockfish integration, data normalization
- `schemas.py`: Pydantic models and TypedDicts for data validation
- `constants.py`: Platform-specific paths (Stockfish binary, model, scaling)

### Stockfish Integration

Platform-specific Stockfish binaries are in `stockfish_/`:
- Windows: `stockfish-windows-x86-64-avx2.exe`
- macOS: `stockfish-macos-m1-apple-silicon`
- Linux: (path not configured)

The `constants.py` file automatically selects the correct binary based on `sys.platform`.

### Rating Normalization

Player ratings are min-max normalized using `models/scaling.json`, which is generated during training data preparation and stores the min/max ratings from the training set.

### Frontend Architecture

React app (`my-app/src/App.jsx`) features:
- Interactive chessboard using `react-chessboard` and `chess.js`
- Real-time position evaluation on every move
- FEN input/output
- Visual evaluation bar (Stockfish eval)
- Player rating and time inputs
- Drag-and-drop + click-to-move piece interaction

## Configuration Files

- `pyproject.toml`: Python dependencies, mypy/bandit config
- `ml_config.yml`: Training hyperparameters (epochs, lr, gamma, scheduler, batch_size)
- `data_processing_config.yml`: Data pipeline settings (sf_depth, file paths)
- `.pre-commit-config.yaml`: Git hooks configuration
- `my-app/package.json`: Frontend dependencies and scripts

## Notes

- The model expects input of shape (batch_size, 70) where the last 6 features are: white_time_ratio, black_time_ratio, sf_eval, turn, white_rating_normalized, black_rating_normalized
- Training uses both CrossEntropyLoss (for backprop) and CrossEntropyLoss with reduction='sum' (for metrics)
- Model loading uses `torch.load` with `# nosec: CWE-502` comment for Bandit security scanner
- The `tbc/` directory contains experimental code not currently in use

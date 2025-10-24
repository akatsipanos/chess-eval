# Scripts Directory

This directory contains utility scripts for the chess-eval project.

## mypy_with_stub_sync.py

A wrapper script for mypy that automatically adds missing type stub packages to `pyproject.toml`.

### Problem Statement

When mypy runs and discovers it needs type stubs (e.g., `types-requests`, `pandas-stubs`), it automatically installs them into the virtual environment. However, these packages are not reflected in `pyproject.toml`, which can lead to:

- Inconsistencies between development environments
- Missing dependencies when setting up the project fresh
- Manual tracking of stub packages

### Solution

This script wraps mypy execution and:

1. Runs mypy with all the same arguments
2. Monitors mypy's output for stub package installations
3. Automatically adds discovered stub packages to `pyproject.toml` using `uv add --dev`
4. Maintains proper dependency tracking without slowing down pre-commit

### How It Works

1. The script runs mypy and streams its output in real-time
2. It parses the output for patterns indicating stub package installations
3. After mypy completes, it extracts the package names
4. It runs `uv add --dev <package>` for each discovered stub package
5. The script returns mypy's exit code, so pre-commit workflows work correctly

### Integration

The script is integrated into the pre-commit workflow via `.pre-commit-config.yaml`:

```yaml
- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v1.11.2
  hooks:
    - id: mypy
      entry: python scripts/mypy_with_stub_sync.py
      args:
        [
          --explicit-package-bases
        ]
```

### Benefits

- **No Performance Penalty**: Mypy runs only once (no dry runs)
- **Automatic Dependency Management**: Stub packages are added to `pyproject.toml` automatically
- **Consistent Environments**: All developers get the same type stubs
- **Works with uv/poetry**: Uses `uv add` to properly manage dependencies
- **Dev Group**: Stubs are correctly added to the dev dependency group

### Usage

The script is automatically invoked by pre-commit. If you need to run it manually:

```bash
python scripts/mypy_with_stub_sync.py [mypy-args]
```

For example:

```bash
python scripts/mypy_with_stub_sync.py --explicit-package-bases chess_eval/
```

### Note

When the script adds packages to `pyproject.toml`, you'll see a message indicating which packages were added. You may need to stage the updated `pyproject.toml` and `uv.lock` files:

```bash
git add pyproject.toml uv.lock
```

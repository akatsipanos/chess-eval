#!/usr/bin/env python3
"""
Mypy wrapper that automatically adds missing type stubs to pyproject.toml.

This script runs mypy and monitors its output for stub package installations.
When mypy installs stub packages, they are automatically added to the
pyproject.toml dev dependency group using `uv add --dev`.
"""

import re
import subprocess
import sys
from pathlib import Path
from typing import List, Set


def parse_stub_packages_from_output(output: str) -> Set[str]:
    """
    Parse mypy output to extract stub package names.

    Mypy outputs messages like:
    - "Installing missing stub packages:"
    - "/path/to/python -m pip install types-foo"
    - Or suggests: "python3 -m pip install types-foo"
    """
    stub_packages: Set[str] = set()

    # Pattern to match pip install commands for stub packages
    # Matches: pip install types-foo, pip install pandas-stubs, etc.
    pip_install_pattern = re.compile(r'pip install\s+((?:types-|.*-stubs)\S+)', re.MULTILINE)

    # Find all stub packages mentioned in pip install commands
    for match in pip_install_pattern.finditer(output):
        package = match.group(1)
        # Remove version specifiers if any (e.g., types-foo>=1.0.0 -> types-foo)
        package = re.split(r'[>=<\[]', package)[0]
        stub_packages.add(package)

    return stub_packages


def add_stubs_to_pyproject(stub_packages: Set[str], project_root: Path) -> None:
    """
    Add stub packages to pyproject.toml dev dependency group using uv.

    Args:
        stub_packages: Set of stub package names to add
        project_root: Path to the project root directory
    """
    if not stub_packages:
        return

    print("\n" + "=" * 70)
    print("Mypy installed new stub packages. Adding them to pyproject.toml...")
    print("=" * 70)

    for package in sorted(stub_packages):
        print(f"Adding {package} to dev dependencies...")
        try:
            result = subprocess.run(
                ["uv", "add", "--dev", package],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                print(f"  ✓ {package} added successfully")
            else:
                print(f"  ✗ Failed to add {package}: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print(f"  ✗ Timeout while adding {package}")
        except Exception as e:
            print(f"  ✗ Error adding {package}: {e}")

    print("=" * 70)
    print("Note: pyproject.toml has been updated. You may need to stage the changes.")
    print("=" * 70 + "\n")


def run_mypy_with_monitoring(args: List[str], project_root: Path) -> int:
    """
    Run mypy with the given arguments and monitor for stub installations.

    Args:
        args: Command line arguments to pass to mypy
        project_root: Path to the project root directory

    Returns:
        Exit code from mypy
    """
    # Run mypy via uv
    cmd = ["uv", "run", "mypy"] + args

    try:
        # Run mypy and capture output in real-time
        process = subprocess.Popen(
            cmd,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Collect output while streaming to console
        full_output = []
        if process.stdout:
            for line in process.stdout:
                # Print line immediately so user sees progress
                print(line, end='')
                full_output.append(line)

        # Wait for process to complete
        return_code = process.wait()

        # Parse collected output for stub packages
        output_text = ''.join(full_output)
        stub_packages = parse_stub_packages_from_output(output_text)

        # Add any discovered stub packages to pyproject.toml
        if stub_packages:
            add_stubs_to_pyproject(stub_packages, project_root)

        return return_code

    except FileNotFoundError:
        print("Error: 'uv' command not found. Please ensure uv is installed.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error running mypy: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Main entry point."""
    # Get project root (parent of scripts directory)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    # Pass all command line arguments to mypy
    mypy_args = sys.argv[1:]

    return run_mypy_with_monitoring(mypy_args, project_root)


if __name__ == "__main__":
    sys.exit(main())

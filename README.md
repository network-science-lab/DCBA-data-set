# DCBA Data Set

This repository contains the DCBA data set along with the necessary code to generate and load
the data.

## Runtime Configuration

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/). `uv` manages the Python
   environment and dependency installation for this project.
2. From the repository root, sync the project environment and install dependencies:
   ```bash
   uv sync
   ```
3. Use `uv shell` to enter the project environment, or prefix commands with `uv run`.
4. Additionally, to use DVC with Google Drive as remote storage, install:
   ```bash
   uv tool install 'dvc[gdrive]'
   ```
5. Download the data required for experiments using DVC:
   ```bash
   dvc pull
   ```

## Development Notes

- Update the project version in `pyproject.toml`

## Doodles

- Add DVC remote
- Add code to load generated data

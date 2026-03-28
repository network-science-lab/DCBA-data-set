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
3. Resolve Julia dependencies (required on first setup):
   ```bash
   uv run python -c "import juliapkg; juliapkg.resolve(force=True)"
   ```
4. Additionally, to use DVC with Google Drive as remote storage, install:
   ```bash
   uv tool install 'dvc[gdrive]'
   ```
4. Download the data required for experiments using DVC:
   ```bash
   dvc pull
   ```

## Development Notes

- Update the project version in `pyproject.toml`

## Doodles

- Add DVC remote
- Add code to load generated data
- for mABCD the dataset generator should be modified to generate n-laytered networks (now it's only
   hardcoded to 3 amd in fact still doesn't work)

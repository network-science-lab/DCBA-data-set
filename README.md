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
5. Download the data required for experiments using DVC:
   ```bash
   dvc pull
   ```

## Usage

To run the code execute: `uv run dcba-data-set <path to the configuration file>`. Example configs
are provided in `scripts/configs/example_generate/`.

There are three functionalities provided by this repository:

### 1. ABCD Generator

Generates a single ABCD graph from a given configuration. Set `experiment_type: "generate-abcd"`
in the config. See `scripts/configs/example_generate/abcd.yaml` for an example.

### 2. mABCD Generator

Generates a single multilayer mABCD graph from a given configuration. Set
`experiment_type: "generate-mabcd"` in the config. See
`scripts/configs/example_generate/mabcd.yaml` for an example.

### 3. Dataset Generator

Samples multiple configurations from provided parameter ranges and generates a network for each.
Set `experiment_type: "generate-dataset"` in the config. See
`scripts/configs/example_generate/dataset.yaml` (ABCD) and
`scripts/configs/example_generate/dataset_mabcd.yaml` (mABCD) for examples.

## Development Notes

- Update the project version in `pyproject.toml`

## Doodles

- Add DVC remote
- Add code to load generated data
- for mABCD the dataset generator should be modified to generate n-laytered networks (now it's only
   hardcoded to 3 amd in fact still doesn't work)

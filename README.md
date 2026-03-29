# DCBA Data Set

This repository contains the DCBA data set along with the necessary code to generate and load the
data.

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
6. Install `pre-commit`:
   ```bash
   uv run pre-commit install --config .pre-commit-config.yaml
   ```

## Usage

To run the code execute: `uv run dcba-data-set <path to the configuration file>`. Example configs
are provided in `scripts/configs/example_generate/`.

There are three functionalities provided by this repository:

### 1. ABCD Generator

Generates a single ABCD graph from a given configuration. Set `experiment_type: "generate-abcd"` in
the config. See `scripts/configs/example_generate/abcd.yaml` for an example.

### 2. mABCD Generator

Generates a single multilayer mABCD graph from a given configuration. Set
`experiment_type: "generate-mabcd"` in the config. See `scripts/configs/example_generate/mabcd.yaml`
for an example.

### 3. Dataset Generator

Samples multiple configurations from provided parameter ranges and generates a network for each. Set
`experiment_type: "generate-dataset"` in the config. See
`scripts/configs/example_generate/dataset.yaml` (ABCD) and
`scripts/configs/example_generate/dataset_mabcd.yaml` (mABCD) for examples.

## Doodles

- Add DVC remote
- Add code to load generated data
- for mABCD the dataset generator should be modified to generate n-laytered networks (now it's only
  hardcoded to 3 amd in fact still doesn't work)
- Community IDs in `DCBAHeteroData` currently store raw values from the generation files. Two
  encoding alternatives worth exploring:
  - Option A (partial invariance): canonically reindex IDs per layer by descending community size,
    so ID 0 is always the largest community.
  - Option B (full invariance): replace IDs with per-node statistics (community_size, intra_degree)
    per layer — shape `[num_actors, 2, num_layers]`. Fully invariant by construction; directly
    mirrors ABCD parameters like xi/mu.

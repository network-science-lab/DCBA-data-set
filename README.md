# DCBA Data Set

This repository contains the DCBA data set along with the necessary code to generate and load the
data. It is part of the codebase for the paper: "Graph Data Augmentation via Contrastive Generator
Inversion (DCBA)" accepted to the 5th Learning on Graphs Conference (Boston, USA, 2026).

## Runtime configuration

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
   This tool will automatically check code formatting (it's a very convenient configuration) and run
   tests before each commit. To skip checks, use `git commit --no-verify`; to scan all files
   execute: `uv run pre-commit run --all-files --config .pre-commit-config.yaml`.

## Datasets

All datasets live under `data/` and are tracked by DVC. After `dvc pull` the following are
available:

| Dataset               | Instances | Graphs | Notes                               |
| --------------------- | --------- | ------ | ----------------------------------- |
| `abcd-big`            | 8 744     | 87 440 | big set; used in the paper          |
| `abcd-borderline`     | 486       | 4 860  | hard set; used in the paper         |
| `abcd-interim`        | 200       | 2 000  | small set for test trainings        |
| `abcd-distinct-comms` | 200       | 1 999  | easy set with separated communities |

Graphs within `abcd-big` are prefixed with their chunk name in the instance ID (e.g.
`chunk-7/3f1a...`). Instances that timed out mid-generation are excluded by default.

To download the dataset, one must authenticate with a Google account that has access to the shared
Google Drive: `https://drive.google.com/drive/u/1/folders/1YdaLLIRZNaptO6QzHpfS2IrfPoEFyvYq`. If you
need access, please contact the authors.

## Data loaders

```python
from dcba_data_set.graph_io import load_dataset, DCBAHeteroData

records = load_dataset("data/abcd-big")

graph: DCBAHeteroData = DCBAHeteroData.from_replica_record(
    records[0].replicas[0],
    instance_id=records[0].instance_id,
    net_type=records[0].net_type,
)
```

`load_dataset` handles both flat (`report.json` at root) and chunked layouts automatically. Pass
`discard_failed=False` to retain instances that timed out with partial replica sets.

## Generators

To run the code, execute: `uv run dcba-data-set <path to the configuration file>`. Example configs
are provided in `scripts/configs/example_generate/`.

This repository provides three functionalities:

### 1. ABCD Generator

Generates a single ABCD graph from a given configuration. Set `experiment_type: "generate-abcd"` in
the config. See `scripts/configs/example_generate/abcd.yaml` for an example.

### 2. mABCD Generator

Generates a single multilayer mABCD graph from a given configuration. Set
`experiment_type: "generate-mabcd"` in the config. See `scripts/configs/example_generate/mabcd.yaml`
for an example.

### 3. Dataset Generator

Samples multiple configurations from provided parameter ranges and generates a network for each. Set
`experiment_type: "generate-dataset"` in the config. See `scripts/configs/` for examples.

## Citing the code

If you use the data set or the code, please consider citing us:

```bibtex
@inproceedings{stolarski2026dcba,
   title={Graph Data Augmentation via Contrastive Generator Inversion (DCBA)},
   author={
      Stolarski, Mateusz and Czuba, Micha{\l} and Krai\'{n}ski, \L{}ukasz and Musial, Katarzyna and
      Pra\l{}at, Pawe\l{} and Kami\'{n}ski, Bogumi\l{} and Br{\'o}dka, Piotr
   },
}
```

## Acknowledgement

This research was partially supported by: (1) National Science Centre, Poland, grant no.
2022/45/B/ST6/04145; (2) Polish National Agency for Academic Exchange, Strategic Partnerships, grant
no. BPI/PST/2024/1/00129/U/00001; (3) Wrocław Tech, Academia Professorum Iuniorum. Views and
opinions expressed here are those of the authors only and do not necessarily reflect those of the
funding agencies.

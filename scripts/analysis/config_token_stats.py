r"""
Analyse token-length distribution of ABCD config strings as seen by the Qwen tokeniser.

For every dataset listed in ABCD_DATASETS the script:
  1. Loads all config.yaml files via the dataset loader.
  2. Filters the config dict to the keys appropriate for that network type.
  3. Serialises the filtered dict to compact JSON - the same format used during training.
  4. Tokenises with the Qwen3-4B-Base tokeniser (no model weights are loaded).
  5. Records the token count per config.
  6. Writes per-dataset statistics (min, max, mean) and a histogram PNG to data/analysis/.
  7. Writes a combined JSON summary for all datasets to data/analysis/config_token_stats.json.

Usage:
    uv run --group analysis scripts/analysis/config_token_stats.py
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

from dcba_data_set.graph_io.data_models import DCBAInstanceConfig
from dcba_data_set.graph_io.loader import load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT: Path = Path(__file__).parents[2]
_DATA_ROOT: Path = _REPO_ROOT / "data"
_OUTPUT_DIR: Path = _DATA_ROOT / "analysis"

#: ABCD parameter keys, in the same order used by DCBA's ABCDConfigSchema.
ABCD_CONFIG_KEYS: list[str] = ["n", "t1", "t2", "xi", "c_min", "c_max", "d_min", "d_max", "nout"]

#: Maps net_type (as recorded in report.json) to the ordered list of config keys
#: that should be included in the JSON string fed to the tokeniser.
#: Add an mabcd entry here when the mABCD schema is defined.
CONFIG_KEYS_BY_NET_TYPE: dict[str, list[str]] = {
    "abcd": ABCD_CONFIG_KEYS,
}

#: Datasets analysed by default (abcd family only for now).
ABCD_DATASETS: list[str] = ["abcd-big", "abcd-borderline", "abcd-interim"]

QWEN_MODEL_NAME: str = "Qwen/Qwen3-4B-Base"


def _resolve_hf_cache() -> str:
    """
    Resolve the HuggingFace tokeniser cache directory.

    Checks the sibling DCBA repo's cache first; falls back to a local ``.hf-cache``
    directory inside this repo (created on demand, excluded from git).

    :returns: Absolute path string suitable for ``cache_dir`` in ``from_pretrained``.
    """
    dcba_cache = (_REPO_ROOT.parent / "DCBA" / ".wandb" / "hf-cache").resolve()
    if dcba_cache.is_dir():
        logger.info(f"Using HuggingFace cache at {dcba_cache}")
        return str(dcba_cache)
    local_cache = (_REPO_ROOT / ".hf-cache").resolve()
    local_cache.mkdir(exist_ok=True)
    logger.info(f"DCBA cache not found; downloading tokeniser to {local_cache}")
    return str(local_cache)


def _load_token_counts(
    dataset_root: Path,
    config_keys: list[str],
    tokenizer: AutoTokenizer,
) -> list[int]:
    """
    Load all configs from ``dataset_root``, serialise, and return per-config token counts.

    :param dataset_root: Root directory of the dataset (passed to :func:`load_dataset`).
    :param config_keys: Config keys to include in the JSON string.
    :param tokenizer: Initialised Qwen tokeniser.

    :returns: List of token counts, one per instance.
    """
    records = load_dataset(dataset_root)
    counts: list[int] = []
    for r in records:
        cfg = DCBAInstanceConfig.from_instance_record(r).data
        text = json.dumps({k: float(cfg[k]) for k in config_keys}, separators=(",", ":"))
        counts.append(len(tokenizer.encode(text)))
    return counts


def _compute_stats(counts: list[int]) -> dict:
    """
    Compute summary statistics for a list of token counts.

    :param counts: Token counts per config.

    :returns: Dict with keys ``count``, ``min``, ``max``, ``mean``.
    """
    arr = np.array(counts, dtype=np.int64)
    return {
        "count": int(arr.size),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
    }


def _save_histogram(
    counts: list[int],
    dataset_name: str,
    stats: dict,
    output_dir: Path,
) -> dict:
    """
    Save a histogram PNG for ``counts`` and return the histogram data dict.

    :param counts: Token counts per config.
    :param dataset_name: Human-readable name used in the plot title and filename.
    :param stats: Pre-computed stats dict (used for axis labels).
    :param output_dir: Directory where the PNG is written.

    :returns: Dict with ``counts`` and ``bin_edges`` arrays (as lists) suitable for JSON.
    """
    arr = np.array(counts, dtype=np.int64)
    bins = np.arange(stats["min"], stats["max"] + 2)
    hist_counts, bin_edges = np.histogram(arr, bins=bins)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        bin_edges[:-1], hist_counts, width=1.0, align="edge", color="steelblue", edgecolor="none"
    )
    ax.set_xlabel("Token count")
    ax.set_ylabel("Number of configs")
    ax.set_title(
        f"{dataset_name} - config token distribution\n"
        f"n={stats['count']:,}  min={stats['min']}  max={stats['max']}  "
        f"mean={stats['mean']:.2f}"
    )
    ax.set_xlim(stats["min"] - 0.5, stats["max"] + 0.5)
    fig.tight_layout()

    png_path = output_dir / f"config_token_stats_{dataset_name}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved histogram: {png_path}")

    return {"counts": hist_counts.tolist(), "bin_edges": bin_edges.tolist()}


def main() -> None:
    """Run the token-count analysis for all ABCD datasets and write results to disk."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
        QWEN_MODEL_NAME,
        cache_dir=_resolve_hf_cache(),
    )
    logger.info(f"Loaded tokeniser: {QWEN_MODEL_NAME}")

    summary: dict = {}

    for dataset_name in ABCD_DATASETS:
        dataset_root = _DATA_ROOT / dataset_name
        if not dataset_root.exists():
            logger.warning(f"Dataset directory not found, skipping: {dataset_root}")
            continue

        logger.info(f"Processing dataset: {dataset_name}")

        reports = sorted(dataset_root.glob("**/report.json"))
        if not reports:
            logger.warning(f"No report.json found in {dataset_root}, skipping.")
            continue
        with reports[0].open(encoding="utf-8") as fh:
            net_type = json.load(fh).get("net_type", "abcd")

        if net_type not in CONFIG_KEYS_BY_NET_TYPE:
            logger.warning(
                f"net_type {net_type!r} has no registered config keys; skipping {dataset_name}."
            )
            continue

        counts = _load_token_counts(dataset_root, CONFIG_KEYS_BY_NET_TYPE[net_type], tokenizer)
        if not counts:
            logger.warning(f"No configs found in {dataset_root}.")
            continue

        stats = _compute_stats(counts)
        logger.info(
            f"{dataset_name} - count={stats['count']}  min={stats['min']}  "
            f"max={stats['max']}  mean={stats['mean']:.2f}"
        )

        histogram = _save_histogram(counts, dataset_name, stats, _OUTPUT_DIR)
        summary[dataset_name] = {"net_type": net_type, "stats": stats, "histogram": histogram}

    json_path = _OUTPUT_DIR / "config_token_stats.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info(f"Saved combined stats: {json_path}")


if __name__ == "__main__":
    main()

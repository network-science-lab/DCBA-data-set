"""Unified loader for ABCD and mABCD datasets produced by DatasetGenerator."""

import json
import logging
from dataclasses import replace
from pathlib import Path

from dcba_data_set.graph_io.data_models import InstanceRecord, ReplicaRecord

logger = logging.getLogger(__name__)

_KNOWN_NET_TYPES = {"abcd", "mabcd", "single_layer", "multi_layer"}


def load_report(report_path: Path) -> list[InstanceRecord]:
    """
    Index a dataset from a ``report.json`` manifest without reading any data files.

    Reads only ``report.json``, resolves all paths relative to its parent directory, and returns
    one :class:`InstanceRecord` per instance.  Replicas marked ``"ok": false`` are skipped with a
    warning.

    :param report_path: Path to the ``report.json`` file written by
        :class:`~dcba_data_set.ds_generator.DatasetGenerator`.

    :returns: A list of :class:`InstanceRecord` objects, one per instance in the manifest.
    """
    report_path = Path(report_path)
    root = report_path.parent

    with report_path.open(encoding="utf-8") as f:
        report = json.load(f)

    net_type: str = report["net_type"]
    if net_type not in _KNOWN_NET_TYPES:
        raise ValueError(f"Unknown net_type {net_type!r} in {report_path}.")

    records: list[InstanceRecord] = []

    for instance in report["instances"]:
        instance_id: str = instance["id"]
        config_path = root / instance.get("config", "")

        replicas: list[ReplicaRecord] = []
        for replica_entry in instance["replicas"]:
            if not replica_entry.get("ok", True):
                logger.warning(
                    "Skipping failed replica %d of instance %s",
                    replica_entry["replica"],
                    instance_id,
                )
                continue
            replicas.append(
                ReplicaRecord(
                    replica=replica_entry["replica"],
                    edges_path=root / replica_entry["edges"],
                    communities_path=root / replica_entry["communities"],
                )
            )

        if not replicas:
            logger.warning("Skipping instance %s: no valid replicas.", instance_id)
            continue
        records.append(
            InstanceRecord(
                instance_id=instance_id,
                net_type=net_type,
                config_path=config_path,
                replicas=replicas,
            )
        )

    logger.info("Indexed %d instances from %s", len(records), report_path)
    return records


def _drop_failed(records: list[InstanceRecord], chunk_root: Path) -> list[InstanceRecord]:
    """Drop instances with no replicas that are listed in an adjacent ``hard_instances.json``."""
    hard_path = chunk_root / "hard_instances.json"
    if not hard_path.exists():
        return records
    with hard_path.open(encoding="utf-8") as f:
        hard_ids = {entry["id"] for entry in json.load(f)}
    filtered = [r for r in records if r.instance_id not in hard_ids]
    dropped = len(records) - len(filtered)
    if dropped:
        logger.info("Discarded %d failed instance(s) listed in %s.", dropped, hard_path)
    return filtered


def load_dataset(dataset_root: Path, *, discard_failed: bool = True) -> list[InstanceRecord]:
    """
    Load all instances from a dataset directory, handling both flat and chunked layouts.

    **Flat layout** (e.g. ``abcd-interim``, ``abcd-borderline``): ``report.json`` sits directly
    inside ``dataset_root``.

    **Chunked layout** (e.g. ``abcd-big``): ``report.json`` files sit inside immediate
    subdirectories of ``dataset_root``; chunks without a ``report.json`` are silently skipped.
    Each instance's ``instance_id`` is prefixed with the chunk directory name
    (e.g. ``chunk-7/3f1a...``).

    When ``discard_failed=True`` and a ``hard_instances.json`` file exists alongside a
    ``report.json``, all instances listed in ``hard_instances.json`` are dropped regardless of
    how many replicas they produced.

    :param dataset_root: Root directory of the dataset.
    :param discard_failed: Drop fully-failed instances recorded in ``hard_instances.json``.

    :returns: A flat list of :class:`InstanceRecord` objects across all chunks.
    """
    dataset_root = Path(dataset_root)

    flat_report = dataset_root / "report.json"
    if flat_report.exists():
        logger.info("----------------------------------------")
        records = load_report(flat_report)
        if discard_failed:
            records = _drop_failed(records, dataset_root)
        total_graphs = sum(len(r.replicas) for r in records)
        logger.info(
            "Loaded %d instances (%d graphs) from %s.",
            len(records),
            total_graphs,
            dataset_root,
        )
        logger.info("----------------------------------------")
        return records

    chunk_dirs = sorted(
        d for d in dataset_root.iterdir() if d.is_dir() and (d / "report.json").exists()
    )
    if not chunk_dirs:
        raise ValueError(f"No report.json found in {dataset_root} or its immediate subdirectories.")

    all_records: list[InstanceRecord] = []
    for chunk_dir in chunk_dirs:
        logger.info("----------------------------------------")
        logger.info("Loading chunk %s", chunk_dir.name)
        records = load_report(chunk_dir / "report.json")
        if discard_failed:
            records = _drop_failed(records, chunk_dir)
        chunk_name = chunk_dir.name
        all_records.extend(replace(r, instance_id=f"{chunk_name}-{r.instance_id}") for r in records)

    total_graphs = sum(len(r.replicas) for r in all_records)
    logger.info("----------------------------------------")
    logger.info(
        "Loaded %d instances (%d graphs) from %d chunk(s) in %s.",
        len(all_records),
        total_graphs,
        len(chunk_dirs),
        dataset_root,
    )
    logger.info("----------------------------------------")
    return all_records

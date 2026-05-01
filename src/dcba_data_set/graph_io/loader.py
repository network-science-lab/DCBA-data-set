"""Unified loader for ABCD and mABCD datasets produced by DatasetGenerator."""

import json
import logging
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

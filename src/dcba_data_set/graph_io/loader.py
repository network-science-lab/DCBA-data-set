"""Unified loader for ABCD and mABCD datasets produced by DatasetGenerator."""

import json
import logging
from pathlib import Path

import yaml

from dcba_data_set.graph_io.data_models import ConfigRecord, DCBAHeteroData

logger = logging.getLogger(__name__)

_FACTORY = {
    "abcd": DCBAHeteroData.from_abcd_files,
    "mabcd": DCBAHeteroData.from_mabcd_files,
}


def load_report(
    report_path: Path,
) -> tuple[dict[str, ConfigRecord], list[DCBAHeteroData]]:
    """
    Load an ABCD or mABCD dataset from a ``report.json`` manifest.

    The graph type is determined by the ``net_type`` field in the manifest
    (``"abcd"`` or ``"mabcd"``). Each replica of each instance becomes one
    :class:`DCBAHeteroData` object. Replicas marked ``"ok": false`` are skipped
    with a warning.

    :param report_path: Path to the ``report.json`` file written by
        :class:`~dcba_data_set.ds_generator.DatasetGenerator`.
    :returns: A tuple ``(configs, graphs)`` where:

        - ``configs`` maps ``instance_id → ConfigRecord`` (one entry per instance).
        - ``graphs`` is a flat list of :class:`DCBAHeteroData` objects, one per
          successful replica across all instances.
    :raises ValueError: If the ``net_type`` field in the manifest is not recognised.
    """
    report_path = Path(report_path)
    root = report_path.parent

    with report_path.open(encoding="utf-8") as f:
        report = json.load(f)

    net_type: str = report["net_type"]
    if net_type not in _FACTORY:
        raise ValueError(f"Unknown net_type {net_type!r} in {report_path}.")
    factory = _FACTORY[net_type]

    configs: dict[str, ConfigRecord] = {}
    graphs: list[DCBAHeteroData] = []

    for instance in report["instances"]:
        instance_id: str = instance["id"]

        config_path = root / instance["config"]
        with config_path.open(encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        configs[instance_id] = ConfigRecord(
            instance_id=instance_id,
            data=config_data,
            path=config_path,
        )

        for replica_entry in instance["replicas"]:
            if not replica_entry.get("ok", True):
                logger.warning(
                    "Skipping failed replica %d of instance %s",
                    replica_entry["replica"],
                    instance_id,
                )
                continue

            replica_idx: int = replica_entry["replica"]
            edges_path = root / replica_entry["edges"]
            communities_path = root / replica_entry["communities"]

            graph = factory(
                edges_path=edges_path,
                communities_path=communities_path,
                instance_id=instance_id,
                replica=replica_idx,
            )
            graphs.append(graph)

    logger.info(
        "Loaded %d configs and %d graphs from %s",
        len(configs),
        len(graphs),
        report_path,
    )
    return configs, graphs

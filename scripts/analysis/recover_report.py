# ruff: noqa
"""Recover a missing report.json for ABCD dataset chunks from filesystem state.

For each chunk directory supplied, the script:
  - skips the chunk if report.json already exists
  - reads the INCOMPLETE sentinel (if present) to identify the instance that was
    mid-generation when the process died, then verifies it is absent from disk
  - scans every UUID instance directory for complete (edges_N.dat, communities_N.dat)
    pairs and treats each pair as one valid replica
  - writes report.json with all discovered instances and their complete replicas

Usage:
    uv run scripts/analysis/recover_report.py data/abcd-big/chunk-1 data/abcd-big/chunk-3 ...
    uv run scripts/analysis/recover_report.py --net-type abcd data/abcd-big/chunk-5
"""

import argparse
import json
import logging
import re
import sys
import uuid
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_STUCK_INSTANCE_RE = re.compile(r"Julia got stuck at instance: ([0-9a-f-]{36})")


def _read_net_type(chunk_path: Path) -> str | None:
    for yaml_file in chunk_path.glob("*.yaml"):
        try:
            cfg = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
            net_type = (cfg or {}).get("generator", {}).get("net_type")
            if net_type:
                return net_type
        except Exception:
            pass
    return None


def _parse_stuck_instance(chunk_path: Path) -> str | None:
    sentinel = chunk_path / "INCOMPLETE"
    if not sentinel.exists():
        return None
    m = _STUCK_INSTANCE_RE.search(sentinel.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def _is_uuid_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        uuid.UUID(path.name)
        return True
    except ValueError:
        return False


def recover_chunk(chunk_path: Path, net_type: str) -> None:
    report_path = chunk_path / "report.json"
    if report_path.exists():
        logger.info("Skipping %s: report.json already exists.", chunk_path)
        return

    stuck_id = _parse_stuck_instance(chunk_path)
    if stuck_id:
        if (chunk_path / stuck_id).exists():
            logger.warning(
                "Stuck instance directory %s is still present in %s — its replicas may be "
                "incomplete. It will be included only with the replica pairs that exist.",
                stuck_id,
                chunk_path,
            )
        else:
            logger.info(
                "Confirmed: stuck instance %s has been removed from %s.", stuck_id, chunk_path
            )

    instances = []
    for instance_dir in sorted(chunk_path.iterdir()):
        if not _is_uuid_dir(instance_dir):
            continue

        edges_indices = {
            int(m.group(1))
            for f in instance_dir.iterdir()
            if (m := re.fullmatch(r"edges_(\d+)\.dat", f.name))
        }
        comm_indices = {
            int(m.group(1))
            for f in instance_dir.iterdir()
            if (m := re.fullmatch(r"communities_(\d+)\.dat", f.name))
        }
        complete = sorted(edges_indices & comm_indices)

        if not complete and not (instance_dir / "config.yaml").exists():
            logger.warning("Skipping %s: no config.yaml and no replica files found.", instance_dir)
            continue

        instances.append(
            {
                "id": instance_dir.name,
                "config": f"{instance_dir.name}/config.yaml",
                "replicas": [
                    {
                        "replica": n,
                        "edges": f"{instance_dir.name}/edges_{n}.dat",
                        "communities": f"{instance_dir.name}/communities_{n}.dat",
                        "ok": True,
                    }
                    for n in complete
                ],
            }
        )

    report = {"net_type": net_type, "instances": instances}
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    replica_counts = [len(inst["replicas"]) for inst in instances]
    complete_instances = sum(1 for c in replica_counts if c == max(replica_counts, default=0))
    logger.info(
        "Wrote %s (%d instances, %d with full replica set).",
        report_path,
        len(instances),
        complete_instances,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "chunks",
        nargs="+",
        type=Path,
        metavar="CHUNK_DIR",
        help="One or more chunk directories to recover.",
    )
    parser.add_argument(
        "--net-type",
        default=None,
        metavar="TYPE",
        help="Override net_type (read from dataset YAML by default).",
    )
    args = parser.parse_args(argv)

    exit_code = 0
    for chunk_path in args.chunks:
        if not chunk_path.is_dir():
            logger.error("%s is not a directory.", chunk_path)
            exit_code = 1
            continue
        net_type = args.net_type or _read_net_type(chunk_path)
        if not net_type:
            logger.error("Cannot determine net_type for %s. Supply --net-type.", chunk_path)
            exit_code = 1
            continue
        recover_chunk(chunk_path, net_type)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

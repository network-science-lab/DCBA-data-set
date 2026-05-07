"""Module to generate datasets based on specified configurations."""

import json
import logging
import uuid
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from dcba_data_set.config_generators import (
    BorderlineConfigGenerator,
    GridConfigGenerator,
    RandomConfigGenerator,
)
from dcba_data_set.julia_ports.abcd import ABCDGraphGenerator
from dcba_data_set.julia_ports.mabcd import mABCDGraphGenerator
from dcba_data_set.utils import create_out_dir

logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Generate a dataset of graphs by sampling configurations and running the graph generator."""

    def __call__(self, config: dict[str, Any]) -> Any:
        """Run the dataset generation loop and write outputs and a report to ``out_dir``."""
        net_ranges = config["net_ranges"]
        gen_cfg = config["generator"]
        net_type = gen_cfg["net_type"]
        out_dir = create_out_dir(gen_cfg["out_dir"])
        sampling = gen_cfg["sampling"]
        method = sampling["method"]
        net_replicas = sampling["replicas"]

        if net_type == "abcd":
            generator = ABCDGraphGenerator
        elif net_type == "mabcd":
            generator = mABCDGraphGenerator
        else:
            raise ValueError(f"Unknown network type: {net_type}")

        report = {"net_type": net_type, "instances": []}

        if method == "grid":
            configs = GridConfigGenerator(
                cfg_type=net_type, grid_steps=sampling["grid_steps"]
            ).generate(net_ranges)
        elif method == "random":
            configs = RandomConfigGenerator(
                rng_seed=config["run"]["rng_seed"],
                cfg_type=net_type,
                max_trials=sampling["max_trials"],
            ).generate(net_ranges, sampling["instances"])
        elif method == "borderline":
            configs = BorderlineConfigGenerator(cfg_type=net_type).generate(net_ranges)
        else:
            raise ValueError(f"Unknown sampling method: {method!r}")

        self._run_instances(configs, generator, net_replicas, config, out_dir, report)

        with open(out_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    def _run_instances(
        self,
        configs: list[Any],
        generator: Any,
        net_replicas: int,
        config: dict[str, Any],
        out_dir: Path,
        report: dict[str, Any],
    ) -> None:
        """Iterate over a config list, create instance dirs and run graph generation."""
        p_bar = tqdm(configs, desc="Instance", leave=False, colour="green")
        for sampled_config in p_bar:
            instance_id = str(uuid.uuid4())[:8]
            instance_dir = out_dir / instance_id
            instance_dir.mkdir(parents=True, exist_ok=True)
            self._run_instance(
                sampled_config, instance_id, instance_dir, generator, net_replicas, config, report
            )

    def _run_instance(
        self,
        sampled_config: Any,
        instance_id: str,
        instance_dir: Path,
        generator: Any,
        net_replicas: int,
        config: dict[str, Any],
        report: dict[str, Any],
    ) -> None:
        """Write config to disk and generate all replicas for a single instance."""
        with open(instance_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(sampled_config.to_yaml(), f)

        instance_record = {
            "id": instance_id,
            "config": f"{instance_id}/config.yaml",
            "replicas": [],
        }

        for replica in range(1, net_replicas + 1):
            edges_rel = f"{instance_id}/edges_{replica}.dat"
            communities_rel = f"{instance_id}/communities_{replica}.dat"
            sampled_config.edges_filename = str(instance_dir / f"edges_{replica}.dat")
            sampled_config.communities_filename = str(instance_dir / f"communities_{replica}.dat")
            sampled_config.seed = config["run"]["rng_seed"]

            try:
                generator()(sampled_config)
                instance_record["replicas"].append(
                    {
                        "replica": replica,
                        "edges": edges_rel,
                        "communities": communities_rel,
                        "ok": True,
                    }
                )
            except Exception as e:
                logger.error("Instance %s replica %d failed: %s", instance_id, replica, e)
                instance_record["replicas"].append(
                    {
                        "replica": replica,
                        "edges": edges_rel,
                        "communities": communities_rel,
                        "ok": False,
                        "error": str(e),
                    }
                )

        report["instances"].append(instance_record)

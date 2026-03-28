"""Module to generate datasets based on specified configurations."""

import json
import logging
import uuid
from typing import Any, Literal

import numpy as np
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

from dcba_data_set.params_handler import create_out_dir
from dcba_data_set.julia_ports.mabcd import mABCDConfig, mABCDGraphGenerator
from dcba_data_set.julia_ports.abcd import ABCDConfig, ABCDGraphGenerator


# TODO: update this class to support mABCD whose config shape depends on the number of layers (n).
class ConfigGenerator:

    def __init__(self, rng_seed: int, cfg_type: Literal["abcd", "mabcd"], max_trials: int) -> None:
        self.rng = np.random.default_rng(seed=rng_seed)
        self.julia_config = ABCDConfig if cfg_type == "abcd" else mABCDConfig
        self.max_trials = max_trials

    def sample_from_range(self, smaller: float | int, bigger: float | int) -> float | int:
        """Sample a number from the given range using the specified distribution."""
        return self.rng.uniform(smaller, bigger)

    def _draw_config(self, config_ranges: dict[str, Any]) -> dict[str, Any]:
        sampled_config = {}
        for param_name, param_range in config_ranges.items():
            if isinstance(param_range, list) and len(param_range) == 2:
                param = self.sample_from_range(param_range[0], param_range[1])
                param = int(param) if isinstance(param_range[0], int) else round(param, 4)
                sampled_config[param_name] = param
            else:
                sampled_config[param_name] = param_range
        return sampled_config

    def __call__(self, config_ranges: dict[str, Any]) -> Any:
        for trial in range(1, self.max_trials + 1):
            sampled_config_dict = self._draw_config(config_ranges)
            try:
                return self.julia_config.from_yaml(sampled_config_dict)
            except Exception as e:
                logger.warning("Trial %d/%d failed: %s", trial, self.max_trials, e)
        raise RuntimeError(f"Exceeded {self.max_trials} trials without a valid configuration.")


class DatasetGenerator:

    def __call__(self, config: dict[str, Any]) -> Any:
        
        net_ranges = config["net_ranges"]
        net_type = config["generator"]["net_type"]
        net_replicas = config["generator"]["replicas"]
        instances = config["generator"]["instances"]
        max_trials = config["generator"]["max_trials"]
        out_dir = create_out_dir(config["generator"]["out_dir"])

        if net_type == "abcd":
            generator = ABCDGraphGenerator
        elif net_type == "mabcd":
            generator = mABCDGraphGenerator
        else:
            raise ValueError(f"Unknown network type: {net_type}")

        cg = ConfigGenerator(
            rng_seed=config["run"]["rng_seed"],
            cfg_type=net_type,
            max_trials=max_trials
        )

        report = {"net_type": net_type, "instances": []}

        p_bar = tqdm(np.arange(instances), desc="", leave=False, colour="green")
        for instance in p_bar:
            p_bar.set_description_str("Instance")

            instance_id = str(uuid.uuid4())[:8]
            instance_dir = out_dir / instance_id
            instance_dir.mkdir(parents=True, exist_ok=True)

            try:
                sampled_config = cg(net_ranges)
            except RuntimeError as e:
                logger.error("Skipping instance %d — %s", instance, e)
                instance_dir.rmdir()
                report["instances"].append(
                    {"id": instance_id, "status": "skipped", "error": str(e), "replicas": []}
                )
                continue

            with open(instance_dir / "config.yaml", "w") as f:
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
                    logger.error(
                        "Instance %d replica %d failed during generation: %s", instance, replica, e
                    )
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

        with open(out_dir / "report.json", "w") as f:
            json.dump(report, f, indent=2)

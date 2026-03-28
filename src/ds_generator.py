"""Module to generate datasets based on specified configurations."""

import uuid
from typing import Any, Literal

import numpy as np
import yaml
from tqdm import tqdm

from src.params_handler import create_out_dir
from src.julia_ports.mabcd import mABCDConfig, mABCDGraphGenerator
from src.julia_ports.abcd import ABCDConfig, ABCDGraphGenerator


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
        trial_nbr = 1
        while trial_nbr < self.max_trials:
            sampled_config_dict = self._draw_config(config_ranges)
            try:
                sampled_config = self.julia_config.from_yaml(sampled_config_dict)
                return sampled_config
            except Exception as e:
                trial_nbr += 1
        raise RuntimeError("Max trials exceeded while sampling a valid configuration.")


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

        p_bar = tqdm(np.arange(instances), desc="", leave=False, colour="green")
        for instance in p_bar:
            p_bar.set_description_str("Instance")

            instance_dir = out_dir / str(uuid.uuid4())[:8]
            instance_dir.mkdir(parents=True, exist_ok=True)

            sampled_config = cg(net_ranges)
            with open(instance_dir / "config.yaml", "w") as f:
                yaml.dump(sampled_config.to_yaml(), f)

            for replica in range(1, net_replicas + 1):

                sampled_config.edges_filename = str(instance_dir / f"edges_{replica}.dat")
                sampled_config.communities_filename = str(instance_dir / f"communities_{replica}.dat")
                sampled_config.seed = config["run"]["rng_seed"]

                generator()(sampled_config)

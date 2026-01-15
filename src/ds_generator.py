"""Module to generate datasets based on specified configurations."""

import uuid
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

from src.params_handler import create_out_dir
from src.julia_ports.mabcd import MLNConfig, MLNABCDGraphGenerator
from src.julia_ports.abcd import ABCDConfig, ABCDGraphGenerator


class ConfigGenerator:

    def __init__(self, rng_seed: int) -> None:
        self.rng = np.random.default_rng(seed=rng_seed)

    def sample_from_range(self, smaller: float | int, bigger: float | int) -> float | int:
        """Sample a number from the given range using the specified distribution."""
        return self.rng.uniform(smaller, bigger)

    def __call__(self, config_ranges: dict[str, Any]) -> Any:
        sampled_config = {}
        for param_name, param_range in config_ranges.items():
            if isinstance(param_range, list) and len(param_range) == 2:
                param = self.sample_from_range(param_range[0], param_range[1])
                param = int(param) if isinstance(param_range[0], int) else round(param, 4)
                sampled_config[param_name] = param
            else:
                sampled_config[param_name] = param_range
        return sampled_config
    



class DatasetGenerator:

    def __call__(self, config: dict[str, Any]) -> Any:
        
        net_ranges = config["net_ranges"]
        cg = ConfigGenerator(rng_seed=config["run"]["rng_seed"])
        net_type = config["generator"]["net_type"]
        net_replicas = config["generator"]["replicas"]
        instances = config["generator"]["instances"]
        out_dir = create_out_dir(config["generator"]["out_dir"])

        if net_type == "abcd":
            config_handler, generator = ABCDConfig, ABCDGraphGenerator
        elif net_type == "mabcd":
            config_handler, generator = MLNConfig, MLNABCDGraphGenerator
        else:
            raise ValueError(f"Unknown network type: {net_type}")

        p_bar = tqdm(np.arange(instances), desc="", leave=False, colour="green")
        for instance in p_bar:
            p_bar.set_description_str("Instance")

            instance_dir = out_dir / str(uuid.uuid4())[:8]
            instance_dir.mkdir(parents=True, exist_ok=True)

            smapled = False
            trials = 0
            while not smapled:
                try:
                    trials += 1
                    sampled_config = cg(net_ranges)
                    net_config = config_handler.from_yaml(sampled_config)
                    sampled = True
                except BaseException as e:
                    pass
                if trials >= 10:
                    print("Exceeded maximum trials for sampling configuration.")
                    continue

            with open(instance_dir / "config.yaml", "w") as f:
                yaml.dump(sampled_config, f)

            for replica in range(1, net_replicas + 1):

                net_config.edges_filename = str(instance_dir / f"edges_{replica}.dat")
                net_config.communities_filename = str(instance_dir / f"communities_{replica}.dat")
                net_config.seed = config["run"]["rng_seed"]

                generator()(net_config)

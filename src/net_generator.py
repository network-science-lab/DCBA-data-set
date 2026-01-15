"""Main runner of the generator."""

from typing import Any

import juliacall
import numpy as np
from tqdm import tqdm

from src.julia_ports.mabcd import MLNConfig, MLNABCDGraphGenerator
from src.julia_ports.abcd import ABCDConfig, ABCDGraphGenerator
from src.params_handler import create_out_dir


class MABCDGenerator:
    """Wrapper for the MABCD graph generator runnable from CLI."""

    julia_class = MLNABCDGraphGenerator
    julia_config = MLNConfig

    def run_experiments(self, config: dict[str, Any]) -> None:

        _net_config = config["net_config"]
        _net_config["seed"] = config["run"]["rng_seed"]
        net_config = self.julia_config.from_yaml(_net_config)
        repetitions = config["generator"]["repetitions"]
        out_dir = create_out_dir(config["generator"]["out_dir"])
        e_name, e_stem = config["net_config"]["edges_filename"].split(".")
        c_name, c_stem = config["net_config"]["communities_filename"].split(".")

        p_bar = tqdm(np.arange(repetitions), desc="", leave=False, colour="green")
        for repetition in p_bar:
            p_bar.set_description_str("Repetition")
            net_config.edges_filename = str(out_dir / f"{e_name}_{repetition}.{e_stem}")
            net_config.communities_filename = str(out_dir / f"{c_name}_{repetition}.{c_stem}")
            self.julia_class()(config=net_config)

    def __call__(self, config: dict[str, Any]) -> None:
        return self.run_experiments(config)
    

class ABCDGenerator(MABCDGenerator):
    """Wrapper for the ABCD graph generator runnable from CLI."""

    julia_class = ABCDGraphGenerator
    julia_config = ABCDConfig

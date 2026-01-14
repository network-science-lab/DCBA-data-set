"""Main runner of the generator."""

from typing import Any

import juliacall
import numpy as np
from tqdm import tqdm

from src.mln_abcd.julia_wrapper import MLNConfig, MLNABCDGraphGenerator
from src.params_handler import create_out_dir


def run_experiments(config: dict[str, Any]) -> None:

    _mln_config = config["mln_config"]
    _mln_config["seed"] = config["run"]["rng_seed"]
    mln_config = MLNConfig.from_yaml(_mln_config)

    repetitions = config["generator"]["repetitions"]
    out_dir = create_out_dir(config["generator"]["out_dir"])
    e_name, e_stem = config["mln_config"]["edges_filename"].split(".")
    c_name, c_stem = config["mln_config"]["communities_filename"].split(".")

    p_bar = tqdm(np.arange(repetitions), desc="", leave=False, colour="green")
    for repetition in p_bar:
        p_bar.set_description_str("Repetition")
        mln_config.edges_filename = str(out_dir / f"{e_name}_{repetition}.{e_stem}")
        mln_config.communities_filename = str(out_dir / f"{c_name}_{repetition}.{c_stem}")
        MLNABCDGraphGenerator()(config=mln_config)

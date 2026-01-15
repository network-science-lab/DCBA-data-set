import argparse
import yaml

from src.generator import ABCDGenerator, MABCDGenerator
from src.utils import set_rng_seed


def parse_args(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        help="Experiment config file (default: config.yaml).",
        nargs="?",
        type=str,
        default="scripts/configs/example_generate/mabcd.yaml",
        # default="scripts/configs/example_generate/abcd.yaml",
    )
    return parser.parse_args(*args)


if __name__ == "__main__":

    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"Loaded config: {config}")
    
    if random_seed := config["run"].get("random_seed"):
        print(f"Setting randomness seed as {random_seed}!")
        set_rng_seed(config["run"]["random_seed"])

    if (experiment_type := config["run"].get("experiment_type")) == "simulate":
        entrypoint = lambda x: x
    elif experiment_type == "generate-mabcd":
        entrypoint = MABCDGenerator()
    elif experiment_type == "generate-abcd":
        entrypoint = ABCDGenerator()
    else:
        raise ValueError(f"Unknown experiment type {experiment_type}")

    print(f"Inferred experiment type as: {experiment_type}")
    entrypoint(config)

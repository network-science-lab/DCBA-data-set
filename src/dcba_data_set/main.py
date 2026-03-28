import argparse
import logging
import yaml

from dcba_data_set.net_generator import ABCDGenerator, mABCDGenerator
from dcba_data_set.ds_generator import DatasetGenerator
from dcba_data_set.utils import set_rng_seed

logger = logging.getLogger(__name__)


# TODO: consider replacing argparse with hydra for config management
def parse_args(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        help="Experiment config file (default: config.yaml).",
        nargs="?",
        type=str,
        # default="scripts/configs/example_generate/mabcd.yaml",
        # default="scripts/configs/example_generate/abcd.yaml",
        default="scripts/configs/example_generate/dataset-abcd.yaml",
        # default="scripts/configs/example_generate/dataset-mabcd.yaml",
    )
    return parser.parse_args(*args)


def main() -> None:
    """Main entrypoint for the DCBA dataset generator."""
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config: %s", config)

    if random_seed := config["run"].get("random_seed"):
        logger.info("Setting randomness seed as %s", random_seed)
        set_rng_seed(config["run"]["random_seed"])

    if (experiment_type := config["run"].get("experiment_type")) == "generate-dataset":
        entrypoint = DatasetGenerator()
    elif experiment_type == "generate-mabcd":
        entrypoint = mABCDGenerator()
    elif experiment_type == "generate-abcd":
        entrypoint = ABCDGenerator()
    else:
        raise ValueError(f"Unknown experiment type {experiment_type}")

    logger.info("Inferred experiment type as: %s", experiment_type)
    entrypoint(config)


if __name__ == "__main__":
    main()

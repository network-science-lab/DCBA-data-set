"""Utility functions for various common tasks."""

import logging
import warnings

from network_diffusion.utils import fix_random_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

warnings.filterwarnings(action="ignore", category=FutureWarning)


def set_rng_seed(seed: int) -> None:
    fix_random_seed(seed=seed)  # TODO: use it directly from nd once new version is released

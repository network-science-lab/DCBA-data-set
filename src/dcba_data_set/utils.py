"""Utility functions for common tasks."""

import logging
import tempfile
import warnings
from pathlib import Path

from network_diffusion.utils import fix_random_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

warnings.filterwarnings(action="ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


def set_rng_seed(seed: int) -> None:
    """Set the global random seed for reproducible runs."""
    fix_random_seed(seed=seed)  # TODO: use it directly from nd once new version is released


def create_out_dir(out_dir: str) -> Path:
    """
    Create and return the output directory, falling back to a temp dir on collision.

    :param out_dir: Desired output directory path as a string.
    :returns: A :class:`~pathlib.Path` pointing to the directory that was created.
    """
    try:
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(exist_ok=True, parents=True)
    except FileExistsError:
        logger.warning("Redirecting output to a temporary directory after existing-path collision.")
        out_dir_path = Path(tempfile.mkdtemp())
    return out_dir_path

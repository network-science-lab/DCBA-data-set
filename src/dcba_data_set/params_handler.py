"""Functions to facilitate setting up output directories."""

import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


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

"""A script with strings used across module in form of variables."""

# TODO: review this code and decidee whether it's needed

from functools import wraps
from pathlib import Path
from typing import Callable


MLN_ABCD_DATA_PATH = Path(__file__).parent.parent.parent / "data/nets_generated"

SEPARATOR = "^"


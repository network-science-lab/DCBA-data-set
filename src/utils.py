"""Utility functions for various common tasks."""

# TODO: review this code and decidee whether it's needed

import datetime
import logging
import warnings

import git
from network_diffusion.utils import fix_random_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


warnings.filterwarnings(action="ignore", category=FutureWarning)


def get_current_time():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def get_diff_of_times(strftime_1, strftime_2):
    fmt = "%Y-%m-%d %H:%M:%S"
    t_1 = datetime.datetime.strptime(strftime_1, fmt)
    t_2 = datetime.datetime.strptime(strftime_2, fmt)
    return t_2 - t_1


def get_recent_git_sha() -> str:
    repo = git.Repo(search_parent_directories=True)
    git_sha = repo.head.object.hexsha
    return git_sha


def set_rng_seed(seed: int) -> None:
    fix_random_seed(seed=seed) # TODO: use it directly from nd once new version is released

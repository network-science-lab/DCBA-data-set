"""Config generators for random, grid-based, and borderline dataset sampling."""

import itertools
import logging
from typing import Any, Literal

import numpy as np

from dcba_data_set.julia_ports.abcd import ABCDConfig
from dcba_data_set.julia_ports.mabcd import mABCDConfig

logger = logging.getLogger(__name__)


def denormalise_abcd_config(params: dict[str, Any]) -> dict[str, Any]:
    """
    Convert relative ABCD parameters to absolute values.

    Fractional keys (``*_frac``) are multiplied by ``n`` and rounded to the nearest
    integer. Cross-parameter floors are applied to prevent trivially invalid configs
    (e.g. ``d_max < d_min``); the Pydantic validator in ``ABCDConfig`` is the
    authoritative guard for the rest.

    Affected keys (all others passed through unchanged):

    - ``d_min_frac`` → ``d_min``
    - ``d_max_frac`` → ``d_max``
    - ``c_min_frac`` → ``c_min``
    - ``c_max_frac`` → ``c_max``
    - ``nout_frac``  → ``nout``

    :param params: Parameter dict possibly containing ``*_frac`` keys alongside ``n``.

    :returns: New dict with fractional keys replaced by their absolute counterparts.
    """
    n: int = params["n"]
    result = {k: v for k, v in params.items() if not k.endswith("_frac")}

    if "d_min_frac" in params:
        d_min = max(1, round(params["d_min_frac"] * n))
        result["d_min"] = d_min
    if "d_max_frac" in params:
        d_min = result.get("d_min", 1)
        result["d_max"] = max(d_min + 1, round(params["d_max_frac"] * n))
    if "c_min_frac" in params:
        c_min = max(1, round(params["c_min_frac"] * n))
        result["c_min"] = c_min
    if "c_max_frac" in params:
        c_min = result.get("c_min", 1)
        result["c_max"] = max(c_min + 1, min(n, round(params["c_max_frac"] * n)))
    if "nout_frac" in params:
        result["nout"] = round(params["nout_frac"] * n)

    return result


# TODO: update this class to support mABCD whose config shape depends on the number of layers (n).
class RandomConfigGenerator:
    """Randomly sample valid graph configurations from parameter ranges."""

    def __init__(self, rng_seed: int, cfg_type: Literal["abcd", "mabcd"], max_trials: int) -> None:
        """Initialise the generator with a random seed, config type, and retry limit."""
        self.rng = np.random.default_rng(seed=rng_seed)
        self.julia_config = ABCDConfig if cfg_type == "abcd" else mABCDConfig
        self.max_trials = max_trials

    def _sample_from_range(self, smaller: float | int, bigger: float | int) -> float | int:
        """Return a single value sampled uniformly from ``[smaller, bigger]``."""
        return self.rng.uniform(smaller, bigger)

    def _draw_config(self, config_ranges: dict[str, Any]) -> dict[str, Any]:
        """
        Sample one parameter dict from *config_ranges*.

        List values are sampled uniformly; scalar values are passed through unchanged.
        ``denormalise_abcd_config`` is applied afterwards for ABCD configs.

        :param config_ranges: Parameter ranges from the dataset YAML.

        :returns: Sampled parameter dict ready for ``from_yaml``.
        """
        sampled_config = {}
        for param_name, param_range in config_ranges.items():
            if isinstance(param_range, list) and len(param_range) == 2:
                param = self._sample_from_range(param_range[0], param_range[1])
                param = int(param) if isinstance(param_range[0], int) else round(param, 4)
                sampled_config[param_name] = param
            else:
                sampled_config[param_name] = param_range

        if self.julia_config is ABCDConfig:
            sampled_config = denormalise_abcd_config(sampled_config)

        return sampled_config

    def _sample_one(self, config_ranges: dict[str, Any]) -> Any:
        """
        Sample and validate a single config, retrying up to ``max_trials`` times.

        :param config_ranges: Parameter ranges from the dataset YAML.

        :raises RuntimeError: If no valid configuration is found within the trial limit.
        """
        for trial in range(1, self.max_trials + 1):
            sampled_config_dict = self._draw_config(config_ranges)
            try:
                return self.julia_config.from_yaml(sampled_config_dict)
            except Exception as e:
                logger.warning("Trial %d/%d failed: %s", trial, self.max_trials, e)
        raise RuntimeError(f"Exceeded {self.max_trials} trials without a valid configuration.")

    def generate(self, config_ranges: dict[str, Any], n_instances: int) -> list[Any]:
        """
        Return up to *n_instances* valid configs sampled from *config_ranges*.

        Failed sampling attempts are logged at ERROR level and skipped; the returned
        list may be shorter than *n_instances* if some attempts exhaust ``max_trials``.

        :param config_ranges: Parameter ranges from the dataset YAML.
        :param n_instances: Number of valid configs to attempt to produce.

        :returns: List of validated config instances.
        """
        configs = []
        for i in range(n_instances):
            try:
                configs.append(self._sample_one(config_ranges))
            except RuntimeError as e:
                logger.error("Skipping instance %d: %s", i, e)
        return configs


class GridConfigGenerator:
    """Generate all valid configs in the Cartesian product of a parameter grid."""

    def __init__(
        self, cfg_type: Literal["abcd", "mabcd"], grid_steps: int | dict[str, int]
    ) -> None:
        """
        Initialise the grid generator.

        :param cfg_type: Network type; only ``"abcd"`` is currently supported.
        :param grid_steps: Number of evenly-spaced values per ``[min, max]`` range.
            Either a single integer applied uniformly to all range parameters, or a
            mapping from parameter name to its individual step count. When a mapping is
            given, every parameter whose value is a ``[min, max]`` range must have a
            corresponding entry.
        """
        if cfg_type != "abcd":
            raise NotImplementedError(
                f"Grid sampling is not yet supported for cfg_type={cfg_type!r}"
            )
        self.julia_config = ABCDConfig
        self.grid_steps = grid_steps

    def _steps_for(self, param_name: str) -> int:
        """
        Return the number of grid steps for *param_name*.

        :param param_name: Name of the parameter being gridded.

        :returns: Step count for this parameter.

        :raises ValueError: If no grid_steps entry is found for the parameter.
        """
        if isinstance(self.grid_steps, dict):
            if param_name not in self.grid_steps:
                raise ValueError(
                    f"No grid_steps entry for range parameter {param_name!r}. "
                    "Add it to the grid_steps mapping or use a single integer value."
                )
            return self.grid_steps[param_name]
        return self.grid_steps

    def generate(self, config_ranges: dict[str, Any]) -> list[Any]:
        """
        Return all valid configs in the Cartesian product of the parameter grid.

        For each ``[min, max]`` pair in *config_ranges*, the corresponding number of
        evenly-spaced values is produced via ``numpy.linspace`` — either the global
        ``grid_steps`` integer or the per-parameter value from the ``grid_steps``
        mapping. Fixed (non-list) values are held constant. ``denormalise_abcd_config``
        is applied to each combination before Pydantic validation; invalid points are
        silently dropped (one WARNING per point).

        :param config_ranges: Parameter ranges from the dataset YAML.

        :returns: List of validated config instances.
        """
        param_names: list[str] = []
        param_grids: list[list[Any]] = []

        for name, value in config_ranges.items():
            if isinstance(value, list) and len(value) == 2:
                steps = self._steps_for(name)
                lo, hi = value
                if isinstance(lo, int) and isinstance(hi, int):
                    raw = np.linspace(lo, hi, steps)
                    grid: list[Any] = list(np.unique(raw.astype(int)))
                else:
                    grid = list(np.linspace(lo, hi, steps))
            else:
                grid = [value]
            param_names.append(name)
            param_grids.append(grid)

        total = sum(1 for _ in itertools.product(*param_grids))
        valid_configs: list[Any] = []

        for combo in itertools.product(*param_grids):
            params = dict(zip(param_names, combo))
            params = denormalise_abcd_config(params)
            try:
                valid_configs.append(self.julia_config.from_yaml(params))
            except Exception as e:
                logger.warning("Dropping grid point %s: %s", params, e)

        logger.info("Grid: %d valid / %d attempted", len(valid_configs), total)
        return valid_configs


class BorderlineConfigGenerator:
    """Generate configs from all combinations of the min and max of every parameter range.

    Each ``[min, max]`` range contributes two candidate values; the generator yields the
    Cartesian product of those pairs across all range parameters (2^N combinations for N
    range parameters).  Fixed (non-list) values are held constant throughout.
    """

    def __init__(self, cfg_type: Literal["abcd", "mabcd"]) -> None:
        """
        Initialise the borderline generator.

        :param cfg_type: Network type; only ``"abcd"`` is currently supported.
        """
        if cfg_type != "abcd":
            raise NotImplementedError(
                f"Borderline sampling is not yet supported for cfg_type={cfg_type!r}"
            )
        self.julia_config = ABCDConfig

    def generate(self, config_ranges: dict[str, Any]) -> list[Any]:
        """
        Return all valid configs at the corners of the parameter space.

        For every ``[min, max]`` pair in *config_ranges* both boundary values are used;
        the full Cartesian product across all range parameters is enumerated.
        ``denormalise_abcd_config`` is applied to each combination before Pydantic
        validation; invalid corners are silently dropped (one WARNING per point).

        :param config_ranges: Parameter ranges from the dataset YAML.

        :returns: List of validated config instances.
        """
        param_names: list[str] = []
        param_grids: list[list[Any]] = []

        for name, value in config_ranges.items():
            if isinstance(value, list) and len(value) == 2:
                lo, hi = value
                grid: list[Any] = [lo, hi]
            else:
                grid = [value]
            param_names.append(name)
            param_grids.append(grid)

        total = sum(1 for _ in itertools.product(*param_grids))
        valid_configs: list[Any] = []

        for combo in itertools.product(*param_grids):
            params = dict(zip(param_names, combo))
            params = denormalise_abcd_config(params)
            try:
                valid_configs.append(self.julia_config.from_yaml(params))
            except Exception as e:
                logger.warning("Dropping borderline point %s: %s", params, e)

        logger.info("Borderline: %d valid / %d attempted", len(valid_configs), total)
        return valid_configs

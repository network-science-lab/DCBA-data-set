"""Module to generate datasets based on specified configurations."""

import itertools
import json
import logging
import uuid
from typing import Any, Literal

import numpy as np
import yaml
from tqdm import tqdm

from dcba_data_set.julia_ports.abcd import ABCDConfig, ABCDGraphGenerator
from dcba_data_set.julia_ports.mabcd import mABCDConfig, mABCDGraphGenerator
from dcba_data_set.utils import create_out_dir

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

    def sample_from_range(self, smaller: float | int, bigger: float | int) -> float | int:
        """Sample a number from the given range using the specified distribution."""
        return self.rng.uniform(smaller, bigger)

    def _draw_config(self, config_ranges: dict[str, Any]) -> dict[str, Any]:
        sampled_config = {}
        for param_name, param_range in config_ranges.items():
            if isinstance(param_range, list) and len(param_range) == 2:
                param = self.sample_from_range(param_range[0], param_range[1])
                param = int(param) if isinstance(param_range[0], int) else round(param, 4)
                sampled_config[param_name] = param
            else:
                sampled_config[param_name] = param_range

        if self.julia_config is ABCDConfig:
            sampled_config = denormalise_abcd_config(sampled_config)

        return sampled_config

    def __call__(self, config_ranges: dict[str, Any]) -> Any:
        """Sample a valid config, retrying up to ``max_trials`` times.

        :raises RuntimeError: If no valid configuration is found within the trial limit.
        """
        for trial in range(1, self.max_trials + 1):
            sampled_config_dict = self._draw_config(config_ranges)
            try:
                return self.julia_config.from_yaml(sampled_config_dict)
            except Exception as e:
                logger.warning("Trial %d/%d failed: %s", trial, self.max_trials, e)
        raise RuntimeError(f"Exceeded {self.max_trials} trials without a valid configuration.")


class GridRandomConfigGenerator:
    """Generate all valid configs in the Cartesian product of a parameter grid."""

    def __init__(self, cfg_type: Literal["abcd", "mabcd"], grid_steps: int) -> None:
        """
        Initialise the grid generator.

        :param cfg_type: Network type; only ``"abcd"`` is currently supported.
        :param grid_steps: Number of evenly-spaced values per ``[min, max]`` range.
        """
        if cfg_type != "abcd":
            raise NotImplementedError(
                f"Grid sampling is not yet supported for cfg_type={cfg_type!r}"
            )
        self.julia_config = ABCDConfig
        self.grid_steps = grid_steps

    def generate(self, config_ranges: dict[str, Any]) -> list[Any]:
        """
        Return all valid configs in the Cartesian product of the parameter grid.

        For each ``[min, max]`` pair in *config_ranges*, ``grid_steps`` evenly-spaced
        values are produced via ``numpy.linspace``. Fixed (non-list) values are held
        constant. ``denormalise_abcd_config`` is applied to each combination before
        Pydantic validation; invalid points are silently dropped (one WARNING per point).

        :param config_ranges: Parameter ranges as used in the dataset YAML.

        :returns: List of validated ``ABCDConfig`` instances.
        """
        param_names: list[str] = []
        param_grids: list[list[Any]] = []

        for name, value in config_ranges.items():
            if isinstance(value, list) and len(value) == 2:
                lo, hi = value
                if isinstance(lo, int) and isinstance(hi, int):
                    raw = np.linspace(lo, hi, self.grid_steps)
                    grid: list[Any] = list(np.unique(raw.astype(int)))
                else:
                    grid = list(np.linspace(lo, hi, self.grid_steps))
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


class DatasetGenerator:
    """Generate a dataset of graphs by sampling configurations and running the graph generator."""

    def __call__(self, config: dict[str, Any]) -> Any:
        """Run the dataset generation loop and write outputs and a report to ``out_dir``."""
        net_ranges = config["net_ranges"]
        gen_cfg = config["generator"]
        net_type = gen_cfg["net_type"]
        out_dir = create_out_dir(gen_cfg["out_dir"])
        sampling = gen_cfg["sampling"]
        method = sampling["method"]
        net_replicas = sampling["replicas"]

        if net_type == "abcd":
            generator = ABCDGraphGenerator
        elif net_type == "mabcd":
            generator = mABCDGraphGenerator
        else:
            raise ValueError(f"Unknown network type: {net_type}")

        report = {"net_type": net_type, "instances": []}

        if method == "grid":
            gcg = GridRandomConfigGenerator(cfg_type=net_type, grid_steps=sampling["grid_steps"])
            configs = gcg.generate(net_ranges)
            self._run_instances(configs, generator, net_replicas, config, out_dir, report)
        elif method == "random":
            cg = RandomConfigGenerator(
                rng_seed=config["run"]["rng_seed"],
                cfg_type=net_type,
                max_trials=sampling["max_trials"],
            )
            self._run_random_instances(
                cg,
                net_ranges,
                sampling["instances"],
                generator,
                net_replicas,
                config,
                out_dir,
                report,
            )
        else:
            raise ValueError(f"Unknown sampling method: {method!r}")

        with open(out_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    def _run_random_instances(
        self,
        cg: RandomConfigGenerator,
        net_ranges: dict[str, Any],
        n_instances: int,
        generator: Any,
        net_replicas: int,
        config: dict[str, Any],
        out_dir: Any,
        report: dict[str, Any],
    ) -> None:
        """Sample configs one at a time, cleaning up failed attempts, then run each instance."""
        p_bar = tqdm(range(n_instances), desc="Instance", leave=False, colour="green")
        for instance in p_bar:
            instance_id = str(uuid.uuid4())[:8]
            instance_dir = out_dir / instance_id
            instance_dir.mkdir(parents=True, exist_ok=True)
            try:
                sampled_config = cg(net_ranges)
            except RuntimeError as e:
                logger.error("Skipping instance %d: %s", instance, e)
                instance_dir.rmdir()
                report["instances"].append(
                    {"id": instance_id, "status": "skipped", "error": str(e), "replicas": []}
                )
                continue
            self._run_instance(
                sampled_config, instance_id, instance_dir, generator, net_replicas, config, report
            )

    def _run_instances(
        self,
        configs: list[Any],
        generator: Any,
        net_replicas: int,
        config: dict[str, Any],
        out_dir: Any,
        report: dict[str, Any],
    ) -> None:
        """Iterate over a pre-built config list, create instance dirs and run graph generation."""
        p_bar = tqdm(configs, desc="Instance", leave=False, colour="green")
        for sampled_config in p_bar:
            instance_id = str(uuid.uuid4())[:8]
            instance_dir = out_dir / instance_id
            instance_dir.mkdir(parents=True, exist_ok=True)
            self._run_instance(
                sampled_config, instance_id, instance_dir, generator, net_replicas, config, report
            )

    def _run_instance(
        self,
        sampled_config: Any,
        instance_id: str,
        instance_dir: Any,
        generator: Any,
        net_replicas: int,
        config: dict[str, Any],
        report: dict[str, Any],
    ) -> None:
        """Write config to disk and generate all replicas for a single instance."""
        with open(instance_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(sampled_config.to_yaml(), f)

        instance_record = {
            "id": instance_id,
            "config": f"{instance_id}/config.yaml",
            "replicas": [],
        }

        for replica in range(1, net_replicas + 1):
            edges_rel = f"{instance_id}/edges_{replica}.dat"
            communities_rel = f"{instance_id}/communities_{replica}.dat"
            sampled_config.edges_filename = str(instance_dir / f"edges_{replica}.dat")
            sampled_config.communities_filename = str(instance_dir / f"communities_{replica}.dat")
            sampled_config.seed = config["run"]["rng_seed"]

            try:
                generator()(sampled_config)
                instance_record["replicas"].append(
                    {
                        "replica": replica,
                        "edges": edges_rel,
                        "communities": communities_rel,
                        "ok": True,
                    }
                )
            except Exception as e:
                logger.error("Instance %s replica %d failed: %s", instance_id, replica, e)
                instance_record["replicas"].append(
                    {
                        "replica": replica,
                        "edges": edges_rel,
                        "communities": communities_rel,
                        "ok": False,
                        "error": str(e),
                    }
                )

        report["instances"].append(instance_record)

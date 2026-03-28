"""A Python wrapper to the MLNABCDGraphGenerator Julia package."""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from juliacall import JuliaError
from juliacall import Main as jl

from src.julia_ports.base import BaseGraphConfig


@dataclass
class mABCDConfig(BaseGraphConfig):
    """
    A wrapper for jl.MLNABCDGraphGenerator.MLNConfig.

    Note that layer_params stores raw normalised values (0-1 range). Conversion
    to Julia's format is deferred to to_julia_csvs(), called during generation.
    """

    seed: int | None
    edges_cor: pd.DataFrame
    layer_params: pd.DataFrame
    d_max_iter: int
    c_max_iter: int
    t: int
    eps: float
    d: int
    edges_filename: str
    communities_filename: str

    # TODO: the fields below are the full MLNConfig interface exposed by the Julia package.
    # They are kept here for reference until a decision is made on whether to expose them.
    # l: int
    # qs: list[float]
    # ns: list[int]
    # taus: list[float]
    # rs: list[float]
    # gammas: list[float]
    # d_mins: list[int]
    # d_maxs: list[int]
    # betas: list[float]
    # c_mins: list[int]
    # c_maxs: list[int]
    # xis: list[float]
    # skip_edges_correlation: bool
    # edges_cor_matrix: np.ndarray

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(seed=self.seed)
        assert isinstance(self.n, int)
        assert isinstance(self.edges_cor, pd.DataFrame)
        assert isinstance(self.layer_params, pd.DataFrame)
        assert isinstance(self.d_max_iter, int)
        assert isinstance(self.c_max_iter, int)
        assert isinstance(self.t, int)
        assert isinstance(self.d, int)
        assert isinstance(self.eps, float)
        assert isinstance(self.edges_filename, str)
        assert isinstance(self.communities_filename, str)

    def _lp_yaml_helper(self) -> dict[str, Any]:
        return {
            col: list(self.layer_params[col])
            for col in self.layer_params.columns
        }

    def to_yaml(self) -> dict[str, Any]:
        """Convert configuration into a serialisable format."""
        return {
            "seed": self.seed,
            "n": self.n,
            "edges_cor": self.edges_cor.to_numpy().tolist(),
            "layer_params": self._lp_yaml_helper(),
            "d_max_iter": self.d_max_iter,
            "c_max_iter": self.c_max_iter,
            "t": self.t,
            "eps": self.eps,
            "d": self.d,
            "edges_filename": self.edges_filename,
            "communities_filename": self.communities_filename,
        }

    def to_julia_csvs(self, edges_cor_path: str, layer_params_path: str) -> None:
        """
        Save configuration in Julia format, normalising layer params from 0-1 to absolute values.

        :param edges_cor_path: path for the edge correlation matrix CSV
        :param layer_params_path: path for the layer parameters CSV
        """
        self.edges_cor.to_csv(edges_cor_path)

        lp_df = self.layer_params.copy()
        assert all(lp_df["q"].between(0, 1))
        assert all(lp_df["delta"].between(0, 1))
        assert all(lp_df["Delta"].between(0, 1))
        assert all(lp_df["s"].between(0, 1))
        assert all(lp_df["S"].between(0, 1))
        lp_df["_q"] = lp_df["q"] * self.n
        lp_df["delta"] = (lp_df["delta"] * lp_df["_q"]).round(0).astype(int)
        lp_df["Delta"] = (lp_df["Delta"] * lp_df["_q"]).round(0).astype(int)
        lp_df["s"] = (lp_df["s"] * lp_df["_q"]).round(0).astype(int)
        lp_df["S"] = (lp_df["S"] * lp_df["_q"]).round(0).astype(int)
        lp_df[["q", "tau", "r", "gamma", "delta", "Delta", "beta", "s", "S", "xi"]].to_csv(
            layer_params_path, index=False
        )

    @staticmethod
    def get_layer_params(lp: dict[str, Any] | str) -> pd.DataFrame:
        """Load layer parameters from a dict or CSV path (raw 0-1 values, no normalisation)."""
        if isinstance(lp, str):
            return pd.read_csv(lp)
        elif isinstance(lp, dict):
            return pd.DataFrame(lp)
        raise ValueError("LP should be either dict or path to file.")

    @staticmethod
    def get_edges_cor(ec: list[list[float]] | str) -> pd.DataFrame:
        """Load edge correlation matrix from a nested list or CSV path."""
        if isinstance(ec, str):
            return pd.read_csv(ec, index_col=0)
        elif isinstance(ec, list):
            return pd.DataFrame(
                ec,
                index=range(1, len(ec) + 1),
                columns=range(1, len(ec[0]) + 1),
            )
        raise ValueError("EC should be either list or path to file.")

    @classmethod
    def from_yaml(cls, config: dict[str, Any]) -> "mABCDConfig":
        """Read configuration from a YAML-like dictionary."""
        _config = config.copy()
        _config["edges_cor"] = cls.get_edges_cor(config["edges_cor"])
        _config["layer_params"] = cls.get_layer_params(config["layer_params"])
        return cls(**_config)


class mABCDGraphGenerator:
    """A wrapper class for jl.MLNABCDGraphGenerator."""

    edges_filename = "edges.csv"
    layers_filename = "layers.csv"

    @staticmethod
    def install_julia_dependencies() -> None:
        jl.Pkg.add(url="https://github.com/bkamins/ABCDGraphGenerator.jl")
        jl.Pkg.add(url="https://github.com/KrainskiL/MLNABCDGraphGenerator.jl")

    def __call__(self, config: mABCDConfig) -> None:
        try:
            jl.seval("using MLNABCDGraphGenerator")
        except JuliaError:
            self.install_julia_dependencies()
            jl.seval("using MLNABCDGraphGenerator")

        oldstdout = jl.stdout
        jl.redirect_stdout(jl.devnull)

        with tempfile.TemporaryDirectory() as tmpdir:

            # Save dataframes into temp dir in Julia-compatible format
            edges_path = str(Path(tmpdir) / self.edges_filename)
            layers_path = str(Path(tmpdir) / self.layers_filename)
            config.to_julia_csvs(edges_cor_path=edges_path, layer_params_path=layers_path)

            # Load config. Since Julia is called each time as a new process, we use a following
            # workaround to generate random, yet repetitive as a sequence, results
            jl_config = jl.MLNABCDGraphGenerator.MLNConfig(
                int(config._rng.random() * 1000),
                config.n,
                edges_path,
                layers_path,
                config.d_max_iter,
                config.c_max_iter,
                config.t,
                config.eps,
                config.d,
                config.edges_filename,
                config.communities_filename,
            )

            # Active nodes
            active_nodes = jl.MLNABCDGraphGenerator.generate_active_nodes(jl_config)

            # Degree sequences
            degrees = jl.MLNABCDGraphGenerator.generate_degrees(jl_config, active_nodes, False)

            # Sizes of communities
            com_sizes, coms = jl.MLNABCDGraphGenerator.generate_communities(jl_config, active_nodes)

            # Generate ABCD graphs
            edges = jl.MLNABCDGraphGenerator.generate_abcd(jl_config, degrees, com_sizes, coms)

            # Map nodes and communities into agents
            edges = jl.MLNABCDGraphGenerator.map_edges_to_agents(edges, active_nodes)
            coms = jl.MLNABCDGraphGenerator.map_communities_to_agents(jl_config.n, coms, active_nodes)

            # Adjust edges correlation
            edges_rewired = jl.MLNABCDGraphGenerator.adjust_edges_correlation(
                jl_config, edges, coms, active_nodes, False, False
            )

            # Save edges and communities to file
            jl.MLNABCDGraphGenerator.write_edges(jl_config, edges_rewired)
            jl.MLNABCDGraphGenerator.write_communities(jl_config, coms)

        jl.redirect_stdout(oldstdout)


if __name__ == "__main__":

    import yaml

    out_dir = Path("./examples/generate-mabcd")
    out_dir.mkdir(exist_ok=True, parents=True)

    # generate from code
    n = 1000
    layer_params = mABCDConfig.get_layer_params(
        lp={
            "q": [1, 0.75, 0.5, 0.25],
            "tau": [1, 0.75, 0.5, 0.25],
            "r": [1, 0.75, 0.5, 0.25],
            "gamma": [2.5, 2.5, 2.5, 2.5],
            "delta": [0.0020, 0.0027, 0.0040, 0.0080],
            "Delta": [0.0250, 0.0333, 0.0400, 0.0800],
            "beta": [1.5, 1.5, 1.7, 1.7],
            "s": [0.0080, 0.0107, 0.0160, 0.0320],
            "S": [0.0320, 0.0427, 0.0640, 0.1280],
            "xi": [0.2, 0.2, 0.2, 0.1],
        }
    )
    edges_cor = mABCDConfig.get_edges_cor(
        [
            [1.0, 0.15, 0.15, 0.12],
            [0.15, 1.0, 0.2, 0.1],
            [0.15, 0.2, 1.0, 0.2],
            [0.12, 0.1, 0.2, 1.0],
        ]
    )
    mabcd_config = mABCDConfig(
        seed=43,
        n=n,
        edges_cor=edges_cor,
        layer_params=layer_params,
        d_max_iter=1000,
        c_max_iter=1000,
        t=100,
        eps=0.05,
        d=2,
        edges_filename=str(out_dir / "_edges.dat"),
        communities_filename=str(out_dir / "_communities.dat"),
    )
    mABCDGraphGenerator()(config=mabcd_config)

    # or from file
    with open("scripts/configs/example_generate/mabcd.yaml", "r") as file:
        _config = yaml.safe_load(file)
    config = _config["net_config"]
    config["seed"] = _config["run"]["rng_seed"]
    config["edges_filename"] = str(out_dir / config["edges_filename"])
    config["communities_filename"] = str(out_dir / config["communities_filename"])
    mabcd_config = mABCDConfig.from_yaml(config)
    mABCDGraphGenerator()(config=mabcd_config)

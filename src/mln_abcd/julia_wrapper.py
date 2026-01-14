"""A Python wrapper to the MLNABCDGraphGenerator Julia package."""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from juliacall import JuliaError
from juliacall import Main as jl


@dataclass
class MLNConfig:
    """
    A wrapper for class for class for jl.MLNABCDGraphGenerator.MLNConfig.

    TODO: we can get rid of storing a part of the config in files (see commented out code and:
    https://github.com/KrainskiL/MLNABCDGraphGenerator.jl/blob/main/src/auxiliary.jl#L19)
    """
    seed: int
    n: int
    edges_cor: pd.DataFrame
    layer_params: pd.DataFrame
    d_max_iter: int
    c_max_iter: int
    t: int
    eps: float
    d: int
    edges_filename: str
    communities_filename: str
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
        assert isinstance(self.seed, int)
        assert isinstance(self.n, int)
        assert isinstance(self.edges_cor, pd.DataFrame)
        assert isinstance(self.layer_params, pd.DataFrame)
        assert isinstance(self.d_max_iter, int)
        assert isinstance(self.c_max_iter, int)
        assert isinstance(self.t, int)
        assert isinstance(self.d, int)
        assert isinstance(self.eps, float)
        assert isinstance(self.d, int)
        assert isinstance(self.edges_filename, str)
        assert isinstance(self.communities_filename, str)

    @staticmethod
    def get_layer_params(n: int, lp: dict[str, Any] | str) -> pd.DataFrame:
        if isinstance(lp, str):
            df = pd.read_csv(lp)
        elif isinstance(lp, dict):
            df = pd.DataFrame(lp)
        else:
            raise ValueError(f"LP should be either dict or path to file.")
        assert all(df["q"].between(0, 1))
        assert all(df["delta"].between(0, 1))
        assert all(df["Delta"].between(0, 1))
        assert all(df["s"].between(0, 1))
        assert all(df["S"].between(0, 1))
        df["_q"] = df["q"] * n
        df["delta"] = (df["delta"] * df["_q"]).round(0).astype(int)
        df["Delta"] = (df["Delta"] * df["_q"]).round(0).astype(int)
        df["s"] = (df["s"] * df["_q"]).round(0).astype(int)
        df["S"] = (df["S"] * df["_q"]).round(0).astype(int)
        return df[["q", "tau", "r", "gamma", "delta", "Delta", "beta", "s", "S", "xi"]]

    @staticmethod
    def get_edges_cor(ec: list[list[float]] | str) -> pd.DataFrame:
        if isinstance(ec, str):
            return pd.read_csv(ec, index_col=0)
        elif isinstance(ec, list):
            return pd.DataFrame(
                ec,
                index=range(1, len(ec) + 1),
                columns=range(1, len(ec[0]) + 1),
            )
        raise ValueError(f"EC should be either list or path to file.")

    @classmethod
    def from_yaml(cls, config: dict[str, Any]) -> "MLNConfig":
        _config = config.copy()
        edges_cor = cls.get_edges_cor(config["edges_cor"])
        _config["edges_cor"] = edges_cor
        layer_params = cls.get_layer_params(config["n"], config["layer_params"])
        _config["layer_params"] = layer_params
        return cls(**_config)


class MLNABCDGraphGenerator:
    """A wrapper class for jl.MLNABCDGraphGenerator."""

    edges_filename = "edges.csv"
    layers_filename = "layers.csv"

    @staticmethod
    def install_julia_dependencies():
        jl.Pkg.add(url="https://github.com/bkamins/ABCDGraphGenerator.jl")
        jl.Pkg.add(url="https://github.com/KrainskiL/MLNABCDGraphGenerator.jl")

    def __call__(self, config: MLNConfig) -> None:
        try:
            jl.seval("using MLNABCDGraphGenerator")
        except JuliaError:
            self.install_julia_dependencies()
        
        with tempfile.TemporaryDirectory() as tmpdir:

            # Save dataframes into temp dir
            edges_path = str(Path(tmpdir) / self.edges_filename)
            layers_path = str(Path(tmpdir) / self.layers_filename)
            config.edges_cor.to_csv(edges_path)
            config.layer_params.to_csv(layers_path, index=False)

            # Load config. Since julia is called each time as a new process, we use a following
            # workaround to generate random, yet repetitive as a sequence, results
            config = jl.MLNABCDGraphGenerator.MLNConfig(
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
            active_nodes = jl.MLNABCDGraphGenerator.generate_active_nodes(config)

            # Degree Sequences
            degrees = jl.MLNABCDGraphGenerator.generate_degrees(config, active_nodes, False)

            # Sizes of communities
            com_sizes, coms = jl.MLNABCDGraphGenerator.generate_communities(config, active_nodes)

            # Generate ABCD graphs
            edges = jl.MLNABCDGraphGenerator.generate_abcd(config, degrees, com_sizes, coms)

            # Map nodes and communities into agents
            edges = jl.MLNABCDGraphGenerator.map_edges_to_agents(edges, active_nodes)
            coms = jl.MLNABCDGraphGenerator.map_communities_to_agents(config.n, coms, active_nodes)

            # Adjust edges correlation
            edges_rewired = jl.MLNABCDGraphGenerator.adjust_edges_correlation(
                config, edges, coms, active_nodes, False, False
            )

            # Save edges to file
            jl.MLNABCDGraphGenerator.write_edges(config, edges_rewired)

            # Save communities to file
            jl.MLNABCDGraphGenerator.write_communities(config, coms)


if __name__ == "__main__":

    import yaml
    from pathlib import Path
    
    out_dir = Path("./examples/generate")
    out_dir.mkdir(exist_ok=True, parents=True)

    # generate from code
    layer_params = MLNConfig.get_layer_params(1000, "scripts/configs/example_generate/layers.csv")
    edges_cor = MLNConfig.get_edges_cor("scripts/configs/example_generate/edges.csv")
    mln_config = MLNConfig(
        seed=43,
        n=1000,
        edges_cor=edges_cor,
        layer_params=layer_params,
        d_max_iter=1000,
        c_max_iter=1000,
        t=100,
        eps=0.05,
        d=2,
        edges_filename=str(out_dir / "edges.dat"),
        communities_filename=str(out_dir / "communities.dat"),
    )
    MLNABCDGraphGenerator()(config=mln_config)

    # or from file
    with open("scripts/configs/example_generate/config.yaml", "r") as file:
        _config = yaml.safe_load(file)
    config = _config["mln_config"]
    config["seed"] = _config["run"]["rng_seed"]
    mln_config = MLNConfig.from_yaml(config)
    MLNABCDGraphGenerator()(config=mln_config)

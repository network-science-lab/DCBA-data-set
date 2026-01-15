"""Python wrapper for ABCDGraphGenerator.jl Julia package."""

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from juliacall import JuliaError
from juliacall import Main as jl


@dataclass
class ABCDConfig:
    """Configuration for ABCD graph generation."""
    n: int
    t1: float
    d_min: int
    d_max: int
    d_max_iter: int
    t2: float
    c_min: int
    c_max: int
    c_max_iter: int
    xi: Optional[float] = None
    mu: Optional[float] = None
    islocal: bool = False
    isCL: Optional[bool] = False
    seed: Optional[int] = None
    nout: int = 0
    edges_filename: Optional[str] = None
    communities_filename: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the configuration parameters."""
        if self.mu is not None and self.xi is not None:
            raise ValueError("inconsistent data: only μ or ξ may be provided")
        if self.mu is not None and self.nout > 0:
            raise ValueError("μ is not supported with outliers")
        if self.nout > self.n:
            raise ValueError("number of outliers cannot be larger than graph size")
        if self.islocal and self.nout > 0:
            raise ValueError("local graph is not supported with outliers")
        if self.isCL and self.nout > 0:
            raise ValueError("Chung-Lu graph is not supported with outliers")
        if self.c_max > self.n:
            raise ValueError("c_max cannot be larger than graph size")

    @classmethod
    def from_yaml(cls, cfg: Dict[str, Any]) -> "ABCDConfig":
        """Create an instance from a dictionary."""
        return cls(**cfg)


class ABCDGraphGenerator:
    """Wrapper for the ABCDGraphGenerator Julia package."""

    @staticmethod
    def install_julia_dependencies() -> None:
        """Install required Julia packages."""
        jl.Pkg.add(url="https://github.com/bkamins/ABCDGraphGenerator.jl")

    def __init__(self) -> None:
        """Initialise the generator and load Julia packages."""
        try:
            jl.seval("using ABCDGraphGenerator")
            jl.seval("using Random")
        except JuliaError:
            self.install_julia_dependencies()
            jl.seval("using ABCDGraphGenerator")

    def _run(self, c: ABCDConfig) -> Dict[str, Any]:
        """Execute the graph generation process."""
        if c.seed is not None:
            jl.Random.seed_b(c.seed)

        # prepare auxiliary structures
        n_eff = c.n - c.nout
        degs = jl.ABCDGraphGenerator.sample_degrees(
            c.t1,
            c.d_min,
            c.d_max,
            n_eff + c.nout,
            c.d_max_iter,
        )
        coms = jl.ABCDGraphGenerator.sample_communities(
            c.t2,
            c.c_min,
            c.c_max,
            n_eff,
            c.c_max_iter,
        )
        if c.nout > 0:
            jl.pushfirst_b(coms, c.nout)
        params = jl.ABCDGraphGenerator.ABCDParams(
            degs,
            coms,
            c.mu,
            c.xi,
            c.isCL,
            c.islocal,
            c.nout > 0,
        )

        # generate the graph
        edges, clusters = jl.ABCDGraphGenerator.gen_graph(params)

        # convert to Python-friendly formats
        edges = np.array(sorted((int(a), int(b)) for a, b in edges))
        edges_df = pd.DataFrame(edges, columns=["u", "v"])
        clusters = np.array(clusters, dtype=int)
        communities_df = pd.DataFrame(
            {
                "node": np.arange(1, len(clusters) + 1),
                "community": clusters,
            }
        )

        return {
            "degrees": np.array(degs, dtype=int),
            "community_sizes": np.array(coms, dtype=int),
            "edges": edges_df,
            "communities": communities_df,
        }

    def __call__(self, config: ABCDConfig) -> Dict[str, Any]:
        """Generate a graph based on the provided configuration."""
        result = self._run(config)
        result["edges"].to_csv(config.edges_filename, index=False)
        result["communities"].to_csv(config.communities_filename, index=False)
        return result


if __name__ == "__main__":
    import yaml
    from pathlib import Path

    out_dir = Path("./examples/generate-abcd")
    out_dir.mkdir(exist_ok=True, parents=True)

    gen = ABCDGraphGenerator()

    config_inline = ABCDConfig(
        seed=42,
        n=1000,
        t1=2.5,
        d_min=10,
        d_max=100,
        d_max_iter=100,
        t2=1.5,
        c_min=20,
        c_max=50,
        c_max_iter=100,
        xi=0.2,
        islocal=False,
        nout=0,
        edges_filename=str(out_dir / "edges.dat"),
        communities_filename=str(out_dir / "communities.dat"),
    )

    result = gen(config_inline)

    with open("scripts/configs/example_generate/abcd.yaml") as f:
        _config = yaml.safe_load(f)
    config = _config["net_config"]
    config["seed"] = _config["run"]["rng_seed"]
    net_config = ABCDConfig.from_yaml(config)
    result = gen(net_config)

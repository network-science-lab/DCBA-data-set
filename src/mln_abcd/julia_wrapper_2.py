"""Python wrapper for ABCDGraphGenerator.jl Julia package."""

from dataclasses import dataclass
from pathlib import Path
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

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "ABCDConfig":
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

    @staticmethod
    def _validate(c: ABCDConfig) -> None:
        """Validate the configuration parameters."""
        if c.mu is not None and c.xi is not None:
            raise ValueError("inconsistent data: only μ or ξ may be provided")
        if c.mu is not None and c.nout > 0:
            raise ValueError("μ is not supported with outliers")
        if c.nout > c.n:
            raise ValueError("number of outliers cannot be larger than graph size")
        if c.islocal and c.nout > 0:
            raise ValueError("local graph is not supported with outliers")
        if c.isCL and c.nout > 0:
            raise ValueError("Chung-Lu graph is not supported with outliers")

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
            jl.pushfirst_(coms, c.nout)
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
        self._validate(config)
        return self._run(config)


if __name__ == "__main__":
    gen = ABCDGraphGenerator()
    config = ABCDConfig(
        n=100,
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
        seed=42,
        nout=0,
    )
    result = gen(config)
    print(result)
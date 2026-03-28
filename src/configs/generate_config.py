"""Configuration generator for synthetic graph networks."""

from dataclasses import dataclass, asdict

import yaml


@dataclass
class LayerParams:
    """Parameters for individual network layers."""
    q: list[float]  # fraction of active actors
    tau: list[float]  # correlation between degrees and labels
    r: list[float]  # correlation with reference layer
    gamma: list[float]  # power-law degree exponent
    delta: list[float]  # min degree
    Delta: list[float]  # max degree
    beta: list[float]  # power-law community size exponent
    s: list[float]  # min community size
    S: list[float]  # max community size
    xi: list[float]  # noise level


@dataclass
class MLNConfig:
    """Configuration for multilayer network generation."""
    n: int
    edges_cor: list[list[float]]
    layer_params: LayerParams
    d_max_iter: int = 1000
    c_max_iter: int = 1000
    t: int = 100
    eps: float = 0.05
    d: int = 2
    edges_filename: str = "edges.dat"
    communities_filename: str = "communities.dat"


@dataclass
class RunConfig:
    """Run configuration."""
    experiment_type: str = "generate"
    rng_seed: int = 43


@dataclass
class GeneratorConfig:
    """Generator configuration."""
    repetitions: int = 5
    out_dir: str = "./examples/generate"


@dataclass
class Config:
    """Complete configuration for network generation."""
    run: RunConfig
    mln_config: MLNConfig
    generator: GeneratorConfig

    def to_dict(self):
        """Convert to dictionary for YAML serialization."""
        return {
            'run': asdict(self.run),
            'mln_config': {
                **{k: v for k, v in asdict(self.mln_config).items() if k != 'layer_params'},
                'layer_params': asdict(self.mln_config.layer_params)
            },
            'generator': asdict(self.generator)
        }

    def save(self, filepath: str) -> None:
        """Save configuration to YAML file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def create_config(
    n_actors: int = 1000,
    n_layers: int = 4,
    rng_seed: int = 43,
    out_dir: str = "./examples/generate",
    repetitions: int = 5,
    **layer_kwargs
) -> Config:
    """
    Create a configuration for network generation.
    
    Args:
        n_actors: Number of actors in the network
        n_layers: Number of layers
        rng_seed: Random seed for reproducibility
        out_dir: Output directory for generated networks
        repetitions: Number of networks to generate
        **layer_kwargs: Additional layer parameters
    
    Returns:
        Config object ready to be saved
    """
    # Create default layer parameters
    layer_params = LayerParams(
        q=[1.0] + [0.75 - 0.25*i for i in range(n_layers-1)],
        tau=[1.0] + [0.75 - 0.25*i for i in range(n_layers-1)],
        r=[1.0] + [0.75 - 0.25*i for i in range(n_layers-1)],
        gamma=[2.5] * n_layers,
        delta=[0.002 * (i+1) for i in range(n_layers)],
        Delta=[0.025 * (i+1) for i in range(n_layers)],
        beta=[1.5] * (n_layers//2) + [1.7] * (n_layers - n_layers//2),
        s=[0.008 * (i+1) for i in range(n_layers)],
        S=[0.032 * (i+1) for i in range(n_layers)],
        xi=[0.2] * (n_layers - 1) + [0.1]
    )
    
    # Create correlation matrix
    edges_cor = [[1.0 if i == j else 0.15 for j in range(n_layers)] for i in range(n_layers)]
    
    mln_config = MLNConfig(
        n=n_actors,
        edges_cor=edges_cor,
        layer_params=layer_params
    )
    
    return Config(
        run=RunConfig(rng_seed=rng_seed),
        mln_config=mln_config,
        generator=GeneratorConfig(repetitions=repetitions, out_dir=out_dir)
    )

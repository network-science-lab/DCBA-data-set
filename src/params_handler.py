"""A script with functions to facilitate liading simulation's parameters and input data."""

# TODO: review this code and decidee whether it's needed

import itertools
import json
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import network_diffusion as nd

from src.loaders.net_loader import load_network
from src.loaders.constants import SEPARATOR


@dataclass(frozen=True)
class Network:
    n_type: str
    n_name: str
    n_graph_pt: nd.MultilayerNetworkTorch
    n_graph_nx: nd.MultilayerNetwork

    @property
    def rich_name(self) -> str:
        _type = self.n_type.replace("/", ".")
        _name = self.n_name.replace("/", ".")
        if _type == _name:
            return _type
        return f"{_type}{SEPARATOR}{_name}"


def create_out_dir(out_dir: str) -> Path:
    try:
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(exist_ok=True, parents=True)
    except FileExistsError:
        print("Redirecting output to hell...")
        out_dir_path = Path(tempfile.mkdtemp())
    return out_dir_path


def load_networks(networks: list[str], device: str) -> list[Network]:
    nets = []
    for net_regex in networks:
        net_type, net_name = net_regex.split(SEPARATOR)
        print(f"Loading network(s): {net_type} - {net_name}")
        for (net_type, net_name), net_graph in load_network(net_type=net_type, net_name=net_name).items():
            print("\tconverting to PyTorch")
            nets.append(
                Network(
                    n_type=net_type,
                    n_name=net_name,
                    n_graph_nx=net_graph,
                    n_graph_pt=nd.MultilayerNetworkTorch.from_mln(net_graph, device)
                )
            )
    print(f"Loaded {len(nets)} networks")
    return nets

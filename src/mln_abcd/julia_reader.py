"""Functions to read networks according to MLNABCD formatting."""

from pathlib import Path

import networkx as nx
import network_diffusion as nd
import pandas as pd


def load_edgelist(edgelist_path: Path) -> nd.MultilayerNetwork:
    edge_list = pd.read_csv(edgelist_path, sep="\t", names=["source", "target", "layer"])
    layer_names = edge_list["layer"].unique()
    layer_graphs = {}
    for layer_name in layer_names:
        el_layer = edge_list.loc[edge_list["layer"] == layer_name]
        layer_graphs[str(layer_name)] = nx.from_pandas_edgelist(el_layer)
    return nd.MultilayerNetwork(layers=layer_graphs)


def load_communities(communities_path: Path) -> pd.DataFrame:
    communities = pd.read_csv(communities_path, sep="\t", names=["community", "layer"])
    communities.index.name = "actor"
    return communities

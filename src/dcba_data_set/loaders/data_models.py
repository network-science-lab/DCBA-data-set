"""Data models for DCBA graph + config pairs."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bidict import bidict
from torch_geometric.data import HeteroData


@dataclass
class ConfigRecord:
    """Config side of a (graph, config) pair.

    :param instance_id: Identifier linking this config to its DCBAHeteroData replicas.
    :param data: Raw YAML dict of the generation parameters (e.g. ABCDConfig fields).
    :param path: Path to the config.yaml from which data was loaded.
    """

    instance_id: str
    data: dict
    path: Path


class DCBAHeteroData(HeteroData):
    """Graph side of a (graph, config) pair.

    Node store ``"actor"`` carries:

    - ``community``: int tensor ``[num_actors, num_layers]`` of raw community IDs.
      Inactive nodes (mABCD only) carry ``community = 0``.

    Edge store ``("actor", "l_<i>", "actor")`` carries one COO ``edge_index`` per layer.
    ABCD graphs expose a single layer ``l_0``; mABCD graphs expose one relation per layer.

    Metadata attributes: ``instance_id`` (links to :class:`ConfigRecord`), ``replica``
    (index within the instance), ``actors_map`` (bidict str node_id → tensor index),
    ``layers_map`` (bidict original layer name → ``"l_<i>"``).
    """

    @classmethod
    def from_abcd_files(
        cls,
        edges_path: Path,
        communities_path: Path,
        instance_id: str,
        replica: int,
    ) -> "DCBAHeteroData":
        """Build a DCBAHeteroData from a single-layer ABCD edge and community file.

        :param edges_path: Path to the edges CSV (with header) with columns
            ``u``, ``v`` (1-indexed node IDs).
        :param communities_path: Path to the communities CSV (with header) with
            columns ``node``, ``community`` (1-indexed).
        :param instance_id: Identifier linking this graph to its ConfigRecord.
        :param replica: Replica index within the instance.
        :returns: A populated DCBAHeteroData instance with a single layer ``l_0``.
        """
        data = cls()

        layers_map: bidict[str, str] = bidict({"0": "l_0"})

        # Build actors_map from the community file so isolated nodes are included.
        comm_df = pd.read_csv(communities_path)
        sorted_node_ids = sorted(comm_df["node"].astype(str).tolist(), key=lambda x: int(x))
        actors_map: bidict[str, int] = bidict(
            {node_id: idx for idx, node_id in enumerate(sorted_node_ids)}
        )

        edges_df = pd.read_csv(edges_path)
        src = edges_df["u"].astype(str).map(actors_map.get).values
        dst = edges_df["v"].astype(str).map(actors_map.get).values
        edge_index = torch.from_numpy(np.array([src, dst])).long()
        # Files list each undirected edge once (u < v); add both directions.
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        data["actor", "l_0", "actor"].edge_index = edge_index

        # Community tensor [num_actors, 1] — raw IDs from file.
        num_actors = len(actors_map)
        community = torch.zeros(num_actors, 1, dtype=torch.long)
        actor_indices = comm_df["node"].astype(str).map(actors_map.get).values
        community[actor_indices, 0] = torch.tensor(comm_df["community"].values, dtype=torch.long)
        data["actor"].community = community

        data.instance_id = instance_id
        data.replica = replica
        data.actors_map = actors_map
        data.layers_map = layers_map

        return data

    @classmethod
    def from_mabcd_files(
        cls,
        edges_path: Path,
        communities_path: Path,
        instance_id: str,
        replica: int,
    ) -> "DCBAHeteroData":
        """Build a DCBAHeteroData from mABCD multilayer edge and community files.

        :param edges_path: Path to the edges TSV (no header) with tab-separated columns
            ``node_a``, ``node_b``, ``layer_idx`` (all 1-indexed).
        :param communities_path: Path to the communities TSV (no header) with tab-separated
            columns ``community_id``, ``layer_idx``.  Node IDs are implied by row position within
            each layer block (first row of a block = node 1).  Inactive nodes carry
            ``community = 0``.
        :param instance_id: Identifier linking this graph to its ConfigRecord.
        :param replica: Replica index within the instance.
        :returns: A populated DCBAHeteroData instance.
        """
        data = cls()

        edges_df = pd.read_csv(edges_path, sep="\t", header=None, names=["u", "v", "layer"])
        layer_indices = sorted(edges_df["layer"].unique())

        layers_map: bidict[str, str] = bidict(
            {str(l_idx): f"l_{i}" for i, l_idx in enumerate(layer_indices)}
        )

        comm_df = pd.read_csv(communities_path, sep="\t", header=None, names=["community", "layer"])

        # Each layer block has exactly n rows (one per actor, positional 1-indexed).
        # Use the first layer block to determine the actor count.
        first_layer_comm = comm_df[comm_df["layer"] == layer_indices[0]]
        num_actors = len(first_layer_comm)

        sorted_node_ids = [str(i) for i in range(1, num_actors + 1)]
        actors_map: bidict[str, int] = bidict(
            {node_id: idx for idx, node_id in enumerate(sorted_node_ids)}
        )

        for i, l_idx in enumerate(layer_indices):
            layer_edges = edges_df[edges_df["layer"] == l_idx]
            src = layer_edges["u"].astype(str).map(actors_map.get).values
            dst = layer_edges["v"].astype(str).map(actors_map.get).values
            edge_index = torch.from_numpy(np.array([src, dst])).long()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            data["actor", f"l_{i}", "actor"].edge_index = edge_index

        num_layers = len(layer_indices)
        community = torch.zeros(num_actors, num_layers, dtype=torch.long)
        for i, l_idx in enumerate(layer_indices):
            layer_comm = comm_df[comm_df["layer"] == l_idx].reset_index(drop=True)
            community[:, i] = torch.tensor(layer_comm["community"].values, dtype=torch.long)

        data["actor"].community = community
        data.instance_id = instance_id
        data.replica = replica
        data.actors_map = actors_map
        data.layers_map = layers_map

        return data

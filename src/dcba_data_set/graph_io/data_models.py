"""Data models for DCBA graph + config pairs."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from bidict import bidict
from torch_geometric.data import HeteroData


@dataclass
class ReplicaRecord:
    """
    Paths for a single replica. No I/O is performed at construction time.

    :param replica: Replica index within the parent instance.
    :param edges_path: Path to the edges file for this replica.
    :param communities_path: Path to the communities file for this replica.
    """

    replica: int
    edges_path: Path
    communities_path: Path


@dataclass
class InstanceRecord:
    """
    Paths for one instance (one config → many replicas).

    :param instance_id: Identifier shared with :class:`DCBAInstanceConfig` and
        :class:`DCBAHeteroData`.
    :param net_type: Network type: ``"abcd"`` or ``"mabcd"``.
    :param config_path: Path to the config YAML file.
    :param replicas: Ordered list of :class:`ReplicaRecord` objects for this instance.
    """

    instance_id: str
    net_type: str
    config_path: Path
    replicas: list[ReplicaRecord]


@dataclass
class DCBAInstanceConfig:
    """
    Config side of a (graph, config) pair.

    :param instance_id: Identifier linking this config to its DCBAHeteroData replicas.
    :param data: Raw YAML dict of the generation parameters (e.g. ABCDConfig fields).
    :param path: Path to the config.yaml from which data was loaded.
    """

    instance_id: str
    data: dict
    path: Path

    @classmethod
    def from_instance_record(cls, record: "InstanceRecord") -> "DCBAInstanceConfig":
        """
        Load a DCBAInstanceConfig by reading the YAML file referenced by an InstanceRecord.

        :param record: InstanceRecord whose config_path will be read.

        :returns: A populated DCBAInstanceConfig.
        """
        with record.config_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(instance_id=record.instance_id, data=data, path=record.config_path)


class DCBAHeteroData(HeteroData):
    """
    Class to represent a graphs (both mABCD and ABCD) for the DCBA dataset.

    Node store ``"actor"`` carries:
        - ``community``: int tensor ``[num_actors, num_layers]`` of raw community IDs. Inactive
            nodes (mABCD only) carry ``community = 0``.

    Edge store ``("actor", "l_<i>", "actor")`` carries:
        - one COO ``edge_index`` per layer. ABCD graphs expose a single layer ``l_0``; mABCD graphs
            expose one relation per layer.

    Metadata attributes:
        - ``instance_id`` (links to :class:`DCBAInstanceConfig`),
        - ``replica`` (index within the instance; stacks to ``[B]`` tensor after batching),
        - ``actors_map`` (bidict str node_id -> tensor index; list of bidicts after batching),
        - ``layers_map`` (bidict original layer name -> ``"l_<i>"``;list of bidicts after batching).
    """

    _NATIVE_ATTRS = frozenset({"actors_map", "layers_map"})

    def __setattr__(self, key: str, value: object) -> None:
        # Wrap bidict in a 1-tuple so PyG's collation sees a Sequence whose first element
        # is not a Tensor — falls through to the else branch and is collected as a list.
        if key in self._NATIVE_ATTRS:
            super().__setattr__(key, (value,))
        else:
            super().__setattr__(key, value)

    def __getattr__(self, key: str) -> object:
        if key in self._NATIVE_ATTRS:
            try:
                raw = super().__getattr__(key)
            except (AttributeError, KeyError):
                return None
            if isinstance(raw, list):  # batch: [(bd1,), (bd2,)] -> [bd1, bd2]
                return [t[0] for t in raw]
            if isinstance(raw, tuple):  # single graph: (bd,) -> bd
                return raw[0]
            return None
        return super().__getattr__(key)

    @classmethod
    def from_replica_record(
        cls, record: ReplicaRecord, instance_id: str, net_type: str
    ) -> "DCBAHeteroData":
        """
        Build a DCBAHeteroData by dispatching to the correct factory based on net_type.

        :param record: ReplicaRecord supplying the file paths and replica index.
        :param instance_id: Identifier linking this graph to its DCBAInstanceConfig.
        :param net_type: Network type: ``"abcd"``, ``"mabcd"``, ``"single_layer"``,
        ``"multi_layer"``.

        :returns: A populated DCBAHeteroData instance.
        """
        _factory = {
            "abcd": cls.from_abcd_files,
            "mabcd": cls.from_mabcd_files,
            "single_layer": cls.from_abcd_files,  # For real-world graphs with single layer format
            "multi_layer": cls.from_mabcd_files,  # For real-world graphs with multi layer format
        }
        if net_type not in _factory:
            raise ValueError(f"Unknown net_type {net_type!r}.")
        return _factory[net_type](
            edges_path=record.edges_path,
            communities_path=record.communities_path,
            instance_id=instance_id,
            replica=record.replica,
        )

    @classmethod
    def from_abcd_files(
        cls,
        edges_path: Path,
        communities_path: Path,
        instance_id: str,
        replica: int,
    ) -> "DCBAHeteroData":
        """
        Build a DCBAHeteroData from a single-layer ABCD edge and community file.

        :param edges_path: Path to the edges CSV (with header) with columns ``u``, ``v``
            (1-indexed node IDs).
        :param communities_path: Path to the communities CSV (with header) with columns ``node``,
            ``community`` (1-indexed).
        :param instance_id: Identifier linking this graph to its DCBAInstanceConfig.
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
        data.replica = torch.tensor([replica], dtype=torch.long)
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
        """
        Build a DCBAHeteroData from mABCD multilayer edge and community files.

        :param edges_path: Path to the edges TSV (no header) with tab-separated columns ``node_a``,
            ``node_b``, ``layer_idx`` (all 1-indexed).
        :param communities_path: Path to the communities TSV (no header) with tab-separated
            columns ``community_id``, ``layer_idx``.  Node IDs are implied by row position within
            each layer block (1st row of a block = node 1). Inactive nodes carry ``community = 0``.
        :param instance_id: Identifier linking this graph to its DCBAInstanceConfig.
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
        data.replica = torch.tensor([replica], dtype=torch.long)
        data.actors_map = actors_map
        data.layers_map = layers_map

        return data

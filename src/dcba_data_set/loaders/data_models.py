"""Data models for DCBA graph + config pairs."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bidict import bidict
from torch_geometric.data import HeteroData
from typing_extensions import Self


@dataclass
class ConfigRecord:
    """Config side of a (graph, config) pair.

    Attributes:
        instance_id: Identifier linking this config to its DCBAHeteroData replicas.
        data: Raw YAML dict of the generation parameters (e.g. ABCDConfig fields).
        path: Path to the config.yaml from which data was loaded.
    """

    instance_id: str
    data: dict
    path: Path


class DCBAHeteroData(HeteroData):
    """Graph side of a (graph, config) pair.

    Follows the same heterogeneous layout as MLNHeteroData in infmax-trainer so that
    the same GNN encoder can be reused for both datasets.

    Attributes:
        self["actor"].community: int tensor [num_actors, num_layers] — community ID per
            actor per layer.  Currently stores raw IDs from the communities file.

            Future options to explore:
            - Option A (partial invariance): canonically reindex IDs per layer by
              descending community size, so ID 0 is always the largest community.
            - Option B (full invariance): replace IDs with per-node statistics
              (community_size, intra_degree) per layer → shape [num_actors, 2, num_layers].
              Fully invariant by construction; directly mirrors ABCD parameters like xi/mu.

        self["actor", "l_<i>", "actor"].edge_index: COO edge index for layer i.
            ABCD produces one layer ("l_0"); mABCD produces N layers with the same layout.

        self.instance_id: Links this graph to its ConfigRecord.
        self.replica: Which replica this is (same config → many replicas, each a distinct graph).
        self.actors_map: bidict str(node_id) → tensor index (0-indexed).
        self.layers_map: bidict layer_name → "l_<i>".
    """

    @classmethod
    def from_files(
        cls,
        edges_paths: list[tuple[Path, str]],
        communities_paths: list[tuple[Path, str]],
        instance_id: str,
        replica: int,
    ) -> Self:
        """Build a DCBAHeteroData from edge and community files.

        Args:
            edges_paths: List of (path, layer_name) pairs — one per layer.
                Each file is a CSV with columns ``u,v`` (1-indexed node IDs).
                ABCD passes a single-element list; mABCD passes one per layer.
            communities_paths: List of (path, layer_name) pairs — one per layer,
                in the same order as edges_paths.
                Each file is a CSV with columns ``node,community`` (1-indexed).
            instance_id: Identifier linking this graph to its ConfigRecord.
            replica: Replica index within the instance.
        """
        data = cls()

        layers_map = bidict(
            {layer_name: f"l_{l_idx}" for l_idx, (_, layer_name) in enumerate(edges_paths)}
        )

        # Build actors_map from the union of all nodes across community files.
        # Community files are preferred over edge files because isolated nodes (present
        # in a community but with no edges) would be missed if we only scanned edges.
        all_node_ids: set[str] = set()
        for comm_path, _ in communities_paths:
            df = pd.read_csv(comm_path)
            all_node_ids.update(df["node"].astype(str).tolist())
        sorted_node_ids = sorted(all_node_ids, key=lambda x: int(x))
        actors_map: bidict[str, int] = bidict(
            {node_id: idx for idx, node_id in enumerate(sorted_node_ids)}
        )

        # Edge indices — one relation per layer.
        for l_idx, (edges_path, _) in enumerate(edges_paths):
            df = pd.read_csv(edges_path)
            src = df["u"].astype(str).map(actors_map.get).values
            dst = df["v"].astype(str).map(actors_map.get).values
            edge_index = torch.from_numpy(np.array([src, dst])).long()
            # Files list each undirected edge once (u < v); add both directions.
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            data["actor", f"l_{l_idx}", "actor"].edge_index = edge_index

        # Community tensor [num_actors, num_layers] — raw IDs from file.
        num_actors = len(actors_map)
        num_layers = len(communities_paths)
        community = torch.zeros(num_actors, num_layers, dtype=torch.long)
        for l_idx, (comm_path, _) in enumerate(communities_paths):
            df = pd.read_csv(comm_path)
            actor_indices = df["node"].astype(str).map(actors_map.get).values
            community[actor_indices, l_idx] = torch.tensor(
                df["community"].values, dtype=torch.long
            )
        data["actor"].community = community

        data.instance_id = instance_id
        data.replica = replica
        data.actors_map = actors_map
        data.layers_map = layers_map

        return data

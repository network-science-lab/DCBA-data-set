"""Module for computing baseline ABCDConfig parameters from a graph."""

import igraph
import leidenalg
import torch
from torch_geometric.utils import degree

from dcba_data_set.graph_io.data_models import DCBAHeteroData
from dcba_data_set.julia_ports.abcd import ABCDConfig
from dcba_data_set.powerlaw import estimate_natural_cutoff, fit_truncated_power_law


class BaselineConfig:
    """Class for computing baseline ABCDConfig parameters from a given graph."""

    def __init__(
        self,
        graph: DCBAHeteroData,
        d_max_iter: int = 1000,
        c_max_iter: int = 1000,
        detect_communities: bool = False,
        leiden_resolution: float = 1.0,
        leiden_seed: int | None = None,
        natural_cutoff: bool = False,
    ):
        """
        Initialize the BaselineConfig by extracting necessary information from the graph.

        :param graph: Single-layer graph to extract the baseline config from.
        :param d_max_iter: Maximum number of iterations passed to the resulting ABCDConfig.
        :param c_max_iter: Maximum number of iterations passed to the resulting ABCDConfig.
        :param detect_communities: If True, ignore the ground-truth communities stored on the
            graph and instead detect them from the graph structure using the Leiden algorithm.
            Simulates the scenario where ground-truth communities are not available.
        :param leiden_resolution: Resolution parameter passed to the Leiden algorithm. Only used
            when ``detect_communities`` is True.
        :param leiden_seed: Random seed for the Leiden algorithm. Only used when
            ``detect_communities`` is True.
        :param natural_cutoff: If True, estimate d_max and c_max using the natural cutoff
            formula instead of the raw sample maximum, which underestimates the true upper
            bound of a truncated power-law sample.
        """
        if graph.edge_types != [("actor", "l_0", "actor")]:
            raise NotImplementedError(
                "Baseline config extraction only supports single layer graphs."
            )
        self.src, self.dst = graph["actor", "l_0", "actor"].edge_index
        if detect_communities:
            self.communities = BaselineConfig._detect_communities(
                self.src,
                self.dst,
                graph["actor"].community.size(0),
                resolution=leiden_resolution,
                seed=leiden_seed,
            )
        else:
            self.communities = BaselineConfig._prepare_communities(graph["actor"].community)
        self.n = self.communities.size(0)
        self.e = (
            graph.num_edges
        )  # self.e and intra_community_mask count directed edges consistently
        self.degrees = degree(self.src, dtype=torch.int)
        self.intra_community_mask = self.communities[self.src] == self.communities[self.dst]
        self.d_max_iter = d_max_iter
        self.c_max_iter = c_max_iter
        self.detect_communities = detect_communities
        self.natural_cutoff = natural_cutoff

    @staticmethod
    def _prepare_communities(communities: torch.Tensor) -> torch.Tensor:
        """Prepare the community assignments by ensuring they are zero-indexed and contiguous."""
        return communities[:, 0] - communities.min()

    @staticmethod
    def _detect_communities(
        src: torch.Tensor,
        dst: torch.Tensor,
        num_nodes: int,
        resolution: float = 1.0,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Detect zero-indexed community assignments from the graph structure using Leiden."""
        edges = torch.stack([src, dst], dim=1).tolist()
        ig_graph = igraph.Graph(n=num_nodes, edges=edges, directed=False)
        # src/dst hold both directed copies of each undirected edge; igraph treats (u, v) and
        # (v, u) as a multi-edge pair here, so simplify() collapses each back to a single edge.
        ig_graph.simplify(multiple=True, loops=True)
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            seed=seed,
        )
        return torch.tensor(partition.membership, dtype=torch.long)

    def _get_noise(self) -> float:
        """Calculate the noise ratio (xi) based on the proportion of inter-community edges."""
        num_intra_edges = self.intra_community_mask.sum().item()
        ratio = 1 - num_intra_edges / self.e
        return ratio

    def _get_outlier_scores(self) -> torch.Tensor:
        """
        Calculate the B(v) score for every node.

        Formula: B(v) = 2 * (deg_A(v)/deg(v) - (vol(A)-deg(v))/vol(V)).
        For details see: https://arxiv.org/pdf/2311.04730.
        """
        vol_V = self.degrees.sum()
        vol_A = torch.bincount(self.communities, weights=self.degrees)
        vol_A_v = vol_A[self.communities]
        internal_src = self.src[self.intra_community_mask]
        deg_A = torch.bincount(internal_src, minlength=self.n).to(self.degrees.dtype)
        B_v = 2 * (deg_A / self.degrees - (vol_A_v - self.degrees) / vol_V)
        B_v[self.degrees == 0] = 1.0  # Isolated nodes are not considered outliers
        return B_v

    def _get_n_outliers(self) -> int:
        """Calculate the number of outlier nodes based on their B(v) score."""
        return int((self._get_outlier_scores() < 0).sum().item())

    def _has_outlier_community(self) -> bool:
        """Check whether community 0 is a reserved ABCD outlier bucket.

        ABCD's generator always assigns outlier nodes to community 0 (when the
        graph was generated with outliers), so community 0's nodes are almost entirely
        flagged as outliers by their own B(v) score. A genuine (e.g. ground-truth or
        Leiden-detected) community only rarely has a majority of its nodes score as outliers.
        """
        if self.detect_communities:
            return False
        comm_0_mask = self.communities == 0
        return (self._get_outlier_scores()[comm_0_mask] < 0).float().mean().item() > 0.5

    def get_config(self) -> ABCDConfig:
        """Extract ABCDConfig parameters from a graph."""
        comm_sizes = torch.bincount(self.communities)
        if self._has_outlier_community():
            # Community 0 is the reserved outlier bucket, so its exact size is nout, and it
            # must be excluded from the c_min/c_max/t2 estimation as it is not a genuine community.
            n_outliers = int(comm_sizes[0].item())
            comm_sizes = comm_sizes[1:]
        else:
            n_outliers = self._get_n_outliers()
        t1 = fit_truncated_power_law(self.degrees)
        t2 = fit_truncated_power_law(comm_sizes)
        if self.natural_cutoff:
            d_max = estimate_natural_cutoff(self.degrees, t1, max_bound=self.n - 1)
            c_max = estimate_natural_cutoff(comm_sizes, t2, max_bound=self.n)
        else:
            d_max = int(self.degrees.max().item())
            c_max = int(comm_sizes.max().item())
        return ABCDConfig(
            n=self.n,
            t1=t1,
            d_min=int(self.degrees.min().item()),
            d_max=d_max,
            d_max_iter=self.d_max_iter,
            t2=t2,
            c_min=int(comm_sizes.min().item()),
            c_max=c_max,
            c_max_iter=self.c_max_iter,
            xi=self._get_noise(),
            nout=n_outliers,
        )

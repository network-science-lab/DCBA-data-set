import warnings

import powerlaw
import torch
from torch_geometric.utils import degree

from dcba_data_set.graph_io.data_models import DCBAHeteroData
from dcba_data_set.julia_ports.abcd import ABCDConfig

class BaselineConfig:
    """Class for computing baseline ABCDConfig parameters from a given graph."""
    def __init__(self, graph: DCBAHeteroData):
        if graph["actor"].community.size(1) != 1:
            raise NotImplementedError(
                "Baseline config extraction only supports single layer graphs."
            )
        self.src, self.dst = graph["actor", "l_0", "actor"].edge_index
        self.communities = BaselineConfig._prepare_communities(graph["actor"].community)
        self.n = self.communities.size(0)
        self.e = graph.num_edges
        self.degrees = degree(self.src, dtype=torch.int)
        self.intra_community_mask = self.communities[self.src] == self.communities[self.dst]

    @staticmethod
    def _prepare_communities(communities: torch.Tensor) -> torch.Tensor:
        """Prepares the community tensor by ensuring it is 1D and starts from 0."""
        return communities.transpose(0, 1)[0] - communities.min()

    @staticmethod
    def _get_powerlaw_exponent(data: torch.Tensor, d_min: int, d_max: int) -> float:
        """Fits a power-law distribution to the data and returns the exponent."""
        with warnings.catch_warnings(action="ignore"):
            fit = powerlaw.Fit(data.numpy(), discrete=True, xmin=d_min, xmax=d_max, verbose=0)
            return fit.power_law.alpha

    def _get_noise(self) -> float:
        """Calculates the noise ratio (xi), defined as the proportion of edges that connect nodes from different communities."""
        num_intra_edges = self.intra_community_mask.sum().item()
        ratio = 1 - num_intra_edges / self.e
        return ratio

    def _get_n_outliers(self) -> int:
        """
        Calculates the number of outlier nodes based on their B(v) score.
        Formula: B(v) = 2 * (deg_A(v)/deg(v) - (vol(A)-deg(v))/vol(V))
        For details see: https://arxiv.org/pdf/2311.04730
        """
        vol_V = self.degrees.sum()
        vol_A = torch.bincount(self.communities, weights=self.degrees)
        vol_A_v = vol_A[self.communities]
        internal_src = self.src[self.intra_community_mask]
        deg_A = torch.bincount(internal_src, minlength=self.n).to(self.degrees.dtype)
        B_v = 2 * (deg_A / self.degrees - (vol_A_v - self.degrees) / vol_V)
        n_outliers = int((B_v < 0).sum().item())
        return n_outliers

    def get_config(self) -> ABCDConfig:
        """Extracts ABCDConfig parameters from a graph."""
        d_min = int(self.degrees.min().item())
        d_max = int(self.degrees.max().item())
        comm_sizes = torch.bincount(self.communities)
        c_min = int(comm_sizes.min().item())
        c_max = int(comm_sizes.max().item())
        return ABCDConfig(
            n=self.n,
            t1=BaselineConfig._get_powerlaw_exponent(self.degrees, d_min, d_max),
            d_min=d_min,
            d_max=d_max,
            d_max_iter=1000,
            t2=BaselineConfig._get_powerlaw_exponent(comm_sizes, c_min, c_max),
            c_min=c_min,
            c_max=c_max,
            c_max_iter=1000,
            xi=self._get_noise(),
            nout=self._get_n_outliers(),
        )
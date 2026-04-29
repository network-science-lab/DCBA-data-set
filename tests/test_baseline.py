"""Unit tests for baseline graph-statistics extraction."""

import pytest
import torch

from dcba_data_set.baseline import BaselineConfig
from dcba_data_set.graph_io.data_models import DCBAHeteroData


def _build_graph(edge_index: torch.Tensor, communities: torch.Tensor) -> DCBAHeteroData:
    """Build a minimal single-layer graph for baseline tests."""
    graph = DCBAHeteroData()
    graph["actor", "l_0", "actor"].edge_index = edge_index
    graph["actor"].community = communities
    return graph


def test_initialisation_prepares_single_layer_graph_statistics() -> None:
    """Initialisation flattens communities and derives the expected graph statistics."""
    graph = _build_graph(
        edge_index=torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=torch.long),
        communities=torch.tensor([[1], [1], [2]], dtype=torch.long),
    )

    baseline = BaselineConfig(graph)

    assert baseline.communities.tolist() == [0, 0, 1]
    assert baseline.n == 3
    assert baseline.e == 4
    assert baseline.degrees.tolist() == [2, 1, 1]
    assert baseline.intra_community_mask.tolist() == [True, True, False, False]


def test_initialisation_rejects_multilayer_community_assignments() -> None:
    """Baseline extraction is limited to single-layer community tensors."""
    graph = _build_graph(
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        communities=torch.tensor([[1, 2], [1, 2]], dtype=torch.long),
    )

    with pytest.raises(NotImplementedError, match="single layer graphs"):
        BaselineConfig(graph)


def test_get_noise_uses_inter_community_edges_ratio() -> None:
    """Noise equals the fraction of directed edges crossing community boundaries."""
    graph = _build_graph(
        edge_index=torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=torch.long),
        communities=torch.tensor([[1], [1], [2]], dtype=torch.long),
    )

    baseline = BaselineConfig(graph)

    assert baseline._get_noise() == 0.5


def test_get_n_outliers_counts_negative_bv_nodes() -> None:
    """Outlier count matches the nodes whose baseline B(v) score is negative."""
    graph = _build_graph(
        edge_index=torch.tensor([[0, 2, 1, 3], [2, 0, 3, 1]], dtype=torch.long),
        communities=torch.tensor([[1], [1], [1], [2]], dtype=torch.long),
    )

    baseline = BaselineConfig(graph)

    assert baseline._get_n_outliers() == 1


def test_get_config_extracts_expected_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Config extraction reuses the graph statistics and fitted exponents consistently."""
    fit_calls: list[tuple[list[int], int, int]] = []

    baseline = BaselineConfig(
        _build_graph(
            edge_index=torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=torch.long),
            communities=torch.tensor([[1], [1], [2]], dtype=torch.long),
        )
    )

    def _fake_get_powerlaw_exponent(data: torch.Tensor, d_min: int, d_max: int) -> float:
        fit_calls.append((data.tolist(), d_min, d_max))
        return float(sum(data.tolist()))

    monkeypatch.setattr(BaselineConfig, "_get_powerlaw_exponent", _fake_get_powerlaw_exponent)

    config = baseline.get_config()

    assert config.n == 3
    assert config.t1 == 4.0
    assert config.d_min == 1
    assert config.d_max == 2
    assert config.d_max_iter == 1000
    assert config.t2 == 3.0
    assert config.c_min == 1
    assert config.c_max == 2
    assert config.c_max_iter == 1000
    assert config.xi == 0.5
    assert config.nout == 0
    assert fit_calls == [([2, 1, 1], 1, 2), ([2, 1], 1, 2)]
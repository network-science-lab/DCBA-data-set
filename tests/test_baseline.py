"""Unit tests for BaselineConfig against the ABCD, mABCD, and Karate test datasets."""

import pytest
import torch

from dcba_data_set.baseline import BaselineConfig
from dcba_data_set.graph_io import DCBAHeteroData, InstanceRecord
from dcba_data_set.julia_ports.abcd import ABCDConfig


@pytest.fixture(scope="module")
def abcd_graph(abcd_records: list[InstanceRecord]) -> DCBAHeteroData:
    """Load one ABCD graph for the entire module."""
    record = abcd_records[0]
    return DCBAHeteroData.from_replica_record(
        record.replicas[0], record.instance_id, record.net_type
    )


@pytest.fixture(scope="module")
def mabcd_graph(mabcd_records: list[InstanceRecord]) -> DCBAHeteroData:
    """Load one mABCD graph for the entire module."""
    record = mabcd_records[0]
    return DCBAHeteroData.from_replica_record(
        record.replicas[0], record.instance_id, record.net_type
    )


@pytest.fixture(scope="module")
def karate_graph(karate_records: list[InstanceRecord]) -> DCBAHeteroData:
    """Load the karate graph for the entire module."""
    record = karate_records[0]
    return DCBAHeteroData.from_replica_record(
        record.replicas[0], record.instance_id, record.net_type
    )


@pytest.fixture(scope="module")
def abcd_config(abcd_graph: DCBAHeteroData) -> ABCDConfig:
    """Compute the baseline ABCDConfig from the ABCD graph once for the entire module."""
    return BaselineConfig(abcd_graph).get_config()


@pytest.fixture(scope="module")
def karate_config(karate_graph: DCBAHeteroData) -> ABCDConfig:
    """Compute the baseline ABCDConfig from the karate graph once for the entire module."""
    return BaselineConfig(karate_graph).get_config()


@pytest.fixture(scope="module")
def karate_detected_config(karate_graph: DCBAHeteroData) -> ABCDConfig:
    """Compute the baseline ABCDConfig from Leiden-detected communities on the karate graph."""
    return BaselineConfig(karate_graph, detect_communities=True, leiden_seed=42).get_config()


@pytest.fixture(scope="module")
def abcd_natural_cutoff_config(abcd_graph: DCBAHeteroData) -> ABCDConfig:
    """Compute the baseline ABCDConfig from the ABCD graph using the natural cutoff estimator."""
    return BaselineConfig(abcd_graph, natural_cutoff=True).get_config()


class TestBaselineAbcd:
    """Tests for BaselineConfig against the ABCD test dataset."""

    def test_construction(self, abcd_config: ABCDConfig) -> None:
        """BaselineConfig can be constructed from a single-layer ABCD graph."""
        assert isinstance(abcd_config, ABCDConfig)

    def test_get_config_returns_abcd_config(self, abcd_config: ABCDConfig) -> None:
        """get_config returns an ABCDConfig instance."""
        assert isinstance(abcd_config, ABCDConfig)

    def test_config_n_matches_graph(
        self, abcd_config: ABCDConfig, abcd_graph: DCBAHeteroData
    ) -> None:
        """Extracted n matches the number of actors in the graph."""
        assert abcd_config.n == len(abcd_graph.actors_map)

    def test_noise_bounds(self, abcd_config: ABCDConfig) -> None:
        """Noise ratio xi lies in [0, 1]."""
        assert 0.0 <= abcd_config.xi <= 1.0

    def test_degree_bounds(self, abcd_config: ABCDConfig) -> None:
        """Minimum degree does not exceed maximum degree."""
        assert abcd_config.d_min <= abcd_config.d_max

    def test_community_bounds(self, abcd_config: ABCDConfig) -> None:
        """Minimum community size does not exceed maximum community size."""
        assert abcd_config.c_min <= abcd_config.c_max


class TestBaselineAbcdNaturalCutoff:
    """Tests for BaselineConfig's natural cutoff estimator against the ABCD test dataset."""

    def test_construction(self, abcd_natural_cutoff_config: ABCDConfig) -> None:
        """BaselineConfig can be constructed with natural_cutoff=True."""
        assert isinstance(abcd_natural_cutoff_config, ABCDConfig)

    def test_degree_bounds(self, abcd_natural_cutoff_config: ABCDConfig) -> None:
        """Minimum degree does not exceed maximum degree."""
        assert abcd_natural_cutoff_config.d_min <= abcd_natural_cutoff_config.d_max

    def test_community_bounds(self, abcd_natural_cutoff_config: ABCDConfig) -> None:
        """Minimum community size does not exceed maximum community size."""
        assert abcd_natural_cutoff_config.c_min <= abcd_natural_cutoff_config.c_max

    def test_d_max_within_graph_size(
        self, abcd_natural_cutoff_config: ABCDConfig, abcd_graph: DCBAHeteroData
    ) -> None:
        """d_max never exceeds the largest degree physically possible in the graph."""
        assert abcd_natural_cutoff_config.d_max <= len(abcd_graph.actors_map) - 1

    def test_c_max_within_graph_size(
        self, abcd_natural_cutoff_config: ABCDConfig, abcd_graph: DCBAHeteroData
    ) -> None:
        """c_max never exceeds the number of actors in the graph."""
        assert abcd_natural_cutoff_config.c_max <= len(abcd_graph.actors_map)

    def test_natural_cutoff_at_least_naive(
        self, abcd_natural_cutoff_config: ABCDConfig, abcd_config: ABCDConfig
    ) -> None:
        """Natural cutoff estimates are never smaller than the raw sample maxima."""
        assert abcd_natural_cutoff_config.d_max >= abcd_config.d_max
        assert abcd_natural_cutoff_config.c_max >= abcd_config.c_max


class TestBaselineMabcd:
    """Tests for BaselineConfig against the mABCD test dataset."""

    def test_raises_not_implemented(self, mabcd_graph: DCBAHeteroData) -> None:
        """BaselineConfig raises NotImplementedError for multilayer mABCD graphs."""
        with pytest.raises(NotImplementedError):
            BaselineConfig(mabcd_graph)


class TestBaselineKarate:
    """Tests for BaselineConfig against the Karate test dataset."""

    def test_construction(self, karate_config: ABCDConfig) -> None:
        """BaselineConfig can be constructed from the single-layer karate graph."""
        assert isinstance(karate_config, ABCDConfig)

    def test_get_config_returns_abcd_config(self, karate_config: ABCDConfig) -> None:
        """get_config returns an ABCDConfig instance."""
        assert isinstance(karate_config, ABCDConfig)

    def test_config_n_matches_graph(
        self, karate_config: ABCDConfig, karate_graph: DCBAHeteroData
    ) -> None:
        """Extracted n matches the number of actors in the karate graph."""
        assert karate_config.n == len(karate_graph.actors_map)

    def test_noise_bounds(self, karate_config: ABCDConfig) -> None:
        """Noise ratio xi lies in [0, 1]."""
        assert 0.0 <= karate_config.xi <= 1.0

    def test_degree_bounds(self, karate_config: ABCDConfig) -> None:
        """Minimum degree does not exceed maximum degree."""
        assert karate_config.d_min <= karate_config.d_max

    def test_community_bounds(self, karate_config: ABCDConfig) -> None:
        """Minimum community size does not exceed maximum community size."""
        assert karate_config.c_min <= karate_config.c_max


class TestBaselineKarateDetectedCommunities:
    """Tests for BaselineConfig using Leiden-detected communities on the karate graph."""

    def test_construction(self, karate_detected_config: ABCDConfig) -> None:
        """BaselineConfig can be constructed with detect_communities=True."""
        assert isinstance(karate_detected_config, ABCDConfig)

    def test_config_n_matches_graph(
        self, karate_detected_config: ABCDConfig, karate_graph: DCBAHeteroData
    ) -> None:
        """Extracted n matches the number of actors in the graph, regardless of community source."""
        assert karate_detected_config.n == len(karate_graph.actors_map)

    def test_noise_bounds(self, karate_detected_config: ABCDConfig) -> None:
        """Noise ratio xi lies in [0, 1]."""
        assert 0.0 <= karate_detected_config.xi <= 1.0

    def test_community_bounds(self, karate_detected_config: ABCDConfig) -> None:
        """Minimum community size does not exceed maximum community size."""
        assert karate_detected_config.c_min <= karate_detected_config.c_max

    def test_communities_are_zero_indexed_and_contiguous(
        self, karate_graph: DCBAHeteroData
    ) -> None:
        """Leiden-detected community labels are zero-indexed and contiguous."""
        baseline = BaselineConfig(karate_graph, detect_communities=True, leiden_seed=42)
        unique_communities = baseline.communities.unique(sorted=True)
        expected = torch.arange(unique_communities.numel())
        assert torch.equal(unique_communities, expected)

    def test_detected_communities_differ_from_ground_truth(
        self, karate_graph: DCBAHeteroData
    ) -> None:
        """Leiden detection is independent of the ground-truth community count."""
        ground_truth = BaselineConfig(karate_graph)
        detected = BaselineConfig(karate_graph, detect_communities=True, leiden_seed=42)
        assert detected.communities.size(0) == ground_truth.communities.size(0)

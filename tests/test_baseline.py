"""Unit tests for BaselineConfig against the ABCD, mABCD, and Karate test datasets."""

import pytest

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

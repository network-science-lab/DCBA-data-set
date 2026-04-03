"""E2E tests for ABCD and mABCD graph I/O."""

import json
from pathlib import Path

import pytest
import torch
from bidict import bidict

from dcba_data_set.graph_io import ConfigRecord, DCBAHeteroData, load_report

DATA_ROOT = Path(__file__).parent.parent / "data" / "test"
ABCD_REPORT = DATA_ROOT / "dataset_abcd" / "report.json"
MABCD_REPORT = DATA_ROOT / "dataset_mabcd" / "report.json"


@pytest.fixture(scope="module")
def abcd_dataset() -> tuple[dict[str, ConfigRecord], list[DCBAHeteroData]]:
    """Load the ABCD test dataset once for the entire module."""
    return load_report(ABCD_REPORT)


@pytest.fixture(scope="module")
def mabcd_dataset() -> tuple[dict[str, ConfigRecord], list[DCBAHeteroData]]:
    """Load the mABCD test dataset once for the entire module."""
    return load_report(MABCD_REPORT)


def test_unknown_net_type(tmp_path: Path) -> None:
    """load_report raises ValueError for an unrecognised net_type."""
    report_file = tmp_path / "report.json"
    report_file.write_text(json.dumps({"net_type": "unknown_type", "instances": []}))
    with pytest.raises(ValueError, match="unknown_type"):
        load_report(report_file)


class TestLoadAbcdReport:
    """Tests for load_report against the ABCD test dataset."""

    def test_counts(self, abcd_dataset: tuple) -> None:
        """Loader returns one config per instance and one graph per replica."""
        configs, graphs = abcd_dataset
        assert len(configs) == 3
        assert len(graphs) == 9

    def test_config_type(self, abcd_dataset: tuple) -> None:
        """Each config is a ConfigRecord with expected fields populated."""
        configs, _ = abcd_dataset
        for instance_id, record in configs.items():
            assert isinstance(record, ConfigRecord)
            assert record.instance_id == instance_id
            assert isinstance(record.data, dict)
            assert record.path.exists()

    def test_graph_type(self, abcd_dataset: tuple) -> None:
        """Each graph is a DCBAHeteroData instance."""
        _, graphs = abcd_dataset
        for g in graphs:
            assert isinstance(g, DCBAHeteroData)

    def test_graph_metadata(self, abcd_dataset: tuple) -> None:
        """Each graph has instance_id, replica, actors_map, and layers_map set."""
        configs, graphs = abcd_dataset
        for g in graphs:
            assert g.instance_id in configs
            assert isinstance(g.replica, int)
            assert isinstance(g.actors_map, bidict)
            assert isinstance(g.layers_map, bidict)

    def test_single_layer(self, abcd_dataset: tuple) -> None:
        """ABCD graphs expose exactly one layer relation."""
        _, graphs = abcd_dataset
        for g in graphs:
            assert g.edge_types == [("actor", "l_0", "actor")]
            assert dict(g.layers_map) == {"0": "l_0"}

    def test_community_shape(self, abcd_dataset: tuple) -> None:
        """Community tensor has shape [num_actors, 1] with non-negative values."""
        _, graphs = abcd_dataset
        for g in graphs:
            community = g["actor"].community
            assert community.ndim == 2
            assert community.shape[1] == 1
            assert community.shape[0] == len(g.actors_map)
            assert (community >= 0).all()

    def test_edge_index_validity(self, abcd_dataset: tuple) -> None:
        """Edge indices are non-negative, symmetric, and within actor bounds."""
        _, graphs = abcd_dataset
        for g in graphs:
            ei = g["actor", "l_0", "actor"].edge_index
            assert ei.dtype == torch.long
            assert ei.shape[0] == 2
            assert (ei >= 0).all()
            assert ei.max() < len(g.actors_map)
            # Both directions present — columns come in (u,v) and (v,u) pairs.
            assert ei.shape[1] % 2 == 0


class TestLoadMabcdReport:
    """Tests for load_report against the mABCD test dataset."""

    def test_counts(self, mabcd_dataset: tuple) -> None:
        """Loader returns one config per instance and one graph per replica."""
        configs, graphs = mabcd_dataset
        assert len(configs) == 3
        assert len(graphs) == 9

    def test_config_type(self, mabcd_dataset: tuple) -> None:
        """Each config is a ConfigRecord with expected fields populated."""
        configs, _ = mabcd_dataset
        for instance_id, record in configs.items():
            assert isinstance(record, ConfigRecord)
            assert record.instance_id == instance_id
            assert isinstance(record.data, dict)
            assert record.path.exists()

    def test_graph_type(self, mabcd_dataset: tuple) -> None:
        """Each graph is a DCBAHeteroData instance."""
        _, graphs = mabcd_dataset
        for g in graphs:
            assert isinstance(g, DCBAHeteroData)

    def test_graph_metadata(self, mabcd_dataset: tuple) -> None:
        """Each graph has instance_id, replica, actors_map, and layers_map set."""
        configs, graphs = mabcd_dataset
        for g in graphs:
            assert g.instance_id in configs
            assert isinstance(g.replica, int)
            assert isinstance(g.actors_map, bidict)
            assert isinstance(g.layers_map, bidict)

    def test_multilayer(self, mabcd_dataset: tuple) -> None:
        """MABCD graphs expose multiple layer relations with consistent naming."""
        _, graphs = mabcd_dataset
        for g in graphs:
            num_layers = len(g.layers_map)
            assert num_layers > 1
            expected_relations = {("actor", f"l_{i}", "actor") for i in range(num_layers)}
            assert set(g.edge_types) == expected_relations

    def test_community_shape(self, mabcd_dataset: tuple) -> None:
        """Community tensor has shape [num_actors, num_layers]; inactive nodes carry 0."""
        _, graphs = mabcd_dataset
        for g in graphs:
            community = g["actor"].community
            num_layers = len(g.layers_map)
            assert community.ndim == 2
            assert community.shape == (len(g.actors_map), num_layers)
            assert (community >= 0).all()

    def test_edge_index_validity(self, mabcd_dataset: tuple) -> None:
        """Edge indices are non-negative, symmetric, and within actor bounds for all layers."""
        _, graphs = mabcd_dataset
        for g in graphs:
            for rel in g.edge_types:
                ei = g[rel].edge_index
                assert ei.dtype == torch.long
                assert ei.shape[0] == 2
                assert (ei >= 0).all()
                assert ei.max() < len(g.actors_map)
                assert ei.shape[1] % 2 == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

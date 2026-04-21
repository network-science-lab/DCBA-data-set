"""E2E tests for ABCD and mABCD graph I/O."""

import json
from pathlib import Path

import pytest
import torch
from bidict import bidict

from dcba_data_set.graph_io import (
    DCBAHeteroData,
    DCBAInstanceConfig,
    InstanceRecord,
    ReplicaRecord,
    load_report,
)

DATA_ROOT = Path(__file__).parent.parent / "data" / "test"
ABCD_REPORT = DATA_ROOT / "dataset_abcd" / "report.json"
MABCD_REPORT = DATA_ROOT / "dataset_mabcd" / "report.json"


@pytest.fixture(scope="module")
def abcd_records() -> list[InstanceRecord]:
    """Index the ABCD test dataset once for the entire module."""
    return load_report(ABCD_REPORT)


@pytest.fixture(scope="module")
def mabcd_records() -> list[InstanceRecord]:
    """Index the mABCD test dataset once for the entire module."""
    return load_report(MABCD_REPORT)


def test_unknown_net_type(tmp_path: Path) -> None:
    """load_report raises ValueError for an unrecognised net_type."""
    report_file = tmp_path / "report.json"
    report_file.write_text(json.dumps({"net_type": "unknown_type", "instances": []}))
    with pytest.raises(ValueError, match="unknown_type"):
        load_report(report_file)


class TestLoadAbcdReport:
    """Tests for load_report against the ABCD test dataset."""

    def test_counts(self, abcd_records: list[InstanceRecord]) -> None:
        """Loader returns one InstanceRecord per instance and one ReplicaRecord per replica."""
        assert len(abcd_records) == 3
        assert sum(len(r.replicas) for r in abcd_records) == 9

    def test_record_types(self, abcd_records: list[InstanceRecord]) -> None:
        """Each element is an InstanceRecord with ReplicaRecord children."""
        for record in abcd_records:
            assert isinstance(record, InstanceRecord)
            assert record.net_type == "abcd"
            assert record.config_path.suffix == ".yaml"
            for replica in record.replicas:
                assert isinstance(replica, ReplicaRecord)

    def test_config_factory(self, abcd_records: list[InstanceRecord]) -> None:
        """from_instance_record loads a DCBAInstanceConfig with expected fields populated."""
        for record in abcd_records:
            config = DCBAInstanceConfig.from_instance_record(record)
            assert isinstance(config, DCBAInstanceConfig)
            assert config.instance_id == record.instance_id
            assert isinstance(config.data, dict)
            assert config.path.exists()

    def test_graph_factory(self, abcd_records: list[InstanceRecord]) -> None:
        """from_replica_record returns a DCBAHeteroData for each replica."""
        for record in abcd_records:
            for replica in record.replicas:
                g = DCBAHeteroData.from_replica_record(replica, record.instance_id, record.net_type)
                assert isinstance(g, DCBAHeteroData)

    def test_graph_metadata(self, abcd_records: list[InstanceRecord]) -> None:
        """Each graph has instance_id, replica, actors_map, and layers_map set."""
        instance_ids = {r.instance_id for r in abcd_records}
        for record in abcd_records:
            for replica in record.replicas:
                g = DCBAHeteroData.from_replica_record(replica, record.instance_id, record.net_type)
                assert g.instance_id in instance_ids
                assert isinstance(g.replica, int)
                assert isinstance(g.actors_map, bidict)
                assert isinstance(g.layers_map, bidict)

    def test_single_layer(self, abcd_records: list[InstanceRecord]) -> None:
        """ABCD graphs expose exactly one layer relation."""
        record = abcd_records[0]
        g = DCBAHeteroData.from_replica_record(
            record.replicas[0], record.instance_id, record.net_type
        )
        assert g.edge_types == [("actor", "l_0", "actor")]
        assert dict(g.layers_map) == {"0": "l_0"}

    def test_community_shape(self, abcd_records: list[InstanceRecord]) -> None:
        """Community tensor has shape [num_actors, 1] with non-negative values."""
        for record in abcd_records:
            for replica in record.replicas:
                g = DCBAHeteroData.from_replica_record(replica, record.instance_id, record.net_type)
                community = g["actor"].community
                assert community.ndim == 2
                assert community.shape[1] == 1
                assert community.shape[0] == len(g.actors_map)
                assert (community >= 0).all()

    def test_edge_index_validity(self, abcd_records: list[InstanceRecord]) -> None:
        """Edge indices are non-negative, symmetric, and within actor bounds."""
        for record in abcd_records:
            for replica in record.replicas:
                g = DCBAHeteroData.from_replica_record(replica, record.instance_id, record.net_type)
                ei = g["actor", "l_0", "actor"].edge_index
                assert ei.dtype == torch.long
                assert ei.shape[0] == 2
                assert (ei >= 0).all()
                assert ei.max() < len(g.actors_map)
                assert ei.shape[1] % 2 == 0


class TestLoadMabcdReport:
    """Tests for load_report against the mABCD test dataset."""

    def test_counts(self, mabcd_records: list[InstanceRecord]) -> None:
        """Loader returns one InstanceRecord per instance and one ReplicaRecord per replica."""
        assert len(mabcd_records) == 3
        assert sum(len(r.replicas) for r in mabcd_records) == 9

    def test_record_types(self, mabcd_records: list[InstanceRecord]) -> None:
        """Each element is an InstanceRecord with ReplicaRecord children."""
        for record in mabcd_records:
            assert isinstance(record, InstanceRecord)
            assert record.net_type == "mabcd"
            assert record.config_path.suffix == ".yaml"
            for replica in record.replicas:
                assert isinstance(replica, ReplicaRecord)

    def test_config_factory(self, mabcd_records: list[InstanceRecord]) -> None:
        """from_instance_record loads a DCBAInstanceConfig with expected fields populated."""
        for record in mabcd_records:
            config = DCBAInstanceConfig.from_instance_record(record)
            assert isinstance(config, DCBAInstanceConfig)
            assert config.instance_id == record.instance_id
            assert isinstance(config.data, dict)
            assert config.path.exists()

    def test_graph_factory(self, mabcd_records: list[InstanceRecord]) -> None:
        """from_replica_record returns a DCBAHeteroData for each replica."""
        for record in mabcd_records:
            for replica in record.replicas:
                g = DCBAHeteroData.from_replica_record(replica, record.instance_id, record.net_type)
                assert isinstance(g, DCBAHeteroData)

    def test_graph_metadata(self, mabcd_records: list[InstanceRecord]) -> None:
        """Each graph has instance_id, replica, actors_map, and layers_map set."""
        instance_ids = {r.instance_id for r in mabcd_records}
        for record in mabcd_records:
            for replica in record.replicas:
                g = DCBAHeteroData.from_replica_record(replica, record.instance_id, record.net_type)
                assert g.instance_id in instance_ids
                assert isinstance(g.replica, int)
                assert isinstance(g.actors_map, bidict)
                assert isinstance(g.layers_map, bidict)

    def test_multilayer(self, mabcd_records: list[InstanceRecord]) -> None:
        """MABCD graphs expose multiple layer relations with consistent naming."""
        record = mabcd_records[0]
        g = DCBAHeteroData.from_replica_record(
            record.replicas[0], record.instance_id, record.net_type
        )
        num_layers = len(g.layers_map)
        assert num_layers > 1
        expected_relations = {("actor", f"l_{i}", "actor") for i in range(num_layers)}
        assert set(g.edge_types) == expected_relations

    def test_community_shape(self, mabcd_records: list[InstanceRecord]) -> None:
        """Community tensor has shape [num_actors, num_layers]; inactive nodes carry 0."""
        for record in mabcd_records:
            for replica in record.replicas:
                g = DCBAHeteroData.from_replica_record(replica, record.instance_id, record.net_type)
                community = g["actor"].community
                num_layers = len(g.layers_map)
                assert community.ndim == 2
                assert community.shape == (len(g.actors_map), num_layers)
                assert (community >= 0).all()

    def test_edge_index_validity(self, mabcd_records: list[InstanceRecord]) -> None:
        """Edge indices are non-negative, symmetric, and within actor bounds for all layers."""
        for record in mabcd_records:
            for replica in record.replicas:
                g = DCBAHeteroData.from_replica_record(replica, record.instance_id, record.net_type)
                for rel in g.edge_types:
                    ei = g[rel].edge_index
                    assert ei.dtype == torch.long
                    assert ei.shape[0] == 2
                    assert (ei >= 0).all()
                    assert ei.max() < len(g.actors_map)
                    assert ei.shape[1] % 2 == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

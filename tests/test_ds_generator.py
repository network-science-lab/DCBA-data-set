"""Unit tests for seed assignment in DatasetGenerator and mABCDGenerator."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from dcba_data_set.ds_generator import DatasetGenerator
from dcba_data_set.net_generator import mABCDGenerator

_RNG_SEED = 42


def _sampled_config_mock() -> MagicMock:
    """Return a mock that satisfies the sampled-config interface used by DatasetGenerator."""
    cfg = MagicMock()
    cfg.to_yaml.return_value = {}
    return cfg


def _dataset_config(rng_seed: int = _RNG_SEED) -> dict:
    """Minimal top-level config dict accepted by DatasetGenerator._run_instance."""
    return {"run": {"rng_seed": rng_seed}}


def _net_generator_config(tmp_path: Path, rng_seed: int = _RNG_SEED, repetitions: int = 3) -> dict:
    """Minimal config dict accepted by mABCDGenerator.run_experiments."""
    return {
        "run": {"rng_seed": rng_seed},
        "generator": {"repetitions": repetitions, "out_dir": str(tmp_path)},
        "net_config": {
            "edges_filename": "edges.dat",
            "communities_filename": "communities.dat",
        },
    }


class TestDatasetGeneratorReplicaSeeds:
    """Verify that DatasetGenerator assigns a distinct seed to each replica."""

    _gen = DatasetGenerator()

    def _run(self, tmp_path: Path, net_replicas: int) -> list[int]:
        """Invoke _run_instance with a mocked generator and return recorded seeds."""
        instance_dir = tmp_path / "instance"
        instance_dir.mkdir()

        seeds: list[int] = []
        mock_gen_instance = MagicMock(side_effect=lambda cfg: seeds.append(cfg.seed))
        mock_generator = MagicMock(return_value=mock_gen_instance)

        self._gen._run_instance(
            sampled_config=_sampled_config_mock(),
            instance_id="test-instance",
            instance_dir=instance_dir,
            generator=mock_generator,
            net_replicas=net_replicas,
            config=_dataset_config(),
            report={"instances": []},
        )

        return seeds

    def test_seeds_increment_by_replica_index(self, tmp_path: Path) -> None:
        """Each replica receives rng_seed + replica (1-indexed)."""
        assert self._run(tmp_path, net_replicas=3) == [_RNG_SEED + 1, _RNG_SEED + 2, _RNG_SEED + 3]


class TestMABCDGeneratorRepetitionSeeds:
    """Verify that mABCDGenerator assigns a distinct seed to each repetition."""

    def _run(self, tmp_path: Path, repetitions: int = 3) -> list[int]:
        """Run mABCDGenerator with mocked Julia class and return recorded seeds."""
        seeds: list[int] = []

        net_config_mock = MagicMock()
        julia_instance_mock = MagicMock(side_effect=lambda config: seeds.append(config.seed))

        with (
            patch.object(mABCDGenerator, "julia_config") as mock_cfg_cls,
            patch.object(mABCDGenerator, "julia_class", return_value=julia_instance_mock),
            patch("dcba_data_set.net_generator.create_out_dir", return_value=tmp_path),
        ):
            mock_cfg_cls.from_yaml.return_value = net_config_mock
            mABCDGenerator().run_experiments(
                _net_generator_config(tmp_path, repetitions=repetitions)
            )

        return seeds

    def test_seeds_increment_by_repetition_index(self, tmp_path: Path) -> None:
        """Each repetition receives rng_seed + repetition + 1 (1-indexed)."""
        seeds = self._run(tmp_path, repetitions=3)
        assert seeds == [_RNG_SEED + 1, _RNG_SEED + 2, _RNG_SEED + 3]

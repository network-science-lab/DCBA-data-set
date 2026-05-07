"""Unit tests for RandomConfigGenerator, GridConfigGenerator, and BorderlineConfigGenerator."""

from unittest.mock import MagicMock, patch

import pytest

from dcba_data_set.config_generators import (
    BorderlineConfigGenerator,
    GridConfigGenerator,
    RandomConfigGenerator,
)

_PATCH = "dcba_data_set.config_generators.ABCDConfig.from_yaml"


def _ranges(**overrides: object) -> dict:
    """Minimal ABCD config dict with absolute (non-fractional) values, with optional overrides."""
    params: dict = {
        "n": 1000,
        "t1": 2.5,
        "d_min": 5,
        "d_max": 50,
        "c_min": 10,
        "c_max": 100,
        "nout": 0,
        "t2": 2.0,
        "d_max_iter": 1000,
        "c_max_iter": 1000,
        "xi": 0.3,
    }
    params.update(overrides)
    return params


class TestRandomConfigGeneratorDrawConfig:
    """Verify parameter sampling and type handling in _draw_config."""

    def test_int_range_yields_int(self) -> None:
        """A [int, int] range is sampled as a Python int."""
        gen = RandomConfigGenerator(rng_seed=0, cfg_type="abcd", max_trials=5)
        result = gen._draw_config({"n": 1000, "d_min": [3, 9]})
        assert isinstance(result["d_min"], int)

    def test_int_range_within_bounds(self) -> None:
        """Sampled int falls within the specified range."""
        gen = RandomConfigGenerator(rng_seed=0, cfg_type="abcd", max_trials=5)
        result = gen._draw_config({"n": 1000, "d_min": [3, 9]})
        assert 3 <= result["d_min"] <= 9

    def test_float_range_yields_float_at_4dp(self) -> None:
        """A [float, float] range is sampled and rounded to 4 decimal places."""
        gen = RandomConfigGenerator(rng_seed=0, cfg_type="abcd", max_trials=5)
        result = gen._draw_config({"n": 1000, "t1": [2.0, 3.0]})
        assert result["t1"] == round(result["t1"], 4)

    def test_scalar_passes_through(self) -> None:
        """A non-list parameter value is preserved unchanged."""
        gen = RandomConfigGenerator(rng_seed=0, cfg_type="abcd", max_trials=5)
        result = gen._draw_config({"n": 1000, "xi": 0.3})
        assert result["xi"] == pytest.approx(0.3)

    def test_frac_keys_converted_and_removed(self) -> None:
        """Fractional keys are expanded to absolute values and stripped from the output."""
        gen = RandomConfigGenerator(rng_seed=0, cfg_type="abcd", max_trials=5)
        result = gen._draw_config({"n": 1000, "d_min_frac": [0.005, 0.01]})
        assert "d_min_frac" not in result
        assert "d_min" in result

    def test_reproducible_with_same_seed(self) -> None:
        """Identical seeds produce identical config dicts."""
        ranges = {"n": 1000, "d_min": [3, 9]}
        g1 = RandomConfigGenerator(rng_seed=7, cfg_type="abcd", max_trials=5)
        g2 = RandomConfigGenerator(rng_seed=7, cfg_type="abcd", max_trials=5)
        assert g1._draw_config(ranges) == g2._draw_config(ranges)


class TestRandomConfigGeneratorSampleOne:
    """Verify retry logic and error propagation in _sample_one."""

    def test_returns_config_on_first_success(self) -> None:
        """Returns the config produced by from_yaml when the first attempt succeeds."""
        gen = RandomConfigGenerator(rng_seed=0, cfg_type="abcd", max_trials=3)
        mock_cfg = MagicMock()
        with patch(_PATCH, return_value=mock_cfg):
            assert gen._sample_one(_ranges()) is mock_cfg

    def test_retries_after_failure_then_succeeds(self) -> None:
        """Retries after validation failures and returns the first successful config."""
        gen = RandomConfigGenerator(rng_seed=0, cfg_type="abcd", max_trials=3)
        mock_cfg = MagicMock()
        with patch(_PATCH, side_effect=[ValueError("bad"), ValueError("bad"), mock_cfg]) as m:
            result = gen._sample_one(_ranges())
        assert result is mock_cfg
        assert m.call_count == 3

    def test_raises_runtime_error_after_exhausting_trials(self) -> None:
        """Raises RuntimeError when all max_trials attempts fail."""
        gen = RandomConfigGenerator(rng_seed=0, cfg_type="abcd", max_trials=2)
        with patch(_PATCH, side_effect=ValueError("always invalid")):
            with pytest.raises(RuntimeError, match="2 trials"):
                gen._sample_one(_ranges())


class TestRandomConfigGeneratorGenerate:
    """Verify that generate() returns the correct list of configs."""

    def test_returns_all_n_configs_on_success(self) -> None:
        """Returns a list of length n_instances when all samples succeed."""
        gen = RandomConfigGenerator(rng_seed=0, cfg_type="abcd", max_trials=3)
        with patch(_PATCH, return_value=MagicMock()):
            results = gen.generate(_ranges(), n_instances=5)
        assert len(results) == 5

    def test_skips_failed_instances(self) -> None:
        """Instances that exhaust max_trials are skipped; list is shorter."""
        gen = RandomConfigGenerator(rng_seed=0, cfg_type="abcd", max_trials=1)
        mock_cfg = MagicMock()
        with patch(_PATCH, side_effect=[mock_cfg, ValueError("bad"), mock_cfg, ValueError("bad")]):
            results = gen.generate(_ranges(), n_instances=4)
        assert len(results) == 2

    def test_returns_empty_list_when_all_fail(self) -> None:
        """Returns an empty list when every sampling attempt exhausts max_trials."""
        gen = RandomConfigGenerator(rng_seed=0, cfg_type="abcd", max_trials=1)
        with patch(_PATCH, side_effect=ValueError("always invalid")):
            assert gen.generate(_ranges(), n_instances=3) == []


class TestGridConfigGeneratorInit:
    """Verify constructor behaviour."""

    def test_raises_for_unsupported_cfg_type(self) -> None:
        """Raises NotImplementedError when cfg_type is not 'abcd'."""
        with pytest.raises(NotImplementedError):
            GridConfigGenerator(cfg_type="mabcd", grid_steps=3)

    def test_accepts_abcd_cfg_type(self) -> None:
        """Constructs successfully for cfg_type='abcd'."""
        gen = GridConfigGenerator(cfg_type="abcd", grid_steps=3)
        assert gen.grid_steps == 3


class TestGridConfigGeneratorStepsFor:
    """Verify step-count resolution."""

    def test_global_int_returned_for_any_param(self) -> None:
        """A single integer grid_steps is returned regardless of parameter name."""
        gen = GridConfigGenerator(cfg_type="abcd", grid_steps=5)
        assert gen._steps_for("d_min") == 5
        assert gen._steps_for("c_max") == 5

    def test_per_param_value_from_dict(self) -> None:
        """A dict grid_steps returns the value mapped to the given parameter."""
        gen = GridConfigGenerator(cfg_type="abcd", grid_steps={"d_min": 3, "d_max": 7})
        assert gen._steps_for("d_min") == 3
        assert gen._steps_for("d_max") == 7

    def test_raises_for_param_absent_from_dict(self) -> None:
        """Raises ValueError when the parameter has no entry in the mapping."""
        gen = GridConfigGenerator(cfg_type="abcd", grid_steps={"d_min": 3})
        with pytest.raises(ValueError, match="d_max"):
            gen._steps_for("d_max")


class TestGridConfigGeneratorRegularGrid:
    """Verify grid generation with a single uniform step count (regular grid)."""

    def test_correct_combination_count(self) -> None:
        """Generates n_steps^k combinations for k range parameters."""
        # d_min: [3,9] → 3 distinct ints; d_max: [30,90] → 3 distinct ints → 9 combos
        gen = GridConfigGenerator(cfg_type="abcd", grid_steps=3)
        with patch(_PATCH, return_value=MagicMock()):
            results = gen.generate(_ranges(d_min=[3, 9], d_max=[30, 90]))
        assert len(results) == 9

    def test_single_range_param_yields_n_steps_configs(self) -> None:
        """With one range param and n steps, exactly n configs are produced."""
        gen = GridConfigGenerator(cfg_type="abcd", grid_steps=4)
        with patch(_PATCH, return_value=MagicMock()):
            results = gen.generate(_ranges(d_min=[3, 9]))
        assert len(results) == 4

    def test_fixed_params_unchanged_across_grid(self) -> None:
        """Fixed parameters retain their original value in every grid-point config."""
        gen = GridConfigGenerator(cfg_type="abcd", grid_steps=2)
        captured: list[dict] = []
        with patch(_PATCH, side_effect=lambda p: captured.append(dict(p)) or MagicMock()):
            gen.generate(_ranges(d_min=[3, 9]))
        assert all(p["n"] == 1000 for p in captured)
        assert all(p["xi"] == pytest.approx(0.3) for p in captured)

    def test_integer_range_values_are_whole_numbers(self) -> None:
        """Grid values derived from int ranges equal their integer counterparts."""
        gen = GridConfigGenerator(cfg_type="abcd", grid_steps=3)
        captured: list[dict] = []
        with patch(_PATCH, side_effect=lambda p: captured.append(dict(p)) or MagicMock()):
            gen.generate(_ranges(d_min=[3, 9]))
        assert all(p["d_min"] == int(p["d_min"]) for p in captured)


class TestGridConfigGeneratorIrregularGrid:
    """Verify grid generation when step counts differ per parameter (irregular grid)."""

    def test_per_param_steps_produce_correct_product(self) -> None:
        """d_min: 2 steps × d_max: 4 steps → 8 combinations."""
        gen = GridConfigGenerator(cfg_type="abcd", grid_steps={"d_min": 2, "d_max": 4})
        with patch(_PATCH, return_value=MagicMock()):
            results = gen.generate(_ranges(d_min=[3, 9], d_max=[30, 90]))
        assert len(results) == 8

    def test_asymmetric_steps_per_param(self) -> None:
        """Parameters with different step counts produce independent grid axes."""
        gen = GridConfigGenerator(cfg_type="abcd", grid_steps={"d_min": 3, "d_max": 2})
        captured: list[dict] = []
        with patch(_PATCH, side_effect=lambda p: captured.append(dict(p)) or MagicMock()):
            gen.generate(_ranges(d_min=[3, 9], d_max=[30, 90]))
        d_min_values = sorted({p["d_min"] for p in captured})
        d_max_values = sorted({p["d_max"] for p in captured})
        assert len(d_min_values) == 3
        assert len(d_max_values) == 2

    def test_missing_range_param_raises_value_error(self) -> None:
        """Raises ValueError during generate when a range param is absent from the mapping."""
        gen = GridConfigGenerator(cfg_type="abcd", grid_steps={"d_min": 3})
        with pytest.raises(ValueError, match="d_max"):
            gen.generate(_ranges(d_min=[3, 9], d_max=[30, 90]))


class TestGridConfigGeneratorDropsInvalidPoints:
    """Verify that invalid grid points are silently excluded from the result."""

    def test_invalid_points_are_dropped(self) -> None:
        """Points that fail validation are excluded; valid ones are returned."""
        # d_min: [3,9] with 3 steps → 3 combos; first call fails
        gen = GridConfigGenerator(cfg_type="abcd", grid_steps=3)
        mock_cfg = MagicMock()
        with patch(_PATCH, side_effect=[ValueError("invalid"), mock_cfg, mock_cfg]):
            results = gen.generate(_ranges(d_min=[3, 9]))
        assert len(results) == 2

    def test_all_invalid_returns_empty_list(self) -> None:
        """Returns an empty list when every grid point fails validation."""
        gen = GridConfigGenerator(cfg_type="abcd", grid_steps=2)
        with patch(_PATCH, side_effect=ValueError("always invalid")):
            assert gen.generate(_ranges(d_min=[3, 9])) == []


class TestBorderlineConfigGeneratorInit:
    """Verify constructor behaviour."""

    def test_raises_for_unsupported_cfg_type(self) -> None:
        """Raises NotImplementedError when cfg_type is not 'abcd'."""
        with pytest.raises(NotImplementedError):
            BorderlineConfigGenerator(cfg_type="mabcd")

    def test_accepts_abcd_cfg_type(self) -> None:
        """Constructs successfully for cfg_type='abcd'."""
        gen = BorderlineConfigGenerator(cfg_type="abcd")
        assert gen.julia_config is not None


class TestBorderlineConfigGeneratorGenerate:
    """Verify that generate() enumerates all min/max corners."""

    def test_two_range_params_produce_four_combos(self) -> None:
        """Two range parameters yield 2^2 = 4 corner combinations."""
        gen = BorderlineConfigGenerator(cfg_type="abcd")
        with patch(_PATCH, return_value=MagicMock()):
            results = gen.generate(_ranges(d_min=[3, 9], d_max=[30, 90]))
        assert len(results) == 4

    def test_single_range_param_produces_two_combos(self) -> None:
        """One range parameter yields exactly 2 corners (min and max)."""
        gen = BorderlineConfigGenerator(cfg_type="abcd")
        with patch(_PATCH, return_value=MagicMock()):
            results = gen.generate(_ranges(d_min=[3, 9]))
        assert len(results) == 2

    def test_boundary_values_are_exact_range_endpoints(self) -> None:
        """Generated parameter values are exactly the min and max from the range."""
        gen = BorderlineConfigGenerator(cfg_type="abcd")
        captured: list[dict] = []
        with patch(_PATCH, side_effect=lambda p: captured.append(dict(p)) or MagicMock()):
            gen.generate(_ranges(d_min=[3, 9]))
        d_min_values = sorted({p["d_min"] for p in captured})
        assert d_min_values == [3, 9]

    def test_fixed_params_unchanged_across_all_combos(self) -> None:
        """Fixed parameters retain their original value in every corner config."""
        gen = BorderlineConfigGenerator(cfg_type="abcd")
        captured: list[dict] = []
        with patch(_PATCH, side_effect=lambda p: captured.append(dict(p)) or MagicMock()):
            gen.generate(_ranges(d_min=[3, 9]))
        assert all(p["n"] == 1000 for p in captured)
        assert all(p["xi"] == pytest.approx(0.3) for p in captured)

    def test_invalid_corners_are_dropped(self) -> None:
        """Corners that fail validation are excluded; valid ones are returned."""
        gen = BorderlineConfigGenerator(cfg_type="abcd")
        mock_cfg = MagicMock()
        with patch(_PATCH, side_effect=[ValueError("invalid"), mock_cfg]):
            results = gen.generate(_ranges(d_min=[3, 9]))
        assert len(results) == 1

    def test_all_invalid_returns_empty_list(self) -> None:
        """Returns an empty list when every corner fails validation."""
        gen = BorderlineConfigGenerator(cfg_type="abcd")
        with patch(_PATCH, side_effect=ValueError("always invalid")):
            assert gen.generate(_ranges(d_min=[3, 9])) == []

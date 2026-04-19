"""Unit tests for denormalise_abcd_config."""

import pytest

from dcba_data_set.config_generators import denormalise_abcd_config


def _base(n: int = 1000, **overrides: object) -> dict:
    """Return a minimal fractional param dict, with optional overrides."""
    params: dict = {
        "n": n,
        "d_min_frac": 0.005,
        "d_max_frac": 0.05,
        "c_min_frac": 0.01,
        "c_max_frac": 0.1,
        "nout_frac": 0.002,
        "t1": 2.5,
    }
    params.update(overrides)
    return params


class TestBasicConversion:
    """Verify that each fractional key is correctly converted to an absolute integer."""

    def test_d_min(self) -> None:
        """d_min_frac * n rounds to the expected integer."""
        result = denormalise_abcd_config(_base(n=1000, d_min_frac=0.005))
        assert result["d_min"] == 5

    def test_d_max(self) -> None:
        """d_max_frac * n rounds to the expected integer."""
        result = denormalise_abcd_config(_base(n=1000, d_max_frac=0.05))
        assert result["d_max"] == 50

    def test_c_min(self) -> None:
        """c_min_frac * n rounds to the expected integer."""
        result = denormalise_abcd_config(_base(n=1000, c_min_frac=0.01))
        assert result["c_min"] == 10

    def test_c_max(self) -> None:
        """c_max_frac * n rounds to the expected integer."""
        result = denormalise_abcd_config(_base(n=1000, c_max_frac=0.1))
        assert result["c_max"] == 100

    def test_nout(self) -> None:
        """nout_frac * n rounds to the expected integer."""
        result = denormalise_abcd_config(_base(n=1000, nout_frac=0.002))
        assert result["nout"] == 2


class TestFloorClamping:
    """Verify cross-parameter floors prevent trivially invalid configurations."""

    def test_d_max_floored_above_d_min(self) -> None:
        """When d_max_frac rounds below d_min, d_max is clamped to d_min + 1."""
        result = denormalise_abcd_config(_base(n=100, d_min_frac=0.05, d_max_frac=0.03))
        assert result["d_max"] >= result["d_min"] + 1

    def test_d_min_floor_at_one(self) -> None:
        """A near-zero d_min_frac yields d_min = 1, never less."""
        result = denormalise_abcd_config(_base(n=10, d_min_frac=0.00001))
        assert result["d_min"] == 1

    def test_c_min_floor_at_one(self) -> None:
        """A near-zero c_min_frac yields c_min = 1, never less."""
        result = denormalise_abcd_config(_base(n=10, c_min_frac=0.00001))
        assert result["c_min"] == 1

    def test_c_max_floored_above_c_min(self) -> None:
        """When c_max_frac rounds below c_min, c_max is clamped to c_min + 1."""
        result = denormalise_abcd_config(_base(n=100, c_min_frac=0.05, c_max_frac=0.03))
        assert result["c_max"] >= result["c_min"] + 1


class TestCMaxCap:
    """Verify that c_max is capped at n."""

    def test_c_max_capped_at_n(self) -> None:
        """A very large c_max_frac is capped so that c_max <= n."""
        result = denormalise_abcd_config(_base(n=100, c_max_frac=2.0))
        assert result["c_max"] <= 100


class TestNoutFracZero:
    """Verify that nout_frac = 0.0 produces exactly nout = 0."""

    def test_nout_zero(self) -> None:
        """nout_frac of 0.0 results in nout = 0."""
        result = denormalise_abcd_config(_base(nout_frac=0.0))
        assert result["nout"] == 0


class TestPassThrough:
    """Verify that non-fractional keys are preserved and fractional keys are removed."""

    def test_non_frac_keys_survive(self) -> None:
        """Fixed keys such as t1 are passed through unchanged."""
        result = denormalise_abcd_config(_base(t1=3.14))
        assert result["t1"] == pytest.approx(3.14)

    def test_no_frac_keys_in_output(self) -> None:
        """No ``*_frac`` keys remain in the output dict."""
        result = denormalise_abcd_config(_base())
        assert not any(k.endswith("_frac") for k in result)

    def test_n_survives(self) -> None:
        """The ``n`` key is retained in the output."""
        result = denormalise_abcd_config(_base(n=5000))
        assert result["n"] == 5000

"""Shared pytest fixtures and constants for the DCBA test suite."""

from pathlib import Path

import juliacall  # noqa: F401
import pytest

from dcba_data_set.graph_io import InstanceRecord, load_report

DATA_ROOT = Path(__file__).parent.parent / "data" / "test"
ABCD_REPORT = DATA_ROOT / "dataset_abcd" / "report.json"
MABCD_REPORT = DATA_ROOT / "dataset_mabcd" / "report.json"
KARATE_REPORT = DATA_ROOT / "dataset_karate" / "report.json"


@pytest.fixture(scope="module")
def abcd_records() -> list[InstanceRecord]:
    """Index the ABCD test dataset once for the entire module."""
    return load_report(ABCD_REPORT)


@pytest.fixture(scope="module")
def mabcd_records() -> list[InstanceRecord]:
    """Index the mABCD test dataset once for the entire module."""
    return load_report(MABCD_REPORT)


@pytest.fixture(scope="module")
def karate_records() -> list[InstanceRecord]:
    """Index the karate test dataset once for the entire module."""
    return load_report(KARATE_REPORT)

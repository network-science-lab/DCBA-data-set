"""Unit tests for TimedRunner."""

import time

import pytest

from dcba_data_set.timed_runner import TimedRunner


class _FastGenerator:
    """Dummy generator that completes immediately."""

    def __call__(self, config: object) -> None:
        pass


class _SlowGenerator:
    """Dummy generator that sleeps longer than any reasonable timeout."""

    def __call__(self, config: object) -> None:
        time.sleep(60)


class _ErrorGenerator:
    """Dummy generator that always raises."""

    def __call__(self, config: object) -> None:
        raise ValueError("generation failed")


class TestTimedRunner:
    """Verify TimedRunner behaviour for fast, slow, and failing generators."""

    def test_fast_generator_completes(self) -> None:
        """A generator that finishes quickly does not raise."""
        runner = TimedRunner(_FastGenerator, timeout=5)
        try:
            runner.run(object())
        finally:
            runner.close()

    def test_slow_generator_raises_timeout(self) -> None:
        """A generator that exceeds the timeout raises TimeoutError."""
        runner = TimedRunner(_SlowGenerator, timeout=1)
        with pytest.raises(TimeoutError):
            runner.run(object())
        runner.close()

    def test_worker_restarts_after_timeout(self) -> None:
        """A new live worker process replaces the stuck one after a timeout."""
        runner = TimedRunner(_SlowGenerator, timeout=1)
        try:
            original_pid = runner._worker.pid
            with pytest.raises(TimeoutError):
                runner.run(object())
            assert runner._worker.is_alive()
            assert runner._worker.pid != original_pid
        finally:
            runner.close()

    def test_generator_error_raised_as_runtime_error(self) -> None:
        """An exception inside the worker is re-raised as RuntimeError."""
        runner = TimedRunner(_ErrorGenerator, timeout=5)
        try:
            with pytest.raises(RuntimeError, match="generation failed"):
                runner.run(object())
        finally:
            runner.close()

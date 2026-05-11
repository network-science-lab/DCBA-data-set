"""Subprocess-based timeout wrapper for Julia-backed graph generators."""

import logging
import multiprocessing as mp
import multiprocessing.connection
from typing import Any

logger = logging.getLogger(__name__)


def _worker_loop(conn: mp.connection.Connection, generator_cls: type) -> None:
    """
    Run inside a worker subprocess; receive configs and return results via *conn*.

    Loops until a ``None`` sentinel is received. For each config, instantiates
    ``generator_cls`` and calls it, then sends back ``('ok', None)`` on success or
    ``('error', <message>)`` on exception.

    :param conn: Child end of a ``multiprocessing.Pipe``.
    :param generator_cls: Generator class to instantiate and call for each config.
    """
    while True:
        config = conn.recv()
        if config is None:
            break
        try:
            generator_cls()(config)
            conn.send(("ok", None))
        except Exception as e:  # noqa: BLE001
            conn.send(("error", str(e)))


class TimedRunner:
    """
    Persistent worker subprocess that runs a Julia-backed generator with a timeout.

    Julia initialises once when the worker process starts. If a generation call does
    not complete within ``timeout`` seconds, the worker is killed and restarted so
    subsequent calls can proceed normally.
    """

    def __init__(self, generator_cls: type, timeout: int) -> None:
        """
        Initialise the runner and start the worker subprocess.

        :param generator_cls: Generator class whose instances are callable with a config.
        :param timeout: Maximum seconds to wait for a single generation call.
        """
        self._generator_cls = generator_cls
        self._timeout = timeout
        self._parent_conn: mp.connection.Connection
        self._worker: mp.Process
        self._start_worker()

    def _start_worker(self) -> None:
        """Spawn a fresh worker process with Julia initialised."""
        parent_conn, child_conn = mp.get_context("spawn").Pipe()
        self._parent_conn = parent_conn
        self._worker = mp.get_context("spawn").Process(
            target=_worker_loop,
            args=(child_conn, self._generator_cls),
            daemon=True,
        )
        self._worker.start()
        child_conn.close()

    def run(self, config: Any) -> None:
        """
        Send *config* to the worker and wait up to ``timeout`` seconds for a result.

        Kills and restarts the worker on timeout, then raises ``TimeoutError``.
        Re-raises any exception reported by the worker as ``RuntimeError``.

        :param config: Configuration object accepted by the generator class.
        """
        self._parent_conn.send(config)
        if not self._parent_conn.poll(self._timeout):
            logger.warning("Generation timed out after %ds — restarting worker.", self._timeout)
            self._worker.terminate()
            self._worker.join()
            self._start_worker()
            raise TimeoutError(f"Generation timed out after {self._timeout}s.")

        status, message = self._parent_conn.recv()
        if status == "error":
            raise RuntimeError(message)

    def close(self) -> None:
        """Send the sentinel value to the worker and wait for it to exit cleanly."""
        try:
            self._parent_conn.send(None)
            self._worker.join(timeout=10)
        finally:
            if self._worker.is_alive():
                self._worker.terminate()
                self._worker.join()
            self._parent_conn.close()

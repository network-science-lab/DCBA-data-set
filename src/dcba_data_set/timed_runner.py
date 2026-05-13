"""Subprocess-based timeout wrapper for Julia-backed graph generators."""

import logging
import multiprocessing as mp
import multiprocessing.connection
import multiprocessing.synchronize
from typing import Any

logger = logging.getLogger(__name__)

_STARTUP_TIMEOUT = 600


def _worker_loop(
    conn: mp.connection.Connection,
    generator_cls: type,
    ready: mp.synchronize.Event,
) -> None:
    """
    Run inside a worker subprocess; receive configs and return results via *conn*.

    Signals *ready* once initialised, then loops until a ``None`` sentinel is
    received. For each config, instantiates ``generator_cls`` and calls it, then
    sends back ``('ok', None)`` on success or ``('error', <message>)`` on exception.

    :param conn: Child end of a ``multiprocessing.Pipe``.
    :param generator_cls: Generator class to instantiate and call for each config.
    :param ready: Event set once the worker has finished startup and is ready.
    """
    ready.set()
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

    Julia initialises once when the worker process starts. The generation timeout
    only begins after the worker signals that it is ready, so startup cost is never
    charged against the per-call timeout. If a generation call does not complete
    within ``timeout`` seconds, the worker is killed and restarted so subsequent
    calls can proceed normally.
    """

    def __init__(self, generator_cls: type, timeout: int, mp_context: str = "spawn") -> None:
        """
        Initialise the runner and start the worker subprocess.

        Blocks until the worker signals that it is ready (i.e. Julia has
        initialised), up to ``_STARTUP_TIMEOUT`` seconds.

        :param generator_cls: Generator class whose instances are callable with a config.
        :param timeout: Maximum seconds to wait for a single generation call.
        :param mp_context: Multiprocessing start method — ``"spawn"`` (default, safe for
            Julia) or ``"fork"`` (instant startup when Julia is already loaded in the
            parent, suitable for tests).
        """
        self._generator_cls = generator_cls
        self._timeout = timeout
        self._mp_context = mp_context
        self._parent_conn: mp.connection.Connection
        self._worker: mp.Process
        self._start_worker()

    def _start_worker(self) -> None:
        """Start a fresh worker process and block until it signals ready."""
        ctx = mp.get_context(self._mp_context)
        parent_conn, child_conn = ctx.Pipe()
        ready = ctx.Event()
        self._parent_conn = parent_conn
        self._worker = ctx.Process(
            target=_worker_loop,
            args=(child_conn, self._generator_cls, ready),
            daemon=True,
        )
        self._worker.start()
        child_conn.close()
        if not ready.wait(timeout=_STARTUP_TIMEOUT):
            self._worker.terminate()
            self._worker.join()
            raise RuntimeError(f"Worker process failed to start within {_STARTUP_TIMEOUT}s.")

    def run(self, config: Any) -> None:
        """
        Send *config* to the worker and wait up to ``timeout`` seconds for a result.

        Kills and restarts the worker on timeout or unexpected worker death, then raises
        ``TimeoutError``. Re-raises any exception reported by the worker as ``RuntimeError``.

        :param config: Configuration object accepted by the generator class.
        """
        try:
            self._parent_conn.send(config)
        except BrokenPipeError:
            logger.warning("Worker died unexpectedly before receiving config — restarting.")
            self._start_worker()
            raise TimeoutError("Worker died unexpectedly — treating as timeout.")

        if not self._parent_conn.poll(self._timeout):
            logger.warning("Generation timed out after %ds — restarting worker.", self._timeout)
            self._worker.terminate()
            self._worker.join()
            self._start_worker()
            raise TimeoutError(f"Generation timed out after {self._timeout}s.")

        try:
            status, message = self._parent_conn.recv()
        except EOFError:
            logger.warning("Worker died mid-execution — restarting.")
            self._start_worker()
            raise TimeoutError("Worker died mid-execution — treating as timeout.")

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

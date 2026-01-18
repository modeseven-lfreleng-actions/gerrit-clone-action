# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Matthew Watkins <mwatkins@linuxfoundation.org>

"""Concurrent execution utilities with graceful interruption handling."""

from __future__ import annotations

import logging
import signal
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Callable, Generator

from gerrit_clone.logging import get_logger

logger = get_logger(__name__)

# Global flag to suppress logging after interruption (with thread-safe access)
_logging_suppressed = False
_logging_lock = threading.Lock()


class SuppressLoggingFilter(logging.Filter):
    """Filter that suppresses all log messages when logging is disabled."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False to suppress messages when _logging_suppressed is True."""
        with _logging_lock:
            return not _logging_suppressed


def suppress_logging_after_interrupt() -> None:
    """Suppress all logging output after user interruption.

    This prevents the cascade of error messages from interrupted git operations
    that are no longer relevant after the user has cancelled the operation.

    Thread-safe: Uses a lock to safely modify the global flag.
    """
    global _logging_suppressed
    with _logging_lock:
        _logging_suppressed = True

    # Add filter to root logger to suppress all messages
    root_logger = logging.getLogger()
    suppress_filter = SuppressLoggingFilter()
    root_logger.addFilter(suppress_filter)

    # Also add to all handlers to ensure suppression
    for handler in root_logger.handlers:
        handler.addFilter(suppress_filter)


class _TrackedThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor wrapper that tracks submitted futures.

    This allows cancelling all pending futures on interrupt.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tracked_futures: set[Future[Any]] = set()

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:  # type: ignore[override]
        """Submit a callable and track the returned future."""
        future = super().submit(fn, *args, **kwargs)
        self._tracked_futures.add(future)
        return future

    def cancel_all_pending(self) -> int:
        """Cancel all pending futures.

        Returns:
            Number of futures successfully cancelled
        """
        cancelled_count = 0
        for future in self._tracked_futures:
            if future.cancel():
                cancelled_count += 1
        return cancelled_count


@contextmanager
def interruptible_executor(
    max_workers: int,
    thread_name_prefix: str = "worker",
) -> Generator[_TrackedThreadPoolExecutor, None, None]:
    """Create a ThreadPoolExecutor that handles KeyboardInterrupt gracefully.

    When interrupted with Ctrl+C, this context manager:
    1. Cancels all pending futures immediately
    2. Shuts down the executor without waiting for running tasks
    3. Re-raises KeyboardInterrupt for the caller to handle

    This ensures a single Ctrl+C press cleanly stops all workers.

    Args:
        max_workers: Maximum number of worker threads
        thread_name_prefix: Prefix for thread names (for debugging)

    Yields:
        _TrackedThreadPoolExecutor instance

    Raises:
        KeyboardInterrupt: Re-raised after clean shutdown

    Example:
        >>> with interruptible_executor(max_workers=16) as executor:
        ...     futures = [executor.submit(work_func, item) for item in items]
        ...     for future in as_completed(futures):
        ...         result = future.result()
    """
    executor = _TrackedThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix=thread_name_prefix,
    )

    interrupted = False

    try:
        yield executor

    except KeyboardInterrupt:
        interrupted = True

        # Cancel all pending futures immediately
        cancelled_count = executor.cancel_all_pending()
        logger.debug(f"Cancelled {cancelled_count} pending tasks")

        # Shutdown executor without waiting for running tasks
        executor.shutdown(wait=False, cancel_futures=True)

        # Suppress all further logging to avoid error spam from interrupted operations
        suppress_logging_after_interrupt()

        # Re-raise to allow caller to handle
        raise

    finally:
        if not interrupted:
            # Normal shutdown - wait for completion
            executor.shutdown(wait=True)


def handle_sigint_gracefully() -> None:
    """Configure SIGINT (Ctrl+C) to raise KeyboardInterrupt without delay.

    By default, Python may delay KeyboardInterrupt when threads are running.
    This ensures immediate interruption handling.

    Call this at the start of CLI commands that use threading.
    """
    def signal_handler(signum: int, frame: object) -> None:
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, signal_handler)

"""Optional progress reporting (``verbose=True``).

Uses ``tqdm`` when it is installed (it also renders in notebooks); otherwise falls back to
a throttled one-line report on stderr. When ``verbose`` is false everything here is a no-op.
"""

from __future__ import annotations

import sys
import time
from typing import Any

#: Seconds between two lines of the dependency-free fallback.
FALLBACK_INTERVAL = 5.0

try:  # pragma: no cover - exercised through the tests with and without tqdm
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None


def say(message: str, verbose: bool) -> None:
    """Print a stage message to stderr when verbose."""
    if verbose:
        print(f"[mutualvote] {message}", file=sys.stderr, flush=True)


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


class Progress:
    """A progress counter: ``update(n)``, ``set_info(text)``, ``close()``."""

    def update(self, n: int = 1) -> None:
        pass

    def set_info(self, text: str) -> None:
        pass

    def close(self) -> None:
        pass


class _TqdmProgress(Progress):
    def __init__(self, total: int | None, desc: str) -> None:
        assert _tqdm is not None
        self._bar: Any = _tqdm(total=total, desc=f"mutualvote {desc}", leave=True)

    def update(self, n: int = 1) -> None:
        self._bar.update(n)

    def set_info(self, text: str) -> None:
        self._bar.set_postfix_str(text, refresh=False)

    def close(self) -> None:
        self._bar.close()


class _TextProgress(Progress):
    def __init__(self, total: int | None, desc: str) -> None:
        self._total = total
        self._desc = desc
        self._count = 0
        self._info = ""
        self._start = self._last = time.monotonic()

    def _emit(self, now: float) -> None:
        if self._total:
            where = f"{self._count:,}/{self._total:,} ({100 * self._count / self._total:.1f}%)"
        else:
            where = f"{self._count:,}"
        info = f" | {self._info}" if self._info else ""
        say(f"{self._desc}: {where}, elapsed {format_duration(now - self._start)}{info}", True)
        self._last = now

    def update(self, n: int = 1) -> None:
        self._count += n
        now = time.monotonic()
        if now - self._last >= FALLBACK_INTERVAL:
            self._emit(now)

    def set_info(self, text: str) -> None:
        self._info = text

    def close(self) -> None:
        self._emit(time.monotonic())


def progress(total: int | None, desc: str, verbose: bool) -> Progress:
    """A progress counter for ``total`` steps; a silent no-op unless ``verbose``."""
    if not verbose:
        return Progress()
    if _tqdm is not None:
        return _TqdmProgress(total, desc)
    return _TextProgress(total, desc)

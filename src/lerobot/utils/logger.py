"""Unified logger abstraction for multi-backend metric logging.

Provides a Logger ABC, a MultiLogger composite that fans out to all backends,
and a make_logger factory that lazily imports backends based on config.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig

VALID_MODES = {"train", "eval"}
MAX_CONSECUTIVE_FAILURES = 10


class Logger(ABC):
    """Abstract base class for metric loggers."""

    @abstractmethod
    def log_dict(
        self,
        d: dict,
        step: int | None = None,
        mode: str = "train",
        custom_step_key: str | None = None,
    ) -> None:
        """Log a dictionary of scalar metrics."""

    @abstractmethod
    def log_video(self, video_path: str, step: int, mode: str = "train") -> None:
        """Log a video artifact."""

    @abstractmethod
    def log_policy(self, checkpoint_dir: Path) -> None:
        """Log a model checkpoint."""

    def close(self) -> None:  # noqa: B027
        """Clean up resources. Default no-op; override in backends that need cleanup."""


class MultiLogger(Logger):
    """Composite logger that fans out calls to multiple backends.

    Mode validation happens here once (not in each backend).
    Exception handling: catch Exception per-backend (never BaseException),
    log at warning level, and disable a backend after MAX_CONSECUTIVE_FAILURES.
    close() always calls close on ALL backends even if one raises.
    """

    def __init__(self, backends: list[Logger] | None = None) -> None:
        self._backends: list[Logger] = list(backends) if backends else []
        self._failure_counts: dict[int, int] = {}
        self._disabled: set[int] = set()

    def _active_backends(self) -> list[tuple[int, Logger]]:
        return [(i, b) for i, b in enumerate(self._backends) if i not in self._disabled]

    def _handle_failure(self, idx: int, backend: Logger, method: str, exc: Exception) -> None:
        count = self._failure_counts.get(idx, 0) + 1
        self._failure_counts[idx] = count
        backend_name = type(backend).__name__
        if count >= MAX_CONSECUTIVE_FAILURES:
            logging.error(
                f"{backend_name}.{method} failed {count} consecutive times; "
                f"disabling for this run. Last error: {exc}"
            )
            self._disabled.add(idx)
        else:
            logging.warning(f"{backend_name}.{method} failed ({count}/{MAX_CONSECUTIVE_FAILURES}): {exc}")

    def _reset_failure(self, idx: int) -> None:
        self._failure_counts.pop(idx, None)

    @staticmethod
    def _validate_mode(mode: str) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode {mode!r}. Must be one of {VALID_MODES}.")

    def log_dict(
        self,
        d: dict,
        step: int | None = None,
        mode: str = "train",
        custom_step_key: str | None = None,
    ) -> None:
        self._validate_mode(mode)
        for idx, backend in self._active_backends():
            try:
                backend.log_dict(d, step=step, mode=mode, custom_step_key=custom_step_key)
                self._reset_failure(idx)
            except Exception as exc:
                self._handle_failure(idx, backend, "log_dict", exc)

    def log_video(self, video_path: str, step: int, mode: str = "train") -> None:
        self._validate_mode(mode)
        for idx, backend in self._active_backends():
            try:
                backend.log_video(video_path, step, mode=mode)
                self._reset_failure(idx)
            except Exception as exc:
                self._handle_failure(idx, backend, "log_video", exc)

    def log_policy(self, checkpoint_dir: Path) -> None:
        for idx, backend in self._active_backends():
            try:
                backend.log_policy(checkpoint_dir)
                self._reset_failure(idx)
            except Exception as exc:
                self._handle_failure(idx, backend, "log_policy", exc)

    def close(self) -> None:
        """Close all backends. Always attempts all, even if one raises."""
        errors: list[tuple[str, Exception]] = []
        for _idx, backend in enumerate(self._backends):
            try:
                backend.close()
            except Exception as exc:
                backend_name = type(backend).__name__
                logging.warning(f"{backend_name}.close() failed: {exc}")
                errors.append((backend_name, exc))
        if errors:
            names = ", ".join(name for name, _ in errors)
            logging.warning(f"close() encountered errors in: {names}")


def make_logger(cfg: TrainPipelineConfig) -> MultiLogger:
    """Factory that builds a MultiLogger from config, lazily importing backends.

    Always returns a MultiLogger (possibly empty = no-op).
    Raises RuntimeError if a backend is enabled but its package is not installed.
    """
    backends: list[Logger] = []

    if cfg.wandb.enable:
        from lerobot.rl.wandb_utils import WandBLogger

        backends.append(WandBLogger(cfg))

    if cfg.mlflow.enable:
        try:
            import mlflow  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "MLflow logging is enabled (mlflow.enable=True) but the 'mlflow' package is not installed. "
                "Install it with: pip install 'lerobot[mlflow]'"
            ) from None

        from lerobot.utils.mlflow_logger import MLflowLogger

        backends.append(MLflowLogger(cfg))

    return MultiLogger(backends)

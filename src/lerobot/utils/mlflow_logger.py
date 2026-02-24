"""MLflow logging backend implementing the Logger ABC."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from lerobot.utils.logger import Logger

if TYPE_CHECKING:
    from lerobot.configs.train import TrainPipelineConfig

# MLflow has a 100-param batch limit and 500-char value limit
_MLFLOW_PARAM_BATCH_SIZE = 100
_MLFLOW_PARAM_VALUE_MAX_LEN = 500


class MLflowLogger(Logger):
    """Logger backend that sends metrics and artifacts to MLflow."""

    def __init__(self, cfg: TrainPipelineConfig) -> None:
        import mlflow

        self._mlflow = mlflow
        self._cfg = cfg.mlflow

        if self._cfg.tracking_uri:
            mlflow.set_tracking_uri(self._cfg.tracking_uri)

        mlflow.set_experiment(self._cfg.experiment_name)

        run_kwargs: dict = {}
        if self._cfg.run_name:
            run_kwargs["run_name"] = self._cfg.run_name
        if self._cfg.run_id:
            run_kwargs["run_id"] = self._cfg.run_id

        active_run = mlflow.start_run(**run_kwargs)
        self._run_id = active_run.info.run_id

        # Log config as flattened params (batch to avoid 100-param limit)
        self._log_config_params(cfg)

        logging.info(f"MLflow run started: {self._run_id} (experiment={self._cfg.experiment_name!r})")

    def _log_config_params(self, cfg: TrainPipelineConfig) -> None:
        """Log training config as flattened MLflow params in batches."""
        flat = self._flatten_dict(cfg.to_dict())
        # Truncate values and convert to strings
        params = {k: str(v)[:_MLFLOW_PARAM_VALUE_MAX_LEN] for k, v in flat.items()}

        # Batch to stay under MLflow's 100-param-per-call limit
        keys = list(params.keys())
        for i in range(0, len(keys), _MLFLOW_PARAM_BATCH_SIZE):
            batch = {k: params[k] for k in keys[i : i + _MLFLOW_PARAM_BATCH_SIZE]}
            self._mlflow.log_params(batch)

    @staticmethod
    def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
        """Flatten a nested dict into dot-separated keys."""
        items: list[tuple[str, object]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(MLflowLogger._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def log_dict(
        self,
        d: dict,
        step: int | None = None,
        mode: str = "train",
        custom_step_key: str | None = None,
    ) -> None:
        # Determine step value
        effective_step = step
        if custom_step_key is not None and custom_step_key in d:
            effective_step = int(d[custom_step_key])

        # Prefix keys with mode and filter to numeric values
        metrics = {}
        for k, v in d.items():
            if custom_step_key is not None and k == custom_step_key:
                continue
            if isinstance(v, (int, float)):
                metrics[f"{mode}/{k}"] = v

        if metrics:
            self._mlflow.log_metrics(metrics, step=effective_step)

    def log_video(self, video_path: str, step: int, mode: str = "train") -> None:
        artifact_path = f"{mode}/videos/step_{step}"
        self._mlflow.log_artifact(video_path, artifact_path=artifact_path)

    def log_policy(self, checkpoint_dir: Path) -> None:
        if self._cfg.disable_artifact:
            return
        artifact_path = f"checkpoints/{checkpoint_dir.name}"
        self._mlflow.log_artifacts(str(checkpoint_dir), artifact_path=artifact_path)

    def close(self) -> None:
        """End the MLflow run."""
        self._mlflow.end_run()

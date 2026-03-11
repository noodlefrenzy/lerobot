#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from pathlib import Path

from termcolor import colored

from lerobot.configs.train import TrainPipelineConfig


def _flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict[str, str]:
    """Flatten a nested dict into a single-level dict with joined keys.

    MLFlow params must be flat key-value string pairs with values capped at 500 characters.
    """
    items: dict[str, str] = {}
    for k, v in d.items():
        key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, key, sep))
        else:
            items[key] = str(v)[:500]
    return items


class MLFlowLogger:
    """A helper class to log objects using MLFlow."""

    def __init__(self, cfg: TrainPipelineConfig):
        self.cfg = cfg.mlflow
        self.log_dir = cfg.output_dir
        self.job_name = cfg.job_name
        self.env_fps = cfg.env.fps if cfg.env else None

        import mlflow

        if self.cfg.tracking_uri:
            mlflow.set_tracking_uri(self.cfg.tracking_uri)

        mlflow.set_experiment(self.cfg.experiment_name)

        # Resume an existing run or start a new one.
        if self.cfg.run_id:
            self._run = mlflow.start_run(run_id=self.cfg.run_id)
        else:
            run_name = self.cfg.run_name or self.job_name
            self._run = mlflow.start_run(run_name=run_name)

        run_id = self._run.info.run_id
        # Store the run id back to config so that resumed training can pick it up.
        cfg.mlflow.run_id = run_id

        # Log all hyperparameters. MLFlow caps individual param values at 500 chars
        # and total params at 200 by default (configurable server-side). We flatten
        # the config dict and truncate values to stay within limits.
        flat_params = _flatten_dict(cfg.to_dict())
        # mlflow.log_params has a batch limit of 100 params per call.
        param_items = list(flat_params.items())
        for i in range(0, len(param_items), 100):
            mlflow.log_params(dict(param_items[i : i + 100]))

        # Set tags for easy filtering in the MLFlow UI.
        tags = {"policy": cfg.policy.type, "seed": str(cfg.seed)}
        if cfg.dataset is not None:
            tags["dataset"] = cfg.dataset.repo_id
        if cfg.env is not None:
            tags["env"] = cfg.env.type
        mlflow.set_tags(tags)

        logging.info(colored("Logs will be synced with MLFlow.", "blue", attrs=["bold"]))
        logging.info(f"MLFlow run ID: {colored(run_id, 'yellow', attrs=['bold'])}")
        self._mlflow = mlflow

    def log_policy(self, checkpoint_dir: Path):
        """Log a checkpoint directory as an MLFlow artifact."""
        self._mlflow.log_artifacts(str(checkpoint_dir), artifact_path=f"checkpoints/{checkpoint_dir.name}")

    def log_dict(
        self, d: dict, step: int | None = None, mode: str = "train", custom_step_key: str | None = None
    ):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        if step is None and custom_step_key is None:
            raise ValueError("Either step or custom_step_key must be provided.")

        # When a custom step key is provided (e.g. for async RL), use its value as the step.
        if custom_step_key is not None and custom_step_key in d:
            step = int(d[custom_step_key])

        metrics: dict[str, float] = {}
        for k, v in d.items():
            if not isinstance(v, int | float):
                continue
            # Skip the custom step key itself to avoid logging it as a metric.
            if custom_step_key is not None and k == custom_step_key:
                continue
            metrics[f"{mode}/{k}"] = v

        if metrics:
            self._mlflow.log_metrics(metrics, step=step)

    def log_video(self, video_path: str, step: int, mode: str = "train"):
        if mode not in {"train", "eval"}:
            raise ValueError(mode)

        self._mlflow.log_artifact(video_path, artifact_path=f"{mode}_videos/step_{step}")

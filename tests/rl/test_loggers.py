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
"""Tests for WandB and MLFlow experiment loggers."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lerobot.configs.default import MLFlowConfig, WandBConfig
from lerobot.rl.mlflow_utils import _flatten_dict
from lerobot.rl.wandb_utils import cfg_to_group, get_safe_wandb_artifact_name

# ---------------------------------------------------------------------------
# Helpers / shared utilities
# ---------------------------------------------------------------------------


def _make_train_cfg(**overrides):
    """Build a minimal TrainPipelineConfig-like mock for logger tests."""
    cfg = MagicMock()
    cfg.wandb = WandBConfig(enable=True, project="test-project")
    cfg.mlflow = MLFlowConfig(enable=True, experiment_name="test-experiment")
    cfg.output_dir = Path("/tmp/test_output")
    cfg.job_name = "test_job"
    cfg.env = None
    cfg.resume = False
    cfg.seed = 42
    cfg.dataset = MagicMock()
    cfg.dataset.repo_id = "test/dataset"
    cfg.policy = MagicMock()
    cfg.policy.type = "act"
    cfg.to_dict.return_value = {"seed": 42, "policy": {"type": "act"}}
    for key, val in overrides.items():
        setattr(cfg, key, val)
    return cfg


def _make_mock_wandb():
    """Create a mock wandb module with the attributes WandBLogger expects."""
    mock = MagicMock()
    mock.run.id = "run-123"
    mock.run.get_url.return_value = "https://wandb.ai/run-123"
    return mock


def _make_mock_mlflow():
    """Create a mock mlflow module with the attributes MLFlowLogger expects."""
    mock = MagicMock()
    mock_run = MagicMock()
    mock_run.info.run_id = "mlflow-run-123"
    mock.start_run.return_value = mock_run
    return mock


# ---------------------------------------------------------------------------
# WandB helper function tests
# ---------------------------------------------------------------------------


class TestCfgToGroup:
    def test_returns_joined_string(self):
        cfg = _make_train_cfg()
        group = cfg_to_group(cfg)
        assert "policy:act" in group
        assert "seed:42" in group
        assert "dataset:test/dataset" in group

    def test_returns_list(self):
        cfg = _make_train_cfg()
        group = cfg_to_group(cfg, return_list=True)
        assert isinstance(group, list)
        assert "policy:act" in group

    def test_truncates_long_tags(self):
        cfg = _make_train_cfg()
        cfg.dataset.repo_id = "a" * 100
        group = cfg_to_group(cfg, return_list=True, truncate_tags=True, max_tag_length=64)
        for tag in group:
            assert len(tag) <= 64

    def test_no_env(self):
        cfg = _make_train_cfg(env=None)
        group = cfg_to_group(cfg, return_list=True)
        assert not any(tag.startswith("env:") for tag in group)

    def test_with_env(self):
        env = MagicMock()
        env.type = "pusht"
        cfg = _make_train_cfg(env=env)
        group = cfg_to_group(cfg, return_list=True)
        assert "env:pusht" in group


class TestGetSafeWandbArtifactName:
    def test_replaces_special_chars(self):
        assert get_safe_wandb_artifact_name("a:b/c") == "a_b_c"

    def test_no_special_chars(self):
        assert get_safe_wandb_artifact_name("abc_123") == "abc_123"


# ---------------------------------------------------------------------------
# MLFlow helper function tests
# ---------------------------------------------------------------------------


class TestFlattenDict:
    def test_flat_dict(self):
        result = _flatten_dict({"a": 1, "b": "hello"})
        assert result == {"a": "1", "b": "hello"}

    def test_nested_dict(self):
        result = _flatten_dict({"a": {"b": {"c": 1}}})
        assert result == {"a/b/c": "1"}

    def test_truncation(self):
        result = _flatten_dict({"key": "x" * 600})
        assert len(result["key"]) == 500

    def test_custom_separator(self):
        result = _flatten_dict({"a": {"b": 1}}, sep=".")
        assert "a.b" in result


# ---------------------------------------------------------------------------
# WandBLogger tests (mocked wandb SDK)
# ---------------------------------------------------------------------------


class TestWandBLogger:
    def test_init_calls_wandb_init(self):
        mock_wandb = _make_mock_wandb()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            from lerobot.rl.wandb_utils import WandBLogger

            cfg = _make_train_cfg()
            WandBLogger(cfg)

            mock_wandb.init.assert_called_once()
            kwargs = mock_wandb.init.call_args
            assert kwargs.kwargs["project"] == "test-project"

    def test_log_dict_with_step(self):
        mock_wandb = _make_mock_wandb()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            from lerobot.rl.wandb_utils import WandBLogger

            cfg = _make_train_cfg()
            logger = WandBLogger(cfg)
            logger.log_dict({"loss": 0.5, "lr": 1e-4}, step=10)

            # WandBLogger calls wandb.log once per key-value pair
            assert mock_wandb.log.call_count == 2

    def test_log_dict_rejects_invalid_mode(self):
        mock_wandb = _make_mock_wandb()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            from lerobot.rl.wandb_utils import WandBLogger

            cfg = _make_train_cfg()
            logger = WandBLogger(cfg)
            with pytest.raises(ValueError):
                logger.log_dict({"loss": 0.5}, step=10, mode="bad")

    def test_log_dict_requires_step_or_custom_key(self):
        mock_wandb = _make_mock_wandb()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            from lerobot.rl.wandb_utils import WandBLogger

            cfg = _make_train_cfg()
            logger = WandBLogger(cfg)
            with pytest.raises(ValueError):
                logger.log_dict({"loss": 0.5})

    def test_log_video(self):
        mock_wandb = _make_mock_wandb()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            from lerobot.rl.wandb_utils import WandBLogger

            cfg = _make_train_cfg()
            logger = WandBLogger(cfg)
            logger.log_video("/tmp/video.mp4", step=5, mode="eval")

            mock_wandb.Video.assert_called_once()
            mock_wandb.log.assert_called_once()

    def test_stores_run_id_on_config(self):
        mock_wandb = _make_mock_wandb()
        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            from lerobot.rl.wandb_utils import WandBLogger

            cfg = _make_train_cfg()
            WandBLogger(cfg)

            assert cfg.wandb.run_id == "run-123"


# ---------------------------------------------------------------------------
# MLFlowLogger tests (mocked mlflow SDK)
# ---------------------------------------------------------------------------


class TestMLFlowLogger:
    def test_init_starts_run_and_logs_params(self):
        mock_mlflow = _make_mock_mlflow()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            from lerobot.rl.mlflow_utils import MLFlowLogger

            cfg = _make_train_cfg()
            MLFlowLogger(cfg)

            mock_mlflow.set_experiment.assert_called_once_with("test-experiment")
            mock_mlflow.start_run.assert_called_once()
            mock_mlflow.log_params.assert_called()
            mock_mlflow.set_tags.assert_called_once()

    def test_init_sets_tracking_uri(self):
        mock_mlflow = _make_mock_mlflow()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            from lerobot.rl.mlflow_utils import MLFlowLogger

            cfg = _make_train_cfg()
            cfg.mlflow.tracking_uri = "http://localhost:5000"
            MLFlowLogger(cfg)

            mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")

    def test_init_resumes_run_by_id(self):
        mock_mlflow = _make_mock_mlflow()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            from lerobot.rl.mlflow_utils import MLFlowLogger

            cfg = _make_train_cfg()
            cfg.mlflow.run_id = "existing-run"
            mock_mlflow.start_run.return_value.info.run_id = "existing-run"
            MLFlowLogger(cfg)

            mock_mlflow.start_run.assert_called_once_with(run_id="existing-run")

    def test_init_stores_run_id_on_config(self):
        mock_mlflow = _make_mock_mlflow()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            from lerobot.rl.mlflow_utils import MLFlowLogger

            cfg = _make_train_cfg()
            MLFlowLogger(cfg)

            assert cfg.mlflow.run_id == "mlflow-run-123"

    def test_log_dict_with_step(self):
        mock_mlflow = _make_mock_mlflow()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            from lerobot.rl.mlflow_utils import MLFlowLogger

            cfg = _make_train_cfg()
            logger = MLFlowLogger(cfg)
            logger.log_dict({"loss": 0.5, "lr": 1e-4}, step=10)

            mock_mlflow.log_metrics.assert_called_once_with({"train/loss": 0.5, "train/lr": 1e-4}, step=10)

    def test_log_dict_with_custom_step_key(self):
        mock_mlflow = _make_mock_mlflow()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            from lerobot.rl.mlflow_utils import MLFlowLogger

            cfg = _make_train_cfg()
            logger = MLFlowLogger(cfg)
            logger.log_dict(
                {"loss": 0.5, "Optimization step": 42},
                mode="train",
                custom_step_key="Optimization step",
            )

            # Custom step key value (42) used as step; the key itself excluded from metrics.
            mock_mlflow.log_metrics.assert_called_once_with({"train/loss": 0.5}, step=42)

    def test_log_dict_skips_non_numeric(self):
        mock_mlflow = _make_mock_mlflow()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            from lerobot.rl.mlflow_utils import MLFlowLogger

            cfg = _make_train_cfg()
            logger = MLFlowLogger(cfg)
            logger.log_dict({"loss": 0.5, "video_paths": ["/a.mp4"]}, step=10)

            mock_mlflow.log_metrics.assert_called_once_with({"train/loss": 0.5}, step=10)

    def test_log_dict_rejects_invalid_mode(self):
        mock_mlflow = _make_mock_mlflow()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            from lerobot.rl.mlflow_utils import MLFlowLogger

            cfg = _make_train_cfg()
            logger = MLFlowLogger(cfg)
            with pytest.raises(ValueError):
                logger.log_dict({"loss": 0.5}, step=10, mode="bad")

    def test_log_dict_requires_step_or_custom_key(self):
        mock_mlflow = _make_mock_mlflow()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            from lerobot.rl.mlflow_utils import MLFlowLogger

            cfg = _make_train_cfg()
            logger = MLFlowLogger(cfg)
            with pytest.raises(ValueError):
                logger.log_dict({"loss": 0.5})

    def test_log_policy(self):
        mock_mlflow = _make_mock_mlflow()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            from lerobot.rl.mlflow_utils import MLFlowLogger

            cfg = _make_train_cfg()
            logger = MLFlowLogger(cfg)
            logger.log_policy(Path("/tmp/checkpoints/step_1000"))

            mock_mlflow.log_artifacts.assert_called_once_with(
                "/tmp/checkpoints/step_1000", artifact_path="checkpoints/step_1000"
            )

    def test_log_video(self):
        mock_mlflow = _make_mock_mlflow()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            from lerobot.rl.mlflow_utils import MLFlowLogger

            cfg = _make_train_cfg()
            logger = MLFlowLogger(cfg)
            logger.log_video("/tmp/eval.mp4", step=100, mode="eval")

            mock_mlflow.log_artifact.assert_called_once_with(
                "/tmp/eval.mp4", artifact_path="eval_videos/step_100"
            )

    def test_log_dict_eval_mode(self):
        mock_mlflow = _make_mock_mlflow()
        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            from lerobot.rl.mlflow_utils import MLFlowLogger

            cfg = _make_train_cfg()
            logger = MLFlowLogger(cfg)
            logger.log_dict({"avg_reward": 1.5}, step=10, mode="eval")

            mock_mlflow.log_metrics.assert_called_once_with({"eval/avg_reward": 1.5}, step=10)


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_mlflow_config_defaults(self):
        cfg = MLFlowConfig()
        assert cfg.enable is False
        assert cfg.tracking_uri is None
        assert cfg.experiment_name == "lerobot"
        assert cfg.run_name is None
        assert cfg.run_id is None

    def test_wandb_config_defaults(self):
        cfg = WandBConfig()
        assert cfg.enable is False
        assert cfg.project == "lerobot"
        assert cfg.disable_artifact is False

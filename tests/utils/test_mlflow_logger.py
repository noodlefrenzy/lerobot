"""Tests for MLflowLogger with fully mocked mlflow module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_mlflow():
    """Create a mock mlflow module."""
    mlflow = MagicMock()
    mlflow.start_run.return_value = MagicMock(info=MagicMock(run_id="test-run-123"))
    return mlflow


@pytest.fixture
def mock_cfg():
    """Create a mock TrainPipelineConfig."""
    cfg = MagicMock()
    cfg.mlflow.enable = True
    cfg.mlflow.disable_artifact = False
    cfg.mlflow.tracking_uri = None
    cfg.mlflow.experiment_name = "test-experiment"
    cfg.mlflow.run_name = None
    cfg.mlflow.run_id = None
    cfg.to_dict.return_value = {"policy": {"type": "act"}, "batch_size": 8}
    return cfg


@pytest.fixture
def mlflow_logger(mock_mlflow, mock_cfg):
    """Create an MLflowLogger with mocked mlflow."""
    with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
        from lerobot.utils.mlflow_logger import MLflowLogger

        logger = MLflowLogger.__new__(MLflowLogger)
        logger._mlflow = mock_mlflow
        logger._cfg = mock_cfg.mlflow
        logger._run_id = "test-run-123"
        return logger


class TestMLflowLoggerInit:
    def test_sets_experiment_and_starts_run(self, mock_mlflow, mock_cfg):
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            from lerobot.utils.mlflow_logger import MLflowLogger

            logger = MLflowLogger(mock_cfg)

            mock_mlflow.set_experiment.assert_called_once_with("test-experiment")
            mock_mlflow.start_run.assert_called_once()
            assert logger._run_id == "test-run-123"

    def test_sets_tracking_uri_when_provided(self, mock_mlflow, mock_cfg):
        mock_cfg.mlflow.tracking_uri = "http://mlflow.example.com"
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            from lerobot.utils.mlflow_logger import MLflowLogger

            MLflowLogger(mock_cfg)
            mock_mlflow.set_tracking_uri.assert_called_once_with("http://mlflow.example.com")

    def test_passes_run_name_and_run_id(self, mock_mlflow, mock_cfg):
        mock_cfg.mlflow.run_name = "my-run"
        mock_cfg.mlflow.run_id = "existing-run-id"
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            from lerobot.utils.mlflow_logger import MLflowLogger

            MLflowLogger(mock_cfg)
            mock_mlflow.start_run.assert_called_once_with(run_name="my-run", run_id="existing-run-id")

    def test_logs_config_params_in_batches(self, mock_mlflow, mock_cfg):
        mock_cfg.to_dict.return_value = {f"key_{i}": i for i in range(150)}
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            from lerobot.utils.mlflow_logger import MLflowLogger

            MLflowLogger(mock_cfg)
            # 150 params should be batched into 2 calls (100 + 50)
            assert mock_mlflow.log_params.call_count == 2


class TestMLflowLoggerLogDict:
    def test_prefixes_keys_with_mode(self, mlflow_logger, mock_mlflow):
        mlflow_logger.log_dict({"loss": 0.5, "lr": 1e-4}, step=10, mode="train")
        mock_mlflow.log_metrics.assert_called_once_with(
            {"train/loss": 0.5, "train/lr": 1e-4}, step=10
        )

    def test_eval_mode_prefix(self, mlflow_logger, mock_mlflow):
        mlflow_logger.log_dict({"reward": 1.0}, step=5, mode="eval")
        mock_mlflow.log_metrics.assert_called_once_with({"eval/reward": 1.0}, step=5)

    def test_custom_step_key_extracts_step(self, mlflow_logger, mock_mlflow):
        mlflow_logger.log_dict(
            {"loss": 0.3, "Optimization step": 42},
            mode="train",
            custom_step_key="Optimization step",
        )
        mock_mlflow.log_metrics.assert_called_once_with({"train/loss": 0.3}, step=42)

    def test_skips_non_numeric_values(self, mlflow_logger, mock_mlflow):
        mlflow_logger.log_dict({"loss": 0.5, "name": "test", "count": 10}, step=1)
        mock_mlflow.log_metrics.assert_called_once_with(
            {"train/loss": 0.5, "train/count": 10}, step=1
        )

    def test_empty_metrics_no_call(self, mlflow_logger, mock_mlflow):
        mlflow_logger.log_dict({"name": "only_strings"}, step=1)
        mock_mlflow.log_metrics.assert_not_called()


class TestMLflowLoggerLogVideo:
    def test_logs_artifact_with_correct_path(self, mlflow_logger, mock_mlflow):
        mlflow_logger.log_video("/tmp/video.mp4", step=100, mode="eval")
        mock_mlflow.log_artifact.assert_called_once_with(
            "/tmp/video.mp4", artifact_path="eval/videos/step_100"
        )


class TestMLflowLoggerLogPolicy:
    def test_logs_artifacts_directory(self, mlflow_logger, mock_mlflow):
        checkpoint_dir = Path("/tmp/checkpoints/000100")
        mlflow_logger.log_policy(checkpoint_dir)
        mock_mlflow.log_artifacts.assert_called_once_with(
            str(checkpoint_dir), artifact_path="checkpoints/000100"
        )

    def test_disable_artifact_skips_log_policy(self, mlflow_logger, mock_mlflow):
        mlflow_logger._cfg.disable_artifact = True
        mlflow_logger.log_policy(Path("/tmp/checkpoints/000100"))
        mock_mlflow.log_artifacts.assert_not_called()


class TestMLflowLoggerClose:
    def test_calls_end_run(self, mlflow_logger, mock_mlflow):
        mlflow_logger.close()
        mock_mlflow.end_run.assert_called_once()

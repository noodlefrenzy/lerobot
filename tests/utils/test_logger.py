"""Tests for Logger ABC, MultiLogger composite, and make_logger factory."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lerobot.utils.logger import Logger, MultiLogger, make_logger


class DummyLogger(Logger):
    """Concrete Logger for testing."""

    def __init__(self):
        self.logged_dicts = []
        self.logged_videos = []
        self.logged_policies = []
        self.closed = False

    def log_dict(self, d, step=None, mode="train", custom_step_key=None):
        self.logged_dicts.append({"d": d, "step": step, "mode": mode, "custom_step_key": custom_step_key})

    def log_video(self, video_path, step, mode="train"):
        self.logged_videos.append({"video_path": video_path, "step": step, "mode": mode})

    def log_policy(self, checkpoint_dir):
        self.logged_policies.append(checkpoint_dir)

    def close(self):
        self.closed = True


class FailingLogger(Logger):
    """Logger that raises on every call."""

    def log_dict(self, d, step=None, mode="train", custom_step_key=None):
        raise RuntimeError("log_dict failed")

    def log_video(self, video_path, step, mode="train"):
        raise RuntimeError("log_video failed")

    def log_policy(self, checkpoint_dir):
        raise RuntimeError("log_policy failed")

    def close(self):
        raise RuntimeError("close failed")


class TestMultiLogger:
    def test_empty_logger_is_noop(self):
        ml = MultiLogger()
        # Should not raise
        ml.log_dict({"loss": 0.5}, step=1)
        ml.log_video("/tmp/vid.mp4", step=1)
        ml.log_policy(Path("/tmp/checkpoint"))
        ml.close()

    def test_fans_out_to_all_backends(self):
        b1 = DummyLogger()
        b2 = DummyLogger()
        ml = MultiLogger([b1, b2])

        ml.log_dict({"loss": 0.5}, step=10)
        ml.log_video("/tmp/vid.mp4", step=10, mode="eval")
        ml.log_policy(Path("/tmp/ckpt"))

        assert len(b1.logged_dicts) == 1
        assert len(b2.logged_dicts) == 1
        assert b1.logged_dicts[0]["step"] == 10
        assert b2.logged_dicts[0]["step"] == 10

        assert len(b1.logged_videos) == 1
        assert b1.logged_videos[0]["mode"] == "eval"

        assert len(b1.logged_policies) == 1
        assert len(b2.logged_policies) == 1

    def test_one_failure_does_not_block_others(self):
        failing = FailingLogger()
        good = DummyLogger()
        ml = MultiLogger([failing, good])

        ml.log_dict({"loss": 0.5}, step=1)
        assert len(good.logged_dicts) == 1

        ml.log_video("/tmp/vid.mp4", step=1)
        assert len(good.logged_videos) == 1

        ml.log_policy(Path("/tmp/ckpt"))
        assert len(good.logged_policies) == 1

    def test_close_called_on_all_even_if_one_raises(self):
        failing = FailingLogger()
        good = DummyLogger()
        ml = MultiLogger([failing, good])

        ml.close()
        assert good.closed is True

    def test_mode_validation_rejects_invalid_modes(self):
        ml = MultiLogger()
        with pytest.raises(ValueError, match="Invalid mode"):
            ml.log_dict({"loss": 0.5}, step=1, mode="invalid")

        with pytest.raises(ValueError, match="Invalid mode"):
            ml.log_video("/tmp/vid.mp4", step=1, mode="test")

    def test_failure_counter_disables_backend_after_threshold(self):
        failing = FailingLogger()
        good = DummyLogger()
        ml = MultiLogger([failing, good])

        # Call log_dict 10 times to trigger disable threshold
        for i in range(10):
            ml.log_dict({"loss": float(i)}, step=i)

        # Both should have been called for the first 10 calls
        assert len(good.logged_dicts) == 10

        # After 10 failures, failing backend should be disabled
        assert 0 in ml._disabled

        # Call again — only good should receive it
        ml.log_dict({"loss": 99.0}, step=99)
        assert len(good.logged_dicts) == 11

    def test_successful_call_resets_failure_count(self):
        class SometimesFailing(Logger):
            def __init__(self):
                self.call_count = 0

            def log_dict(self, d, step=None, mode="train", custom_step_key=None):
                self.call_count += 1
                # Fail on odd calls
                if self.call_count % 2 == 1:
                    raise RuntimeError("intermittent failure")

            def log_video(self, video_path, step, mode="train"):
                pass

            def log_policy(self, checkpoint_dir):
                pass

        backend = SometimesFailing()
        ml = MultiLogger([backend])

        # Alternate failures won't accumulate because successes reset the count
        for i in range(20):
            ml.log_dict({"loss": float(i)}, step=i)

        # Should not be disabled (failures never reach 10 consecutive)
        assert 0 not in ml._disabled


class TestMakeLogger:
    def _make_cfg(self, wandb_enable=False, mlflow_enable=False):
        """Build a minimal mock config."""
        cfg = MagicMock()
        cfg.wandb.enable = wandb_enable
        cfg.wandb.project = "test"
        cfg.mlflow.enable = mlflow_enable
        return cfg

    def test_returns_empty_multi_logger_when_nothing_enabled(self):
        cfg = self._make_cfg()
        result = make_logger(cfg)
        assert isinstance(result, MultiLogger)
        assert len(result._backends) == 0

    def test_raises_runtime_error_when_mlflow_enabled_but_not_installed(self):
        cfg = self._make_cfg(mlflow_enable=True)
        with patch.dict("sys.modules", {"mlflow": None}), pytest.raises(RuntimeError, match="mlflow.*not installed"):
            make_logger(cfg)

    def test_wandb_backend_created_when_enabled(self):
        cfg = self._make_cfg(wandb_enable=True)
        with patch("lerobot.rl.wandb_utils.WandBLogger") as mock_wandb:
            mock_wandb.return_value = DummyLogger()
            result = make_logger(cfg)
            assert len(result._backends) == 1
            mock_wandb.assert_called_once_with(cfg)

    def test_mlflow_backend_created_when_enabled(self):
        cfg = self._make_cfg(mlflow_enable=True)
        with (
            patch.dict("sys.modules", {"mlflow": MagicMock()}),
            patch("lerobot.utils.mlflow_logger.MLflowLogger") as mock_mlflow,
        ):
            mock_mlflow.return_value = DummyLogger()
            result = make_logger(cfg)
            assert len(result._backends) == 1
            mock_mlflow.assert_called_once_with(cfg)

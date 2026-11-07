"""
Unit tests for StageVRAMTracker profiler.
Verifies CPU fallback, disabled mode, mocked CUDA profiling calculations, and exception handling.
"""
import unittest
from unittest.mock import patch, MagicMock

import torch
from mssearch.stages.utils_profiler import StageVRAMTracker
from mssearch.config import MSSearchConfig


class TestStageVRAMTracker(unittest.TestCase):

    def test_disabled_mode(self):
        tracker = StageVRAMTracker("test_disabled", enabled=False)
        self.assertFalse(tracker.enabled)
        with tracker:
            pass
        self.assertEqual(tracker.stats, {})

    @patch("torch.cuda.is_available", return_value=False)
    def test_cpu_mode_graceful_pass(self, mock_cuda_avail):
        tracker = StageVRAMTracker("test_cpu", enabled=True)
        self.assertFalse(tracker.enabled)
        with tracker:
            pass
        self.assertEqual(tracker.stats, {})

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.synchronize")
    @patch("torch.cuda.reset_peak_memory_stats")
    @patch("torch.cuda.memory_allocated")
    @patch("torch.cuda.memory_reserved")
    @patch("torch.cuda.max_memory_allocated")
    @patch("torch.cuda.max_memory_reserved")
    def test_mocked_cuda_profiling(
        self,
        mock_max_res,
        mock_max_alloc,
        mock_res,
        mock_alloc,
        mock_reset,
        mock_sync,
        mock_avail,
    ):
        # 1000 MB at start, 1500 MB at end (delta = +500 MB)
        # Peak allocated = 2500 MB, Peak reserved = 3000 MB
        mock_alloc.side_effect = [1000 * 1024 * 1024, 1500 * 1024 * 1024]
        mock_res.side_effect = [2000 * 1024 * 1024, 2500 * 1024 * 1024]
        mock_max_alloc.return_value = 2500 * 1024 * 1024
        mock_max_res.return_value = 3000 * 1024 * 1024

        tracker = StageVRAMTracker("test_stage", enabled=True, device="cuda:0", verbose=False)
        with tracker:
            # Simulate work
            pass

        self.assertEqual(tracker.stats["start_allocated_mb"], 1000.0)
        self.assertEqual(tracker.stats["end_allocated_mb"], 1500.0)
        self.assertEqual(tracker.stats["delta_allocated_mb"], 500.0)
        self.assertEqual(tracker.stats["peak_allocated_mb"], 2500.0)
        self.assertEqual(tracker.stats["peak_reserved_mb"], 3000.0)
        mock_reset.assert_called_once_with("cuda:0")

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.synchronize", side_effect=RuntimeError("CUDA device error"))
    def test_cuda_exception_resilience(self, mock_sync, mock_avail):
        tracker = StageVRAMTracker("test_error", enabled=True)
        # Should not raise exception
        with tracker:
            pass
        self.assertEqual(tracker.stats, {})

    def test_config_enable_vram_profiling_default(self):
        cfg = MSSearchConfig()
        self.assertTrue(hasattr(cfg, "enable_vram_profiling"))
        self.assertTrue(cfg.enable_vram_profiling)


if __name__ == "__main__":
    unittest.main()


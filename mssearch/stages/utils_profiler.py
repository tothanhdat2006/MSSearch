"""
utils_profiler.py: Lightweight Stage-by-Stage VRAM Profiling Context Manager.
Tracks entry, exit, delta, and peak GPU memory allocated/reserved via PyTorch CUDA APIs.
"""
from typing import Dict, Any, Optional
import torch

__all__ = ["StageVRAMTracker"]


class StageVRAMTracker:
    """
    Context manager to profile VRAM allocation, reservation, peak, and net leak
    for an individual pipeline stage. Falls back cleanly to a no-op on CPU or when disabled.
    """

    def __init__(
        self,
        stage_name: str,
        enabled: bool = True,
        device: str = "cuda:0",
        verbose: bool = False,
    ):
        self.stage_name = stage_name
        self.device = device
        self.verbose = verbose
        self.enabled = enabled and torch.cuda.is_available()
        self.stats: Dict[str, Any] = {}
        self.start_allocated: float = 0.0
        self.start_reserved: float = 0.0

    def __enter__(self):
        if self.enabled:
            try:
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
                self.start_allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
                self.start_reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
            except Exception:
                self.enabled = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            try:
                torch.cuda.synchronize(self.device)
                end_allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
                end_reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
                peak_allocated = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                peak_reserved = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)

                self.stats = {
                    "start_allocated_mb": round(self.start_allocated, 1),
                    "end_allocated_mb": round(end_allocated, 1),
                    "delta_allocated_mb": round(end_allocated - self.start_allocated, 1),
                    "peak_allocated_mb": round(peak_allocated, 1),
                    "peak_reserved_mb": round(peak_reserved, 1),
                }

                if self.verbose:
                    print(
                        f"[VRAM Profiler][{self.stage_name}] "
                        f"Peak Alloc: {peak_allocated:,.1f} MB | "
                        f"Peak Reserved: {peak_reserved:,.1f} MB | "
                        f"Net Delta: {self.stats['delta_allocated_mb']:+,.1f} MB"
                    )
            except Exception:
                self.stats = {}


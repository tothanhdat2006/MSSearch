"""
Benchmark Sample Data Structures for STASH-Search Minimal.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class BenchmarkSample:
    """Standardized benchmark evaluation sample across STASH-Bench, Vstar, and HR-Bench."""
    sample_id: str
    question: str
    candidate_image_paths: List[str]
    gt_answer: str
    gt_image_idx: Optional[int] = 0
    options_str: Optional[str] = None
    category: str = "default"
    target_hint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

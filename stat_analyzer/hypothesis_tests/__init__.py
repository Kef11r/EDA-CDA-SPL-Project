from .detectors import detect_type, suggest_tests
from stat_analyzer.hypothesis_tests.runner import (
    run_or_suggest,
    run_test_by_name,
    interpret_result,
    run_all_presets
)
from .runner import load_custom_test
from .runner import TEST_FUNCTIONS
from .presets import HYPOTHESES

__all__ = [
    "detect_type",
    "suggest_tests",
    "run_or_suggest",
    "run_test_by_name",
    "interpret_result",
    "run_all_presets",
    "HYPOTHESES",
]

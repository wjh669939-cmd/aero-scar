"""Harness-owned exception types."""


class HarnessError(RuntimeError):
    """Base error for the closed-loop harness."""


class LLMError(HarnessError):
    """Raised when a model API request cannot be completed."""


class AxisLockError(HarnessError):
    """Raised when a trial edits files outside the active research axis."""


class EvaluatorError(HarnessError):
    """Raised when the independent evaluator rejects a submission."""

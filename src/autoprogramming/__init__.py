"""AutoProgramming — define your inputs and outputs; a coding agent finds the best implementation."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

from .budget import Budget
from .errors import (
    AutoProgrammingError,
    BootstrapModeError,
    BudgetError,
    BudgetExceededError,
    CandidateError,
    DataDisciplineError,
    FinalizedError,
    MemorizationWarning,
    MetricChangedError,
    MetricNotApprovedError,
    NotOptimizedError,
    RunnerError,
    SchemaError,
    ValReliabilityWarning,
    WorkspaceError,
)
from .harness import attach
from .program import Program, program

try:
    __version__ = _dist_version("autoprogramming")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "program",
    "Program",
    "Budget",
    "attach",
    "AutoProgrammingError",
    "SchemaError",
    "DataDisciplineError",
    "MetricNotApprovedError",
    "MetricChangedError",
    "BudgetError",
    "BudgetExceededError",
    "BootstrapModeError",
    "NotOptimizedError",
    "FinalizedError",
    "WorkspaceError",
    "CandidateError",
    "RunnerError",
    "ValReliabilityWarning",
    "MemorizationWarning",
    "__version__",
]

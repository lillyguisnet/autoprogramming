"""Budgets are explicit and have units.

``Budget(dollars=20)``, ``Budget(eval_calls=2000)``, or ``Budget(minutes=30)``
— combinable; optimization stops when the first limit is hit. Evaluation cost
counts against the budget. There is no default; you must say what you're
willing to spend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .errors import BudgetError, BudgetExceededError


@dataclass(frozen=True)
class Budget:
    dollars: float | None = None
    eval_calls: int | None = None
    minutes: float | None = None

    def __post_init__(self) -> None:
        if self.dollars is None and self.eval_calls is None and self.minutes is None:
            raise BudgetError(
                "Budget has no default — pass at least one of dollars=, "
                "eval_calls=, or minutes= to say what you're willing to spend."
            )
        for name in ("dollars", "eval_calls", "minutes"):
            v = getattr(self, name)
            if v is not None and v <= 0:
                raise BudgetError(f"Budget {name} must be positive, got {v!r}.")

    def to_dict(self) -> dict:
        return {"dollars": self.dollars, "eval_calls": self.eval_calls, "minutes": self.minutes}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BudgetLedger:
    """Workspace-backed spend tracking (budget.json).

    Every eval run charges here. ``check()`` raises before work starts once
    any limit is hit; ``finalize()`` deliberately does not check — the
    one-time test evaluation happens even on an exhausted budget, it just
    keeps being recorded.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.exists():
            self._state = json.loads(self.path.read_text())
        else:
            self._state = None

    # ------------------------------------------------------------- lifecycle

    @classmethod
    def start(cls, path: Path, budget: Budget) -> "BudgetLedger":
        """Set (or replace) the limits. Prior spend is preserved — money
        already burned in this workspace stays spent; the minutes clock
        restarts because a new optimization run is starting."""
        ledger = cls(path)
        spent = (ledger._state or {}).get("spent", {"dollars": 0.0, "eval_calls": 0})
        ledger._state = {
            "limits": budget.to_dict(),
            "spent": spent,
            "started_at": _utcnow().isoformat(),
        }
        ledger._save()
        return ledger

    def _require_state(self) -> dict:
        if self._state is None:
            raise BudgetError(
                f"No budget has been set for this workspace ({self.path}); "
                f"optimize() records one via BudgetLedger.start()."
            )
        return self._state

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._state, indent=2) + "\n")

    # ------------------------------------------------------------ accounting

    def charge(self, eval_calls: int = 0, dollars: float = 0.0) -> None:
        state = self._require_state()
        state["spent"]["eval_calls"] += int(eval_calls)
        state["spent"]["dollars"] += float(dollars)
        self._save()

    @property
    def limits(self) -> dict:
        return dict(self._require_state()["limits"])

    @property
    def spent(self) -> dict:
        state = self._require_state()
        return {
            "dollars": state["spent"]["dollars"],
            "eval_calls": state["spent"]["eval_calls"],
            "minutes": self.elapsed_minutes(),
        }

    def elapsed_minutes(self) -> float:
        state = self._require_state()
        started = datetime.fromisoformat(state["started_at"])
        return (_utcnow() - started).total_seconds() / 60.0

    def remaining(self) -> dict:
        """Remaining headroom per unit; None where no limit was set."""
        limits, spent = self.limits, self.spent
        return {
            unit: (None if limits[unit] is None else max(0, limits[unit] - spent[unit]))
            for unit in ("dollars", "eval_calls", "minutes")
        }

    def exhausted(self) -> str | None:
        """Name of the first exhausted limit, or None."""
        limits, spent = self.limits, self.spent
        for unit in ("dollars", "eval_calls", "minutes"):
            if limits[unit] is not None and spent[unit] >= limits[unit]:
                return unit
        return None

    def check(self) -> None:
        unit = self.exhausted()
        if unit is not None:
            limits, spent = self.limits, self.spent
            raise BudgetExceededError(
                f"Budget exhausted: {unit} limit {limits[unit]} reached "
                f"(spent {spent[unit]:.4g}). Optimization stops at the first "
                f"limit hit; run finalize() to evaluate the top candidates on "
                f"test and activate a winner, or optimize() again with a new budget."
            )

from enum import StrEnum

from typing import Any

from sqlalchemy import Case, case


class PriorityType(StrEnum):
    """Stored as text (``String(16)``) so the values must be stable strings."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


#: Semantic ordering for ``priority``. The column holds text, so ordering by it
#: directly would be alphabetical ("high" < "low" < "medium" < "urgent").
PRIORITY_RANK: dict[str, int] = {
    PriorityType.LOW.value: 1,
    PriorityType.MEDIUM.value: 2,
    PriorityType.HIGH.value: 3,
    PriorityType.URGENT.value: 4,
}


def priority_rank(column: Any) -> Case:
    """CASE expression that turns a priority column into a sortable rank."""
    return case(PRIORITY_RANK, value=column, else_=0)

# mypy: disable-error-code="type-arg"
"""
Boolean Logic implementations for various monad types.

Each class is a concrete implementation for a specific monad type,
allowing proper type checking with the returns library's HKT system.
"""

from typing import Any, Generic, TypeVar

from returns.interfaces.container import Container1
from returns.primitives.hkt import Kind1

from muller.logics.aggr2monblat import (
    TwoMonBLat,
)

# =============================================================================
# Identity Monad Boolean Logic (Classical/Deterministic)
# =============================================================================

_MonadType = TypeVar("_MonadType", bound=Container1[Any])


class BooleanLogic(
    Generic[_MonadType], TwoMonBLat[_MonadType, bool]
):
    """Classical boolean logic with Identity monad (deterministic)."""

    def top(self) -> Kind1[_MonadType, bool]:
        return self.monad_from_value(True)

    def bottom(self) -> Kind1[_MonadType, bool]:
        return self.monad_from_value(False)

    def neg(self, a: Kind1[_MonadType, bool]) -> Kind1[_MonadType, bool]:
        return a.map(lambda x: not x)

    def conjunction(
        self, a: Kind1[_MonadType, bool], b: Kind1[_MonadType, bool]
    ) -> Kind1[_MonadType, bool]:
        return a.bind(lambda x: self.bottom() if not x else b)

    def disjunction(
        self, a: Kind1[_MonadType, bool], b: Kind1[_MonadType, bool]
    ) -> Kind1[_MonadType, bool]:
        return a.bind(lambda x: self.top() if x else b)


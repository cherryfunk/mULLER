from __future__ import annotations

from typing import Any, Callable, Iterable, TypeVar, final

from returns.interfaces.container import Container1
from returns.primitives.container import BaseContainer
from returns.primitives.hkt import Kind1, SupportsKind1, dekind

from muller.monad.base import monad_apply

_ValueType = TypeVar("_ValueType")
_NewValueType = TypeVar("_NewValueType")


@final
class Powerset(
    BaseContainer,
    SupportsKind1["Powerset", _ValueType],  # type: ignore[type-arg]
    Container1[_ValueType],
):
    """
    Powerset Monad for Non-deterministic Computation

    Represents non-deterministic computations as sets of possible values.
    """

    def __init__(self, values: Iterable[_ValueType]):
        """
        Initialize a powerset computation.

        Args:
            values: Set, frozenset, or list of values representing possible outcomes
        """
        super().__init__(frozenset(values))

    def bind(
        self,
        function: Callable[[_ValueType], Kind1["Powerset", _NewValueType]],  # type: ignore[type-arg]
    ) -> Powerset[_NewValueType]:
        """
        Monadic bind operation (Kleisli extension).

        Apply a function that returns a powerset to each value in this powerset,
        then take the union of all results.

        Args:
            function: Function from value to powerset

        Returns:
            New powerset containing union of all results
        """
        result = set.union(
            set(),
            *[dekind(function(value))._inner_value for value in self._inner_value],
        )

        return Powerset(result)

    def map(
        self,
        function: Callable[[_ValueType], _NewValueType],
    ) -> Powerset[_NewValueType]:
        """
        Apply a function to all values in the powerset.

        Args:
            function: Function to apply to values

        Returns:
            New powerset with function applied to all values
        """
        return Powerset({function(value) for value in self._inner_value})

    def __repr__(self) -> str:
        sorted_values = sorted(self._inner_value, key=str)
        return f"Powerset({set(sorted_values)})"

    def __eq__(self, other: object) -> bool:
        if type(self) != type(other):  # noqa: WPS516, E721
            return False
        return bool(
            self._inner_value == other._inner_value,  # type: ignore # noqa: SLF001
        )

    def __hash__(self) -> int:
        return hash(self._inner_value)

    apply = monad_apply

    @classmethod
    def from_value(cls, inner_value: _NewValueType) -> "Powerset[_NewValueType]":
        return Powerset({inner_value})


# Convenience functions for creating common powersets


def singleton(value: _ValueType) -> Powerset[_ValueType]:
    """Create a powerset with a single value."""
    return Powerset.from_value(value)


def empty() -> Powerset[Any]:
    """Create an empty powerset (failed computation)."""
    return Powerset(set())


def from_list(values: list[_ValueType]) -> Powerset[_ValueType]:
    """Create powerset from a list of values."""
    return Powerset(values)


def choice(*options: _ValueType) -> Powerset[_ValueType]:
    """Create powerset representing a choice between options."""
    return Powerset(options)

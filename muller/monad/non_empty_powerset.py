from __future__ import annotations

from typing import Callable, FrozenSet, Set, TypeVar, Union, final

from returns.interfaces.container import Container1
from returns.primitives.container import BaseContainer
from returns.primitives.hkt import Kind1, SupportsKind1, dekind, kinded

from muller.monad.base import monad_apply

_ValueType = TypeVar("_ValueType")
_NewValueType = TypeVar("_NewValueType")


@final
class NonEmptyPowerset(
    BaseContainer,
    SupportsKind1["NonEmptyPowerset", _ValueType],  # type: ignore[type-arg]
    Container1[_ValueType],
):
    """
    Non-Empty Powerset Monad for Non-deterministic Computation

    Represents non-deterministic computations as non-empty sets of possible values.

    The key difference from regular powerset monad is that the internal state
    is guaranteed to never be empty, ensuring all computations have at least
    one possible outcome.
    """

    _inner_value: FrozenSet[_ValueType]

    def __init__(
        self, value: Union[Set[_ValueType], FrozenSet[_ValueType], list[_ValueType]]
    ) -> None:
        """
        Initialize a non-empty powerset computation.

        Args:
            value: Set, frozenset, or list of values representing possible outcomes

        Raises:
            ValueError: If values is empty
        """

        value = frozenset(value)

        if len(value) == 0:
            raise ValueError("NonEmptyPowerset cannot be initialized with empty values")

        super().__init__(value)

    @classmethod
    def from_value(cls, value: _NewValueType) -> Kind1["NonEmptyPowerset", _NewValueType]:  # type: ignore[type-arg]
        """
        Create a non-empty powerset with a single value (deterministic computation).
        Implements etaX(x) = {x}

        Args:
            value: The single value to wrap

        Returns:
            NonEmptyPowerset containing only the given value
        """
        return NonEmptyPowerset({value})

    def bind(
        self,
        function: Callable[  # type: ignore[type-arg]
            [_ValueType],
            Kind1["NonEmptyPowerset", _NewValueType],
        ],
    ) -> NonEmptyPowerset[_NewValueType]:
        return NonEmptyPowerset(
            [
                element
                for x in self._inner_value
                for element in dekind(function(x))._inner_value
            ]
        )

    def map(
        self,
        function: Callable[[_ValueType], _NewValueType],
    ) -> NonEmptyPowerset[_NewValueType]:
        return NonEmptyPowerset([function(x) for x in self._inner_value])

    apply = monad_apply

    def __repr__(self) -> str:
        sorted_values = sorted(self._inner_value, key=str)
        return f"NonEmptyPowerset({set(sorted_values)})"


def join(
    monad: NonEmptyPowerset[NonEmptyPowerset[_ValueType]],
) -> NonEmptyPowerset[_ValueType]:
    return NonEmptyPowerset(
        [element for sets in monad._inner_value for element in sets._inner_value]
    )


# Convenience functions for creating common non-empty powersets


@kinded
def singleton(value: _ValueType) -> Kind1["NonEmptyPowerset", _ValueType]:  # type: ignore[type-arg]
    """Create a non-empty powerset with a single value."""
    return NonEmptyPowerset.from_value(value)


def from_list(
    values: list[_ValueType],
) -> NonEmptyPowerset[_ValueType]:
    """
    Create non-empty powerset from a list of values.

    Args:
        values: List of values (must be non-empty)

    Returns:
        NonEmptyPowerset from the list

    Raises:
        ValueError: If list is empty
    """
    return NonEmptyPowerset(values)


def choice(option: _ValueType, *options: _ValueType) -> NonEmptyPowerset[_ValueType]:
    """
    Create non-empty powerset representing a choice between options.

    Args:
        *options: Options to choose from (must provide at least one)

    Returns:
        NonEmptyPowerset representing the choice

    Raises:
        ValueError: If no options provided
    """
    return NonEmptyPowerset(list((option, *options)))

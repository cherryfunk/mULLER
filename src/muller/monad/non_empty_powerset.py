# pip install pymonad
from abc import ABC, abstractmethod
from pymonad.monad import Monad
from typing import Generic, Self, TypeVar, Callable, Set, FrozenSet, Any, Union
import random

from monad.util import ParametrizedMonad

class NonEmptyPowerset[T](ParametrizedMonad[T]):
    """
    Non-Empty Powerset Monad for Non-deterministic Computation

    Represents non-deterministic computations as non-empty sets of possible values.

    The key difference from regular powerset monad is that the internal state
    is guaranteed to never be empty, ensuring all computations have at least
    one possible outcome.
    """

    value: FrozenSet[T]

    def __init__(self, value: Union[Set[T], FrozenSet[T], list]):
        """
        Initialize a non-empty powerset computation.

        Args:
            values: Set, frozenset, or list of values representing possible outcomes

        Raises:
            ValueError: If values is empty
        """

        value = frozenset(value)

        if len(value) == 0:
            raise ValueError("NonEmptyPowerset cannot be initialized with empty values")

        super().__init__(self.value, None)

    @classmethod
    def unit(cls, value: T) -> "NonEmptyPowerset[T]":
        """
        Create a non-empty powerset with a single value (deterministic computation).
        Implements ηX(x) = {x}

        Args:
            value: The single value to wrap

        Returns:
            NonEmptyPowerset containing only the given value
        """
        return cls({value})

    # fmt: off
    def bind[S](self, kleisli_function: Callable[[T], "NonEmptyPowerset[S]"]) -> "NonEmptyPowerset[S]": # pyright: ignore[reportIncompatibleMethodOverride] Python doesn't support Self[S] or other forms of changing type variables in method signatures
        return join(self.map(kleisli_function))
    # fmt: on

    def map[S](self, function: Callable[[T], S]) -> "NonEmptyPowerset[S]":
        return NonEmptyPowerset([function(x) for x in self.value])

    def __repr__(self):
        sorted_values = sorted(self.value, key=str)
        return f"NonEmptyPowerset({set(sorted_values)})"

def join[T](monad: NonEmptyPowerset[NonEmptyPowerset[T]]) -> NonEmptyPowerset[T]:
    return NonEmptyPowerset(
        [element for sets in monad.value for element in sets.value]
    )


# Convenience functions for creating common non-empty powersets


def singleton[T](value: T) -> NonEmptyPowerset[T]:
    """Create a non-empty powerset with a single value."""
    return NonEmptyPowerset.unit(value)


def from_list[T](values: list[T]) -> NonEmptyPowerset[T]:
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


def choice[T](option: T, *options: T) -> NonEmptyPowerset[T]:
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



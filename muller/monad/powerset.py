# pip install pymonad
from pymonad.monad import Monad
from typing import Generic, Iterable, TypeVar, Callable, Set, FrozenSet, Any, Union
import random

ParametrizedMonad = Monad


class Powerset[T](Monad[T]):
    """
    Powerset Monad for Non-deterministic Computation

    Represents non-deterministic computations as sets of possible values.
    """

    value: FrozenSet[T]

    def __init__(self, values: Iterable[T]):
        """
        Initialize a powerset computation.

        Args:
            values: Set, frozenset, or list of values representing possible outcomes
        """
        super().__init__(frozenset(values), None)

    @classmethod
    def insert(cls, value: T) -> "Powerset[T]":
        """
        Create a powerset with a single value (deterministic computation).

        Args:
            value: The single value to wrap

        Returns:
            Powerset containing only the given value
        """
        return cls({value})

    def bind[S](self, kleisli_function: Callable[[T], "Powerset[S]"]) -> "Powerset[S]": # pyright: ignore[reportIncompatibleMethodOverride] This is the correct signature for bind # fmt: skip
        """
        Monadic bind operation (Kleisli extension).

        Apply a function that returns a powerset to each value in this powerset,
        then take the union of all results.

        Args:
            f: Function from value to powerset

        Returns:
            New powerset containing union of all results
        """
        result = set.union(
            set(), *[kleisli_function(value).value for value in self.value]
        )

        return Powerset(result)

    def map[U](self, function: Callable[[T], U]) -> "Powerset[U]":
        """
        Apply a function to all values in the powerset.

        Args:
            function: Function to apply to values

        Returns:
            New powerset with function applied to all values
        """
        return Powerset({function(value) for value in self.value})

    def __repr__(self):
        sorted_values = sorted(self.value, key=str)
        return f"Powerset({set(sorted_values)})"

    def __eq__(self, other):
        if not isinstance(other, Powerset):
            return False
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)


# Convenience functions for creating common powersets


def singleton[T](value: T) -> Powerset[T]:
    """Create a powerset with a single value."""
    return Powerset.insert(value)


def empty() -> Powerset:
    """Create an empty powerset (failed computation)."""
    return Powerset(set())


def from_list[T](values: list[T]) -> Powerset[T]:
    """Create powerset from a list of values."""
    return Powerset(values)


def choice[T](*options: T) -> Powerset[T]:
    """Create powerset representing a choice between options."""
    return Powerset(options)

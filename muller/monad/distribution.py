from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Dict, Tuple, TypeVar, final

from returns.interfaces.container import Container1
from returns.primitives.container import BaseContainer
from returns.primitives.hkt import Kind1, SupportsKind1, dekind

from muller.monad.base import monad_apply

_ValueType = TypeVar("_ValueType")
_NewValueType = TypeVar("_NewValueType")


@final
class Prob(
    BaseContainer,
    SupportsKind1["Prob", _ValueType],  # type: ignore[type-arg]
    Container1[_ValueType],
):
    """
    Probability Distribution Monad

    Represents a discrete probability distribution as a dictionary
    mapping values to their probabilities.
    """

    _inner_value: Dict[_ValueType, float]

    def __init__(self, dist: Dict[_ValueType, float]):
        """
        Initialize a probability distribution.

        Args:
            dist: Dictionary mapping values to probabilities
        """
        # Normalize probabilities
        total = sum(dist.values())
        super().__init__({k: v / total for k, v in dist.items()})

    @classmethod
    def from_value(cls, value: _NewValueType) -> Prob[_NewValueType]:
        """
        Create a distribution with certainty for a single value.
        Also known as 'return' or 'pure' in Haskell.

        Args:
            value: The value with probability 1.0

        Returns:
            Prob distribution with single value
        """
        return Prob({value: 1.0})

    def bind(
        self,
        function: Callable[[_ValueType], Kind1["Prob", _NewValueType]],  # type: ignore[type-arg]
    ) -> Prob[_NewValueType]:
        """
        Monadic bind operation (>>=).

        Apply a function that returns a probability distribution
        to each value in this distribution, weighted by probability.

        Args:
            function: Function from value to probability distribution

        Returns:
            New probability distribution
        """
        result: Dict[_NewValueType, float] = defaultdict(float)

        for value, prob in self._inner_value.items():
            new_dist = dekind(function(value))
            for new_val, new_prob in new_dist._inner_value.items():
                result[new_val] += prob * new_prob

        return Prob(dict(result))

    def map(
        self,
        function: Callable[[_ValueType], _NewValueType],
    ) -> Prob[_NewValueType]:
        """
        Apply a function to all values in the distribution.

        Args:
            function: Function to apply to values

        Returns:
            New probability distribution
        """
        result: Dict[_NewValueType, float] = defaultdict(float)
        for value, prob in self._inner_value.items():
            result[function(value)] += prob
        return Prob(dict(result))

    apply = monad_apply

    def __repr__(self) -> str:
        items = sorted(self._inner_value.items(), key=lambda x: -x[1])
        return f"Prob({dict(items)})"

    # Utility methods

    def expected_value(self, f: Callable[[_ValueType], float]) -> float:
        """Calculate expected value of the distribution."""
        return sum(f(val) * prob for val, prob in self._inner_value.items())

    def sample(self) -> _ValueType:
        """Sample a value from the distribution."""
        values = list(self._inner_value.keys())
        probs = list(self._inner_value.values())
        return random.choices(values, weights=probs)[0]

    def filter(self, predicate: Callable[[_ValueType], bool]) -> Prob[_ValueType]:
        """Filter distribution keeping only values satisfying predicate."""
        filtered = {v: p for v, p in self._inner_value.items() if predicate(v)}
        return Prob(filtered)

    def max_probability(self) -> float:
        """Get the maximum probability value"""
        return max(self._inner_value.values()) if self._inner_value else 0.0

    def argmax(self) -> list[_ValueType]:
        """Get outcomes with maximum probability"""
        if not self._inner_value:
            return []
        max_prob = self.max_probability()
        return [k for k, v in self._inner_value.items() if v >= max_prob - 1e-5]


# Convenience functions for creating common distributions


def uniform(values: list[_ValueType]) -> Prob[_ValueType]:
    """Create uniform distribution over given values."""
    if not values:
        return Prob({})
    prob = 1.0 / len(values)
    return Prob({v: prob for v in values})


def weighted(_ValueType_pairs: list[Tuple[_ValueType, float]]) -> Prob[_ValueType]:
    """Create distribution from (value, weight) pairs."""
    return Prob(dict(_ValueType_pairs))


def bernoulli(
    p: float,
    true_val: bool = True,
    false_val: bool = False,
) -> Prob[bool]:
    """Create Bernoulli distribution."""
    return Prob({true_val: p, false_val: 1 - p})

# pip install pymonad
import random
from collections import defaultdict
from typing import Callable, Dict, Tuple

from muller.monad.base import ParametrizedMonad


class Prob[T](ParametrizedMonad[T]):
    """
    Probability Distribution Monad

    Represents a discrete probability distribution as a dictionary
    mapping values to their probabilities.
    """

    def __init__(self, dist: Dict[T, float]):
        """
        Initialize a probability distribution.

        Args:
            dist: Dictionary mapping values to probabilities
        """
        # Normalize probabilities
        total = sum(dist.values())
        self.value = {k: v / total for k, v in dist.items()}

    @classmethod
    def insert(cls, value: T) -> "Prob[T]":
        """
        Create a distribution with certainty for a single value.
        Also known as 'return' or 'pure' in Haskell.

        Args:
            value: The value with probability 1.0

        Returns:
            Prob distribution with single value
        """
        return cls({value: 1.0})

    def bind[S](self, kleisli_function: Callable[[T], 'Prob[S]']) -> 'Prob[S]':  # pyright: ignore[reportIncompatibleMethodOverride] This is the correct signature for bind # fmt: skip
        """
        Monadic bind operation (>>=).

        Apply a function that returns a probability distribution
        to each value in this distribution, weighted by probability.

        Args:
            f: Function from value to probability distribution

        Returns:
            New probability distribution
        """
        result = defaultdict(float)

        for value, prob in self.value.items():
            new_dist = kleisli_function(value)
            for new_val, new_prob in new_dist.value.items():
                result[new_val] += prob * new_prob

        return Prob(dict(result))

    def map[U](self, function: Callable[[T], U]) -> "Prob[U]":
        """
        Apply a function to all values in the distribution.

        Args:
            f: Function to apply to values

        Returns:
            New probability distribution
        """
        result = defaultdict(float)
        for value, prob in self.value.items():
            result[function(value)] += prob
        return Prob(dict(result))

    def __repr__(self):
        items = sorted(self.value.items(), key=lambda x: -x[1])
        return f"Prob({dict(items)})"

    # Utility methods

    def expected_value(self, f: Callable[[T], float]) -> float:
        """Calculate expected value of the distribution."""
        return sum(f(val) * prob for val, prob in self.value.items())

    def sample(self) -> T:
        """Sample a value from the distribution."""
        values = list(self.value.keys())
        probs = list(self.value.values())
        return random.choices(values, weights=probs)[0]

    def filter(self, predicate: Callable[[T], bool]) -> "Prob[T]":
        """Filter distribution keeping only values satisfying predicate."""
        filtered = {v: p for v, p in self.value.items() if predicate(v)}
        return Prob(filtered)

    def max_probability(self) -> float:
        """Get the maximum probability value"""
        return max(self.value.values()) if self.value else 0.0

    def argmax(self) -> list[T]:
        """Get outcomes with maximum probability"""
        if not self.value:
            return []
        max_prob = self.max_probability()
        return [k for k, v in self.value.items() if v >= max_prob - 1e-5]


# Convenience functions for creating common distributions


def uniform(values: list) -> Prob:
    """Create uniform distribution over given values."""
    if not values:
        return Prob({})
    prob = 1.0 / len(values)
    return Prob({v: prob for v in values})


def weighted[T](pairs: list[Tuple[T, float]]) -> Prob[T]:
    """Create distribution from (value, weight) pairs."""
    return Prob(dict(pairs))


def bernoulli[T](p: float, true_val: T = True, false_val: T = False) -> Prob[T]:
    """Create Bernoulli distribution."""
    return Prob({true_val: p, false_val: 1 - p})

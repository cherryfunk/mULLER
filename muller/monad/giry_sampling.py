from __future__ import annotations

from typing import Callable, List, TypeVar, Union

import numpy as np
from returns.interfaces.container import Container1
from returns.primitives.container import BaseContainer
from returns.primitives.hkt import Kind1, SupportsKind1
from typing_extensions import final

from muller.monad.base import monad_apply

_ValueType = TypeVar("_ValueType")
_NewValueType = TypeVar("_NewValueType")


@final
class GirySampling(
    BaseContainer,
    SupportsKind1["GirySampling", _ValueType],  # type: ignore[type-arg]
    Container1[_ValueType],
):
    """
    A monad that wraps sampling-based distributions for probabilistic programming.
    Uses sampling capabilities for efficient probabilistic inference.
    """

    _inner_value: Callable[[], _ValueType]

    def __init__(self, value: Callable[[], _ValueType]) -> None:
        """
        Initialize the GirySampling monad.

        Args:
            value: A function that returns samples from the distribution
        """
        super().__init__(value)

    @classmethod
    def from_value(cls, value: _NewValueType) -> GirySampling[_NewValueType]:
        """
        Insert a pure value into the monad (equivalent to return/pure).
        Always returns the given value.
        """
        return GirySampling(lambda: value)

    def bind(
        self,
        function: Callable[  # type: ignore[type-arg]
            [_ValueType], Kind1["GirySampling", _NewValueType]
        ],
    ) -> GirySampling[_NewValueType]:
        """
        Monadic bind operation for composing probabilistic computations.
        """

        def bound_sampler() -> _NewValueType:
            # Sample from this distribution
            sample = self._inner_value()

            # Apply the kleisli function to get a new monad
            result_monad = function(sample)

            # Sample from the resulting monad
            return result_monad._inner_value()

        return GirySampling(value=bound_sampler)

    def map(
        self,  # GiriySampling[_ValueType]
        function: Callable[[_ValueType], _NewValueType],  # _ValueType -> _NewValueType
    ) -> GirySampling[_NewValueType]:  # GirySampling[_NewValueType]
        return GirySampling(value=lambda: function(self._inner_value()))

    apply = monad_apply

    def sample(self, num_samples: int = 1000) -> List[_ValueType]:
        """
        Sample from the distribution.
        Returns a list to handle heterogeneous types.
        """
        return [self._inner_value() for _ in range(num_samples)]

    def mean(self, num_samples: int = 10000) -> float:
        """
        Compute the mean of the distribution via sampling.
        Only works for numeric distributions.
        """
        samples: List[_ValueType] = self.sample(num_samples)
        # Convert to numpy array for numeric computations
        try:
            numeric_samples = np.array(samples)
            return float(np.mean(numeric_samples))
        except:  # noqa: E722
            # If conversion fails, try to compute mean of numeric values only
            numeric_values = [s for s in samples if isinstance(s, (int, float))]
            if numeric_values:
                return float(np.mean(numeric_values))

            raise ValueError("Cannot compute mean of non-numeric distribution")


def categorical(vals: List[Union[int, float]]) -> GirySampling[int]:
    """
    Create a categorical distribution from a list of values or weights.
    """

    def sampler() -> int:
        if all(isinstance(v, (int, float)) for v in vals):
            # vals are weights
            probs = np.array(vals, dtype=float)
            probs = probs / np.sum(probs)
            return int(np.random.choice(len(vals), p=probs))
        else:
            # vals are categories with equal weights
            return int(np.random.choice(len(vals)))

    return GirySampling(value=sampler)


def uniform(lower: float, upper: float) -> GirySampling[float]:
    """
    Create a uniform distribution between lower and upper bounds.
    """

    def sampler() -> float:
        return float(np.random.uniform(lower, upper))

    return GirySampling(value=sampler)


def binomial(n: int, p: float) -> GirySampling[int]:
    """
    Create a binomial distribution with parameters n and p.
    """

    def sampler() -> int:
        return int(np.random.binomial(n, p))

    return GirySampling(value=sampler)


def poisson(lam: float) -> GirySampling[int]:
    """
    Create a Poisson distribution with parameter lambda.
    """

    def sampler() -> int:
        return int(np.random.poisson(lam))

    return GirySampling(value=sampler)


def geometric(p: float) -> GirySampling[int]:
    """
    Create a geometric distribution with parameter p.
    """

    def sampler() -> int:
        return int(np.random.geometric(p))

    return GirySampling(value=sampler)


def negative_binomial(n: int, p: float) -> GirySampling[int]:
    """
    Create a negative binomial distribution with parameters n and p.
    """

    def sampler() -> int:
        return int(np.random.negative_binomial(n, p))

    return GirySampling(value=sampler)


def beta(alpha: float, beta: float) -> GirySampling[float]:
    """
    Create a beta distribution with parameters alpha and beta.
    """

    def sampler() -> float:
        return float(np.random.beta(alpha, beta))

    return GirySampling(value=sampler)


def normal(mean: float, std: float) -> GirySampling[float]:
    """
    Create a normal distribution with parameters mean and std.
    """

    def sampler() -> float:
        return float(np.random.normal(mean, std))

    return GirySampling(value=sampler)


def bernoulli(p: float) -> GirySampling[int]:
    """
    Create a Bernoulli distribution with parameter p.
    """

    def sampler() -> int:
        return int(np.random.rand() < p)

    return GirySampling(value=sampler)


def from_sampler_fn(
    sampler_fn: Callable[[], _ValueType],
) -> GirySampling[_ValueType]:
    """
    Create a GirySampling distribution from a sampling function.
    """
    return GirySampling(value=sampler_fn)

import numpy as np
from typing import Callable, List, Union
from muller.monad.base import ParametrizedMonad


class GirySampling[T](ParametrizedMonad[T]):
    """
    A monad that wraps PyMC distributions for probabilistic programming.
    Uses PyMC's sampling capabilities for efficient probabilistic inference.
    """

    def __init__(self, value: Callable[[], T]) -> None:
        """
        Initialize the GirySampling monad.

        Args:
            value: A function that returns samples from the distribution
            value: A constant value (for pure/insert)
        """
        self.value: Callable[[], T] = value

    @classmethod
    def insert[U](cls, value: U) -> "GirySampling[U]":
        """
        Insert a pure value into the monad (equivalent to return/pure).
        Always returns the given value.
        """
        return GirySampling[U](value=lambda: value)

    def bind[S](self, kleisli_function: Callable[[T], 'GirySampling[S]']) -> 'GirySampling[S]':  # pyright: ignore[reportIncompatibleMethodOverride] # fmt: skip # noqa: E501
        """
        Monadic bind operation for composing probabilistic computations.
        """

        def bound_sampler() -> S:
            # Sample from this distribution
            sample = self.value()

            # Apply the kleisli function to get a new monad
            result_monad: "GirySampling[S]" = kleisli_function(sample)

            # Sample from the resulting monad
            return result_monad.value()

        return GirySampling[S](value=bound_sampler)

    def map[S](self, function: Callable[[T], S]) -> "GirySampling[S]":
        return GirySampling(value=lambda: function(self.value()))

    def sample(self, num_samples: int = 1000) -> List[T]:
        """
        Sample from the distribution.
        Returns a list to handle heterogeneous types.
        """
        return [self.value() for _ in range(num_samples)]

    def mean(self, num_samples: int = 10000) -> float:
        """
        Compute the mean of the distribution via sampling.
        Only works for numeric distributions.
        """
        samples: List[T] = self.sample(num_samples)
        # Convert to numpy array for numeric computations
        try:
            numeric_samples = np.array(samples)
            return float(np.mean(numeric_samples))
        except:
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

    return GirySampling[int](value=sampler)


def uniform(lower: float, upper: float) -> GirySampling[float]:
    """
    Create a uniform distribution between lower and upper bounds.
    """

    def sampler() -> float:
        return float(np.random.uniform(lower, upper))

    return GirySampling[float](value=sampler)


def binomial(n: int, p: float) -> GirySampling[int]:
    """
    Create a binomial distribution with parameters n and p.
    """

    def sampler() -> int:
        return int(np.random.binomial(n, p))

    return GirySampling[int](value=sampler)


def poisson(lam: float) -> GirySampling[int]:
    """
    Create a Poisson distribution with parameter lambda.
    """

    def sampler() -> int:
        return int(np.random.poisson(lam))

    return GirySampling[int](value=sampler)


def geometric(p: float) -> GirySampling[int]:
    """
    Create a geometric distribution with parameter p.
    """

    def sampler() -> int:
        return int(np.random.geometric(p))

    return GirySampling[int](value=sampler)


def negative_binomial(n: int, p: float) -> GirySampling[int]:
    """
    Create a negative binomial distribution with parameters n and p.
    """

    def sampler() -> int:
        return int(np.random.negative_binomial(n, p))

    return GirySampling[int](value=sampler)


def beta(alpha: float, beta: float) -> GirySampling[float]:
    """
    Create a beta distribution with parameters alpha and beta.
    """

    def sampler() -> float:
        return float(np.random.beta(alpha, beta))

    return GirySampling[float](value=sampler)


def normal(mean: float, std: float) -> GirySampling[float]:
    """
    Create a normal distribution with parameters mean and std.
    """

    def sampler() -> float:
        return float(np.random.normal(mean, std))

    return GirySampling[float](value=sampler)


def bernoulli(p: float) -> GirySampling[int]:
    """
    Create a Bernoulli distribution with parameter p.
    """

    def sampler() -> int:
        return int(np.random.rand() < p)

    return GirySampling[int](value=sampler)


def from_sampler_fn[T](sampler_fn: Callable[[], T]) -> GirySampling[T]:
    """
    Create a GirySampling distribution from a sampling function.
    """
    return GirySampling[T](value=sampler_fn)

from typing import Callable, List, cast

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln, comb


from muller.monad.base import ParametrizedMonad

type Measure[T] = Callable[[Callable[[T], float]], float]
"""
Type alias for a measure.

A measure is represented as a higher-order function that takes an integrable
function f: T -> float and returns the integral of f with respect to the measure.
This representation allows for a uniform treatment of discrete and continuous
probability distributions.
"""


def integrate[T](f: Callable[[T], float], measure: Measure[T]) -> float:
    """
    Integrate a function with respect to a measure.

    This is a convenience function that applies the measure to the function,
    effectively computing the integral of f with respect to the measure.

    Args:
        f: The function to integrate
        measure: The measure to integrate with respect to

    Returns:
        The integral of f with respect to the measure
    """
    return measure(f)


class GiryMonad[T](ParametrizedMonad[Measure[T]]):
    """
    Giry Monad for Probabilistic Computation with Measures

    Represents probabilistic computations using measures (generalized probability
    distributions). The Giry monad provides a mathematical foundation for probability
    theory that can handle both discrete and continuous distributions uniformly.

    A measure is represented as a higher-order function that takes an integrable function
    and returns the integral of that function with respect to the measure.
    """

    value: Measure[T]

    def __init__(self, value: Measure[T]):
        """
        Initialize a Giry monad with a measure.

        Args:
            value: A measure function that takes an integrable function and returns
                the integral
        """
        self.value = value

    @classmethod
    def unit(cls, value: T) -> "GiryMonad":
        """
        Create a Giry monad with a point mass (Dirac delta measure).
        Also known as 'return' or 'pure' in Haskell.

        Args:
            value: The value at which to place a point mass

        Returns:
            GiryMonad with a Dirac delta measure at the given value
        """
        return cls(lambda f: f(value))

    def bind[S](self, kleisli_function: Callable[[T], "GiryMonad[S]"]) -> "GiryMonad[S]": # pyright: ignore[reportIncompatibleMethodOverride] # fmt: skip # noqa: E501
        """
        Monadic bind operation (>>=) for the Giry monad.

        Implements the integration of measures according to the monadic law.
        This allows for composing probabilistic computations.

        Args:
            kleisli_function: Function from value to GiryMonad (Kleisli arrow)

        Returns:
            New GiryMonad representing the composed probabilistic computation
        """
        rho = kleisli_function

        return GiryMonad(
            lambda f: integrate(lambda m: integrate(f, rho(m).value), self.value)
        )

    def map[U](self, function: Callable[[T], U]) -> "GiryMonad[U]":  # pyright: ignore[reportIncompatibleMethodOverride] # fmt: skip # noqa: E501
        """
        Apply a function to the values in the measure (functor map).

        Transforms the support of the measure by applying the given function.

        Args:
            function: Function to apply to values in the measure

        Returns:
            New GiryMonad with the function applied to the measure
        """
        return GiryMonad(lambda f: integrate(lambda m: f(function(m)), self.value))

    def amap[S](self: 'GiryMonad[Callable[[S], T]]', monad_value: 'GiryMonad[S]') -> 'GiryMonad[T]':   # pyright: ignore[reportIncompatibleMethodOverride] # fmt: skip # noqa: E501
        """
        Applicative functor application for the Giry monad.

        Apply a measure of functions to a measure of values, producing a measure of
        results.

        Args:
            monad_value: GiryMonad containing values to apply functions to

        Returns:
            New GiryMonad with functions applied to values
        """
        g = self.value
        h = monad_value.value
        return GiryMonad(lambda f: g(lambda k: h(lambda x: f(k(x)))))

    def __repr__(self):
        name = self.value.__name__ if hasattr(self.value, "__name__") else "<measure>"
        return f"GiryMonad({name})"


# Convenience functions for creating common measures


# fromMassFunction :: (a -> Double) -> [a] -> Measure a
# fromMassFunction f support = Measure $ \g ->
#   foldl' (\acc x -> acc + f x * g x) 0 support
def fromMassFunction[T](f: Callable[[T], float], support: list[T]) -> GiryMonad[T]:
    """Creates a GiryMonad from a mass function.

    Args:
        f: A function that maps values of type T to their probabilities.
        support: A list of values of type T representing the support of the mass function.

    Returns:
        A GiryMonad representing the mass function.
    """
    return GiryMonad(lambda g: sum(g(x) * f(x) for x in support))


# binomial :: Int -> Double -> Measure Int
# binomial n p = fromMassFunction (pmf n p) [0..n] where
#   pmf n p x
#     | x < 0 || n < x = 0
#     | otherwise = choose n x * p ^^ x * (1 - p) ^^ (n - x)
def binomial(n: int, p: float) -> GiryMonad[int]:
    """Creates a GiryMonad from a binomial distribution.

    Args:
        n: The number of trials.
        p: The probability of success.

    Returns:
        A GiryMonad representing the binomial distribution.
    """

    def mass_function(x: int) -> float:
        if x < 0 or x > n:
            return 0.0

        return cast(float, comb(n, x)) * (p**x) * ((1 - p) ** (n - x))

    support = list(range(n + 1))
    return fromMassFunction(mass_function, support)


# fromDensityFunction :: (Double -> Double) -> Measure Double
# fromDensityFunction d = Measure $ \f ->
#     quadratureTanhSinh (\x -> f x * d x)
#   where
#     quadratureTanhSinh = result . last . everywhere trap
def fromDensityFunction(d: Callable[[float], float]) -> GiryMonad[float]:
    """Creates a GiryMonad from a density function.

    Args:
        d: A function that maps values in the support to their densities.

    Returns:
        A GiryMonad representing the density function.
    """
    return GiryMonad(lambda f: quad(lambda x: f(x) * d(x), -np.inf, np.inf)[0])


# beta :: Double -> Double -> Measure Double
# beta a b = fromDensityFunction (density a b) where
#   density a b p
#     | p < 0 || p > 1 = 0
#     | otherwise = 1 / exp (logBeta a b) * p ** (a - 1) * (1 - p) ** (b - 1)
def beta(a: float, b: float) -> GiryMonad[float]:
    """Creates a GiryMonad from a beta distribution.

    Args:
        a: The alpha parameter of the beta distribution.
        b: The beta parameter of the beta distribution.

    Returns:
        A GiryMonad representing the beta distribution.
    """

    def density(p: float) -> float:
        if p < 0 or p > 1:
            return 0.0
        return 1 / np.exp(betaln(a, b)) * (p ** (a - 1)) * ((1 - p) ** (b - 1))

    return fromDensityFunction(density)


# betaBinomial :: Int -> Double -> Double -> Measure Int
# betaBinomial n a b = beta a b >>= binomial n
def betaBinomial(n: int, a: float, b: float) -> GiryMonad[int]:
    """Creates a GiryMonad from a beta-binomial distribution.

    Args:
        n: The number of trials.
        a: The alpha parameter of the beta distribution.
        b: The beta parameter of the beta distribution.

    Returns:
        A GiryMonad representing the beta-binomial distribution.
    """
    return beta(a, b).bind(lambda p: binomial(n, p))


# fromSample :: Foldable f => f a -> Measure a
# fromSample = Measure . flip weightedAverage
def fromSample[T](sample: List[T]) -> GiryMonad[T]:
    """Creates a GiryMonad from a sample.

    Args:
        sample: A list of values representing the sample.

    Returns:
        A GiryMonad representing the sample.
    """

    def weighted_average(f: Callable[[T], float]) -> float:
        return sum(f(x) for x in sample) / len(sample)

    return GiryMonad(weighted_average)

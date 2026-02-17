from __future__ import annotations

from typing import Callable, List, TypeVar, cast, final

import numpy as np
from returns.interfaces.container import Container1
from returns.primitives.container import BaseContainer
from returns.primitives.hkt import Kind1, SupportsKind1, dekind
from scipy.integrate import quad
from scipy.special import betaln, comb

from muller.monad.base import monad_apply

_ValueType = TypeVar("_ValueType")
_NewValueType = TypeVar("_NewValueType")

Measure = Callable[[Callable[[_ValueType], float]], float] # (_ValueType -> float) -> float
"""
Type alias for a measure.

A measure is represented as a higher-order function that takes an integrable
function f: T -> float and returns the integral of f with respect to the measure.
This representation allows for a uniform treatment of discrete and continuous
probability distributions.
"""


def integrate(f: Callable[[_ValueType], float], measure: Measure[_ValueType]) -> float:
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


@final
class Giry(
    BaseContainer,
    SupportsKind1["Giry", _ValueType],  # type: ignore[type-arg]
    Container1[_ValueType],
):
    """
    Giry Monad for Probabilistic Computation with Measures

    Represents probabilistic computations using measures (generalized probability
    distributions). The Giry monad provides a mathematical foundation for probability
    theory that can handle both discrete and continuous distributions uniformly.

    A measure is represented as a higher-order function that takes an integrable function
    and returns the integral of that function with respect to the measure.
    """

    _inner_value: Measure[_ValueType]

    def __init__(self, value: Measure[_ValueType]):
        """
        Initialize a Giry monad with a measure.

        Args:
            value: A measure function that takes an integrable function and returns
                the integral
        """
        super().__init__(value)

    @classmethod
    def from_value(cls, value: _NewValueType) -> Giry[_NewValueType]:
        """
        Create a Giry monad with a point mass (Dirac delta measure).
        Also known as 'return' or 'pure' in Haskell.

        Args:
            value: The value at which to place a point mass

        Returns:
            GiryMonad with a Dirac delta measure at the given value
        """
        return Giry(lambda f: f(value))

    def bind(
        self,
        function: Callable[[_ValueType], Kind1["Giry", _NewValueType]],  # type: ignore[type-arg]
    ) -> Giry[_NewValueType]:
        """
        Monadic bind operation (>>=) for the Giry monad.

        Implements the integration of measures according to the monadic law.
        This allows for composing probabilistic computations.

        Args:
            function: Function from value to GiryMonad (Kleisli arrow)

        Returns:
            New GiryMonad representing the composed probabilistic computation
        """
        rho = function

        return Giry(
            lambda f: integrate(
                lambda m: integrate(f, dekind(rho(m))._inner_value),
                self._inner_value,
            )
        )

    def map(
        self,
        function: Callable[[_ValueType], _NewValueType],
    ) -> Giry[_NewValueType]:
        """
        Apply a function to the values in the measure (functor map).

        Transforms the support of the measure by applying the given function.

        Args:
            function: Function to apply to values in the measure

        Returns:
            New GiryMonad with the function applied to the measure
        """
        return Giry(lambda f: integrate(lambda m: f(function(m)), self._inner_value))

    apply = monad_apply
    # def apply(
    #     self,
    #     container: Kind1[Giry, Callable[[_ValueType], _NewValueType]],
    # ) -> Giry[_NewValueType]:
    #     """
    #     Applicative functor application for the Giry monad.

    #     Apply a measure of functions to a measure of values, producing a measure of
    #     results.

    #     Args:
    #         container: A GiryMonad containing functions to apply to the values in this GiryMonad

    #     Returns:
    #         New GiryMonad with functions applied to values
    #     """
    #     g = self._inner_value
    #     h = dekind(container)._inner_value
    #     return Giry(lambda f: g(lambda k: h(lambda x: f(k(x)))))

    def __repr__(self) -> str:
        name = (
            self._inner_value.__name__
            if hasattr(self._inner_value, "__name__")
            else "<measure>"
        )
        return f"GiryMonad({name})"


# Convenience functions for creating common measures


def fromMassFunction(
    f: Callable[[_ValueType], float], support: list[_ValueType]
) -> Giry[_ValueType]:
    """Creates a GiryMonad from a mass function.

    Args:
        f: A function that maps values of type T to their probabilities.
        support: A list of values representing the support of the mass function.

    Returns:
        A GiryMonad representing the mass function.
    """
    return Giry(lambda g: sum(g(x) * f(x) for x in support))


def binomial(n: int, p: float) -> Giry[int]:
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


def fromDensityFunction(
    d: Callable[[float], float],
) -> Giry[float]:
    """Creates a GiryMonad from a density function.

    Args:
        d: A function that maps values in the support to their densities.

    Returns:
        A GiryMonad representing the density function.
    """
    return Giry(lambda f: quad(lambda x: f(x) * d(x), -np.inf, np.inf)[0])


def beta(a: float, b: float) -> Giry[float]:
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

        r: float = 1 / np.exp(betaln(a, b)) * (p ** (a - 1)) * ((1 - p) ** (b - 1))
        return r

    return fromDensityFunction(density)


def betaBinomial(n: int, a: float, b: float) -> Giry[int]:
    """Creates a GiryMonad from a beta-binomial distribution.

    Args:
        n: The number of trials.
        a: The alpha parameter of the beta distribution.
        b: The beta parameter of the beta distribution.

    Returns:
        A GiryMonad representing the beta-binomial distribution.
    """
    return beta(a, b).bind(lambda p: binomial(n, p))


def fromSample(sample: List[_ValueType]) -> Giry[_ValueType]:
    """Creates a GiryMonad from a sample.

    Args:
        sample: A list of values representing the sample.

    Returns:
        A GiryMonad representing the sample.
    """

    def weighted_average(f: Callable[[_ValueType], float]) -> float:
        return sum(f(x) for x in sample) / len(sample)

    return Giry(weighted_average)

# -- continuous distribution over UniverseW
# uniformUniverseW :: SamplerIO UniverseW
# uniformUniverseW = do
#   choice <- categorical $ V.fromList [1,1,1]
#   case choice of
#     0 -> do
#       -- sample an integer uniformly from [-10..10]
#       i <- categorical $ V.fromList $ replicate 21 1
#       return (Int (i-10))
#     1 -> do
#       x <- uniform 0 1       -- x ~ U(0,1)
#       return (Double x)
#     2 -> do
#       x <- uniform (-5) 20      -- Pair (x,y), each U(-1,1)
#       y <- uniform 0 5
#       return (Pair (x, y))


import numpy as np


from typing import Callable, List, Optional, Union

from pymonad.monad import Monad

from muller.monad.base import ParametrizedMonad


class GiryNP[T](ParametrizedMonad[T]):
    """
    A monad that wraps PyMC distributions for probabilistic programming.
    Uses PyMC's sampling capabilities for efficient probabilistic inference.
    """

    def __init__(self,
                 sampler_fn: Callable[[], T]) -> None:
        """
        Initialize the GiryNP monad.

        Args:
            sampler_fn: A function that returns samples from the distribution
            value: A constant value (for pure/insert)
        """
        self.sampler_fn: Callable[[], T] = sampler_fn

    @classmethod
    def insert[U](cls, value: U) -> 'GiryNP[U]':
        """
        Insert a pure value into the monad (equivalent to return/pure).
        Always returns the given value.
        """
        return GiryNP[U](sampler_fn=lambda: value)

    def bind[S](self, kleisli_function: Callable[[T], 'GiryNP[S]']) -> 'GiryNP[S]':
        """
        Monadic bind operation for composing probabilistic computations.
        """
        def bound_sampler() -> S:
            # Sample from this distribution
            sample = self.sampler_fn()

            # Apply the kleisli function to get a new monad
            result_monad: 'GiryNP[S]' = kleisli_function(sample)

            # Sample from the resulting monad
            return result_monad.sampler_fn()

        return GiryNP[S](sampler_fn=bound_sampler)
    
    def map[S](self: 'GiryNP[S]', function: Callable[[S], T]) -> 'GiryNP[T]':
        return GiryNP[T](sampler_fn=lambda: function(self.sampler_fn()))

    def sample(self, num_samples: int = 1000) -> List[T]:
        """
        Sample from the distribution.
        Returns a list to handle heterogeneous types.
        """
        return [self.sampler_fn() for _ in range(num_samples)]

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



def categorical(vals: List[Union[int, float]]) -> GiryNP[int]:
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
    
    return GiryNP[int](sampler_fn=sampler)


def uniform(lower: float, upper: float) -> GiryNP[float]:
    """
    Create a uniform distribution between lower and upper bounds.
    """
    def sampler() -> float:
        return float(np.random.uniform(lower, upper))
    
    return GiryNP[float](sampler_fn=sampler)

def binomial(n: int, p: float) -> GiryNP[int]:
    """
    Create a binomial distribution with parameters n and p.
    """
    def sampler() -> int:
        return int(np.random.binomial(n, p))

    return GiryNP[int](sampler_fn=sampler)

def poisson(lam: float) -> GiryNP[int]:
    """
    Create a Poisson distribution with parameter lambda.
    """
    def sampler() -> int:
        return int(np.random.poisson(lam))

    return GiryNP[int](sampler_fn=sampler)

def geometric(p: float) -> GiryNP[int]:
    """
    Create a geometric distribution with parameter p.
    """
    def sampler() -> int:
        return int(np.random.geometric(p))

    return GiryNP[int](sampler_fn=sampler)

def negative_binomial(n: int, p: float) -> GiryNP[int]:
    """
    Create a negative binomial distribution with parameters n and p.
    """
    def sampler() -> int:
        return int(np.random.negative_binomial(n, p))

    return GiryNP[int](sampler_fn=sampler)

def beta(alpha: float, beta: float) -> GiryNP[float]:
    """
    Create a beta distribution with parameters alpha and beta.
    """
    def sampler() -> float:
        return float(np.random.beta(alpha, beta))

    return GiryNP[float](sampler_fn=sampler)

def normal(mean: float, std: float) -> GiryNP[float]:
    """
    Create a normal distribution with parameters mean and std.
    """
    def sampler() -> float:
        return float(np.random.normal(mean, std))

    return GiryNP[float](sampler_fn=sampler)


def from_sampler_fn[T](sampler_fn: Callable[[], T]) -> GiryNP[T]:
    """
    Create a GiryNP distribution from a sampling function.
    """
    return GiryNP[T](sampler_fn=sampler_fn)

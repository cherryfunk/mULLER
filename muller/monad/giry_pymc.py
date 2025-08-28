
from typing import Callable, Union, Any
import pymc as pm
import pytensor.tensor as pt
from pytensor.tensor.variable import TensorVariable
from muller.monad.base import ParametrizedMonad

class GiryMonadPyMC(ParametrizedMonad[TensorVariable]):
    """
    Giry Monad for Probabilistic Computation using PyMC distributions.
    
    This implementation provides a monadic interface for PyMC distributions,
    allowing for compositional probabilistic programming within the PyMC ecosystem.
    """

    def __init__(self, value: TensorVariable):
        """
        Initialize a Giry monad with a PyMC distribution or random variable.

        Args:
            value: A PyMC distribution (TensorVariable) or random variable
        """
        self.value = value

    @classmethod
    def unit(cls, x: TensorVariable) -> "GiryMonadPyMC":
        """
        Create a Giry monad with a deterministic (point mass) distribution.

        Args:
            x: The tensor variable to wrap in a deterministic distribution

        Returns:
            GiryMonadPyMC with a deterministic distribution at the given value
        """
        # The input is already a TensorVariable, so we can use it directly
        return cls(x)

    def bind(self, kleisli_function: Callable[[TensorVariable], 'GiryMonadPyMC']) -> 'GiryMonadPyMC':  # pyright: ignore[reportIncompatibleMethodOverride] # fmt: skip # noqa: E501
        """
        Monadic bind operation (>>=) for the PyMC Giry monad.

        This creates a new distribution that represents the composition of
        probabilistic computations. The implementation relies on PyMC's
        symbolic computation graph to handle the integration.

        Args:
            kleisli_function: Function from TensorVariable to GiryMonadPyMC (Kleisli arrow)

        Returns:
            New GiryMonadPyMC representing the composed probabilistic computation
        """
        # Apply the Kleisli function to get the resulting distribution
        result_monad = kleisli_function(self.value)
        return result_monad

    def map(self, function: Callable[[TensorVariable], TensorVariable]) -> "GiryMonadPyMC":
        """
        Apply a function to the distribution (functor map).

        Transforms the distribution by applying the given function to its values.

        Args:
            function: Function to apply to the distribution

        Returns:
            New GiryMonadPyMC with the function applied to the distribution
        """
        transformed_value = function(self.value)
        return GiryMonadPyMC(transformed_value)

    def sample(self, draws: int = 1000, **kwargs) -> Any:
        """
        Draw samples from the distribution.

        Args:
            draws: Number of samples to draw
            **kwargs: Additional arguments to pass to PyMC sampling

        Returns:
            Samples from the distribution
        """
        return pm.draw(self.value, draws=draws, **kwargs)

    def __repr__(self) -> str:
        """String representation of the GiryMonadPyMC."""
        return f"GiryMonadPyMC({self.value})"


# Convenience functions for creating common PyMC distributions

def constant(value: Union[float, int]) -> GiryMonadPyMC:
    """
    Create a GiryMonadPyMC with a constant (deterministic) value.
    
    This is a convenience function that wraps a raw value in a TensorVariable
    and then creates a unit monad from it. This is the typical entry point
    for users who want to create a deterministic distribution from a raw value.

    Args:
        value: The constant value

    Returns:
        GiryMonadPyMC wrapping a deterministic distribution
    """
    tensor_var = pt.as_tensor_variable(value)
    return GiryMonadPyMC.unit(tensor_var)


def normal(mu: Union[float, int, TensorVariable] = 0.0, 
           sigma: Union[float, int, TensorVariable] = 1.0) -> GiryMonadPyMC:
    """
    Create a GiryMonadPyMC with a normal distribution.

    Args:
        mu: Mean of the normal distribution
        sigma: Standard deviation of the normal distribution

    Returns:
        GiryMonadPyMC wrapping a normal distribution
    """
    dist = pm.Normal.dist(mu=mu, sigma=sigma)  # type: ignore
    return GiryMonadPyMC(dist)


def uniform(lower: Union[float, int, TensorVariable] = 0.0,
            upper: Union[float, int, TensorVariable] = 1.0) -> GiryMonadPyMC:
    """
    Create a GiryMonadPyMC with a uniform distribution.

    Args:
        lower: Lower bound of the uniform distribution
        upper: Upper bound of the uniform distribution

    Returns:
        GiryMonadPyMC wrapping a uniform distribution
    """
    dist = pm.Uniform.dist(lower=lower, upper=upper)  # type: ignore
    return GiryMonadPyMC(dist)


def beta(alpha: Union[float, int, TensorVariable],
         beta_param: Union[float, int, TensorVariable]) -> GiryMonadPyMC:
    """
    Create a GiryMonadPyMC with a beta distribution.

    Args:
        alpha: Alpha parameter of the beta distribution
        beta_param: Beta parameter of the beta distribution

    Returns:
        GiryMonadPyMC wrapping a beta distribution
    """
    dist = pm.Beta.dist(alpha=alpha, beta=beta_param)  # type: ignore
    return GiryMonadPyMC(dist)


def binomial(n: Union[int, TensorVariable],
             p: Union[float, int, TensorVariable]) -> GiryMonadPyMC:
    """
    Create a GiryMonadPyMC with a binomial distribution.

    Args:
        n: Number of trials
        p: Probability of success

    Returns:
        GiryMonadPyMC wrapping a binomial distribution
    """
    dist = pm.Binomial.dist(n=n, p=p)  # type: ignore
    return GiryMonadPyMC(dist)
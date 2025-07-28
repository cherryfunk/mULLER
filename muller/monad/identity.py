# pip install pymonad
from typing import Callable

from pymonad.monad import Monad


class Identity[T](Monad[T]):
    """
    Identity Monad for Deterministic Side-Effect Free Computation

    The simplest monad that just wraps values without any computational effects.
    """

    def __init__(self, value: T):
        """
        Initialize an Identity monad with a value.

        Args:
            value: The value to wrap
        """
        super().__init__(value, None)

    @classmethod
    def insert(cls, value: T) -> "Identity":
        """
        Create an Identity monad (pure computation).

        Args:
            value: The value to wrap

        Returns:
            Identity monad containing the value
        """
        return cls(value)

    def bind[S](self, kleisli_function: Callable[[T], 'Identity[S]']) -> 'Identity[S]':  # pyright: ignore[reportIncompatibleMethodOverride] This is the correct signature for bind # fmt: skip
        """
        Monadic bind operation.

        Args:
            f: Function from value to Identity monad

        Returns:
            Result of applying f to the wrapped value
        """
        return kleisli_function(self.value)

    def map[U](self, function: Callable[[T], U]) -> "Identity[U]":
        """
        Apply a function to the wrapped value.

        Args:
            f: Function to apply

        Returns:
            New Identity monad with transformed value
        """
        return Identity(function(self.value))

    def get(self) -> T:
        """Extract the value from the Identity monad."""
        return self.value

    def __repr__(self):
        return f"Identity({self.value})"

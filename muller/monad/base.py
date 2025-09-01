from typing import Callable, TypeVar, cast
from pymonad.monad import Monad

S = TypeVar("S", contravariant=True)

class ParametrizedMonad[T](Monad[T]):
    """
    Base class for parametrized monads.
    Empty for now but will be used in the future.
    """

    def bind(self, kleisli_function: Callable[[T], 'ParametrizedMonad[S]']) -> 'ParametrizedMonad[S]':  # pyright: ignore[reportIncompatibleMethodOverride] # fmt: skip # noqa: E501
        return cast('ParametrizedMonad[S]', super().bind(kleisli_function))

    @classmethod
    def insert(cls, value: T) -> 'ParametrizedMonad[T]':
        return cast('ParametrizedMonad[T]', super().insert(value))

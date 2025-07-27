from typing import Callable, cast
from pymonad.monad import Monad

ParametrizedMonad = Monad

def bind[T, S](monad: Monad[T], kleisli_function: Callable[[T], Monad[S]]) -> Monad[S]:
    """Applies 'kleisli_function' to the value inside 'monad', returning a new monad value."""
    return cast(Monad[S], monad.bind(kleisli_function))  # pyright: ignore[reportArgumentType]


def bindT[A, B, T: Monad](monad: T, m1: Monad[A], kleisli_function: Callable[[A], Monad[B]]) -> T:
    """Applies 'kleisli_function' to the value inside 'monad', returning a new monad value."""
    assert(monad == m1)
    return cast(T, monad.bind(kleisli_function))  # pyright: ignore[reportArgumentType]



def fmap[T, S](monad: Monad[T], function: Callable[[T], S]) -> Monad[S]:
    """Applies 'function' to the value inside 'monad', returning a new monad value."""
    return cast(Monad[S], monad.map(function))  # pyright: ignore[reportArgumentType]

def fmapT[A, B, T:Monad](monad: T, m1: Monad[A], function: Callable[[A], B]) -> T:
    """Applies 'function' to the value inside 'monad', returning a new monad value."""
    assert(monad == m1)
    return cast(T, monad.map(function))  # pyright: ignore[reportArgumentType]
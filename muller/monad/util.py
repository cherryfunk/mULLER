from typing import Callable, cast

from pymonad.monad import Monad

ParametrizedMonad = Monad


def bind[T, S](monad: Monad[T], kleisli_function: Callable[[T], Monad[S]]) -> Monad[S]:
    """Applies 'kleisli_function' to the value inside 'monad', returning a new
    monad value but with proper typings."""
    return cast(Monad[S], monad.bind(kleisli_function))  # pyright: ignore[reportArgumentType] # fmt: skip


def bind_T[A, B, T: Monad](
    monad: T, m1: Monad[A], kleisli_function: Callable[[A], Monad[B]]
) -> T:
    """Applies 'kleisli_function' to the value inside 'monad', returning a new monad value

    Helper function for typings.

    Args:
        monad: The monad to apply the function to
        m1: same as 'monad', used for type hints
        kleisli_function: The function to apply to the value inside 'monad'
    """
    assert monad == m1
    return cast(T, monad.bind(kleisli_function))  # pyright: ignore[reportArgumentType]


def fmap[T, S](monad: Monad[T], function: Callable[[T], S]) -> Monad[S]:
    """Applies 'function' to the value inside 'monad', returning a new
    monad value but with proper typings."""
    return cast(Monad[S], monad.map(function))  # pyright: ignore[reportArgumentType]


def fmap_T[A, B, T: Monad](monad: T, m1: Monad[A], function: Callable[[A], B]) -> T:
    """Applies 'function' to the value inside 'monad', returning a new monad value.

    Helper function for typings.

    Args:
        monad: The monad to apply the function to
        m1: same as 'monad', used for type hints
        function: The function to apply to the value inside 'monad'
    """
    assert monad == m1
    return cast(T, monad.map(function))  # pyright: ignore[reportArgumentType]

from typing import Callable, cast

from muller.monad.base import ParametrizedMonad


def bind[T, S](
    monad: ParametrizedMonad[T], kleisli_function: Callable[[T], ParametrizedMonad[S]]
) -> ParametrizedMonad[S]:
    """Applies 'kleisli_function' to the value inside 'monad', returning a new
    monad value but with proper typings."""
    return cast(ParametrizedMonad[S], monad.bind(kleisli_function))  # pyright: ignore[reportArgumentType] # fmt: skip


def bind_T[A, B, T: ParametrizedMonad](
    monad: T,
    m1: ParametrizedMonad[A],
    kleisli_function: Callable[[A], ParametrizedMonad[B]],
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


def fmap[T, S](
    monad: ParametrizedMonad[T], function: Callable[[T], S]
) -> ParametrizedMonad[S]:
    """Applies 'function' to the value inside 'monad', returning a new
    monad value but with proper typings."""
    return cast(ParametrizedMonad[S], monad.map(function))  # pyright: ignore[reportArgumentType] # fmt: skip


def fmap_T[A, B, T: ParametrizedMonad](
    monad: T, m1: ParametrizedMonad[A], function: Callable[[A], B]
) -> T:
    """Applies 'function' to the value inside 'monad', returning a new monad value.

    Helper function for typings.

    Args:
        monad: The monad to apply the function to
        m1: same as 'monad', used for type hints
        function: The function to apply to the value inside 'monad'
    """
    assert monad == m1
    return cast(T, monad.map(function))  # pyright: ignore[reportArgumentType]

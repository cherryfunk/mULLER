from collections.abc import Callable

from .framework.nesy import BaseNeSyFramework

from .framework.interpretation import Interpretation

from .hkt import List
from .monad import (
    Identity,
    NonEmptyPowerset,
    Dist,
    bernoulli,
    from_list,
    singleton,
    uniform,
    weighted,
)
from .nesy_framework import nesy, nesy_for_logic
from .parser import parse


def fn[T](func: T) -> T:
    """Decorator to mark a method as a function in the interpretation."""
    func.__muller_type = "fn"  # type: ignore[attr-defined]
    return func


def compfn[T](func: T) -> T:
    """Decorator to mark a method as a computational function in the interpretation."""
    func.__muller_type = "compfn"  # type: ignore[attr-defined]
    return func


def pred[T](func: T) -> T:
    """Decorator to mark a method as a predicate in the interpretation."""
    func.__muller_type = "pred"  # type: ignore[attr-defined]
    return func


def comppred[T](func: T) -> T:
    """Decorator to mark a method as a computational predicate in the interpretation."""
    func.__muller_type = "comppred"  # type: ignore[attr-defined]
    return func


__all__ = [
    "fn",
    "compfn",
    "pred",
    "comppred",
    "BaseNeSyFramework",
    "Interpretation",
    "nesy",
    "nesy_for_logic",
    "parse",
    "Dist",
    "NonEmptyPowerset",
    "Identity",
    "List",
    "from_list",
    "singleton",
    "uniform",
    "weighted",
    "bernoulli",
]

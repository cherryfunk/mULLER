from abc import ABC, abstractmethod
from functools import reduce
from typing import Annotated, Any, Callable, Generic, Iterable, List, Type, TypeVar, cast
import typing
import types

from muller.monad.base import ParametrizedMonad
from muller.monad.giry_pymc import GiryMonadPyMC
from muller.monad.identity import Identity


class DblSGrpBLat[T: ParametrizedMonad](ABC):
    @abstractmethod
    def top(self) -> T: ...

    @abstractmethod
    def bottom(self) -> T: ...

    def neg(self, a: T) -> T:
        return self.implies(a, self.bottom())

    @abstractmethod
    def conjunction(self, a: T, b: T) -> T: ...

    @abstractmethod
    def disjunction(self, a: T, b: T) -> T: ...

    def implies(self, a: T, b: T) -> T:
        return self.disjunction(self.neg(a), b)


class Aggr2SGrpBLat[S, T: ParametrizedMonad](DblSGrpBLat[T]):
    @abstractmethod
    def aggrE[A](self, structure: S, f: Callable[[A], T]) -> T:
        raise NotImplementedError

    @abstractmethod
    def aggrA[A](self, structure: S, f: Callable[[A], T]) -> T:
        raise NotImplementedError


class NeSyLogicMeta[S](ABC):
    def as_base(self) -> "DblSGrpBLat[ParametrizedMonad[S]]":
        """Cast this instance to DblSGrpBLat[ParametrizedMonad[S]]"""
        return cast("DblSGrpBLat[ParametrizedMonad[S]]", self)


def with_list_structure[T: ParametrizedMonad, O](
    cls: Type[DblSGrpBLat[T]], t: Type[T], o: Type[O], name=None
) -> Type[Aggr2SGrpBLat[list[Any], T]]:
    if name is None:
        name = f"{cls.__name__}List"

    def aggrE[A](
        self, structure: list[A], f: Callable[[A], T]
    ) -> T:
        return reduce(
            lambda a, b: self.disjunction(a, b), map(f, structure), self.bottom()
        )

    def aggrA[A](
        self, structure: list[A], f: Callable[[A], T]
    ) -> T:
        return reduce(
            lambda a, b: self.conjunction(a, b), map(f, structure), self.top()
        )

    new_class = types.new_class(
        name,
        (cls, Aggr2SGrpBLat[list, T]),
        {},
        lambda ns: ns.update({"aggrE": aggrE, "aggrA": aggrA}),
    )
    
    new_class.__annotations__ = {"S": List[Any], "T": t, "O": o}

    return new_class


def with_prob_structure[T: ParametrizedMonad, O](
    cls: Type[DblSGrpBLat[T]], 
    t: Type[T],
    o: Type[O],
    name=None
) -> Type[Aggr2SGrpBLat[GiryMonadPyMC, T]]:
    if name is None:
        name = f"{cls.__name__}Prob"

    def aggrE[A](
        self, structure: GiryMonadPyMC, f: Callable[[A], T]
    ) -> T:
        samples = structure.sample(1000)
        return reduce(
            lambda a, b: self.disjunction(a, b), map(f, samples), self.bottom()
        )

    def aggrA[A](
        self, structure: GiryMonadPyMC, f: Callable[[A], T]
    ) -> T:
        samples = structure.sample(1000)
        return reduce(lambda a, b: self.conjunction(a, b), map(f, samples), self.top())

    new_class = types.new_class(
        name,
        (cls, Aggr2SGrpBLat[GiryMonadPyMC, T]),
        {},
        lambda ns: ns.update({"aggrE": aggrE, "aggrA": aggrA}),
    )

    new_class.__annotations__ = {"S": GiryMonadPyMC, "T": t, "O": o}

    return new_class

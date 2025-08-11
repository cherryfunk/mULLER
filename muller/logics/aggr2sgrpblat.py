from abc import ABC, abstractmethod
from functools import reduce
from typing import Generic, Iterable, cast, TypeVar

from muller.monad.base import ParametrizedMonad

T = TypeVar("T", bound=ParametrizedMonad)

class Aggr2SGrpBLat(ABC, Generic[T]):
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

    def aggrE(self, s: Iterable[T]) -> T:
        return reduce(lambda a, b: self.disjunction(a, b), s, self.bottom())

    def aggrA(self, s: Iterable[T]) -> T:
        return reduce(lambda a, b: self.conjunction(a, b), s, self.top())


class NeSyLogicMeta[S](ABC):
    def as_base(self) -> "Aggr2SGrpBLat[ParametrizedMonad[S]]":
        """Cast this instance to Aggr2SGrpBLat[ParametrizedMonad[S]]"""
        return cast("Aggr2SGrpBLat[ParametrizedMonad[S]]", self)

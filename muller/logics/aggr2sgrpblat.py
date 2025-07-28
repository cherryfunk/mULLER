from abc import ABC, abstractmethod
from functools import reduce
from typing import Iterable, cast

from pymonad.monad import Monad


class Aggr2SGrpBLat[T](ABC):
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


class NeSyLogicMeta[T](ABC):
    def as_base(self) -> "Aggr2SGrpBLat[Monad[T]]":
        """Cast this instance to Aggr2SGrpBLat[Monad[T]]"""
        return cast("Aggr2SGrpBLat[Monad[T]]", self)

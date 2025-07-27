# Monad T
# con :: Monad t, DSGBL (t O) => t O -> t O -> t O


import abc
from functools import reduce
from itertools import count
import sys
from typing import Generic, Iterable, Literal, Self, TypeAlias, TypeVar, cast
from pymonad.monad import Monad

from abc import ABC, abstractmethod

from src.muller.monad.util import bind, bindT, fmap, fmapT

from src.muller.monad.non_empty_powerset import NonEmptyPowerset, singleton, from_list
from src.muller.monad.distribution import Prob


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

class NonEmptyPowersetBoolDoubleSemiGroupBoundedLattice(Aggr2SGrpBLat[NonEmptyPowerset[bool]]):
    def top(self) -> NonEmptyPowerset[bool]:
        return singleton(True)

    def bottom(self) -> NonEmptyPowerset[bool]:
        return singleton(False)

    def neg(self, a: NonEmptyPowerset[bool]) -> NonEmptyPowerset[bool]:
        return from_list([not x for x in a.value])

    def conjunction(self, a: NonEmptyPowerset[bool], b: NonEmptyPowerset[bool]) -> NonEmptyPowerset[bool]:
        return from_list([x and y for x in a.value for y in b.value])

    def disjunction(self, a: NonEmptyPowerset[bool], b: NonEmptyPowerset[bool]) -> NonEmptyPowerset[bool]:
        return from_list([x or y for x in a.value for y in b.value])


Priest = Literal[True, False, "Both"]

def priest_conjunction(a: Priest, b: Priest) -> Priest:
    if a == "Both" or b == "Both":
        return "Both"
    return a and b

def priest_disjunction(a: Priest, b: Priest) -> Priest:
    if a == "Both" or b == "Both":
        return "Both"
    return a or b

def priest_negation(a: Priest) -> Priest:
    if a == "Both":
        return "Both"
    return not a

def priest_implication(a: Priest, b: Priest) -> Priest:
    if a == "Both":
        return "Both"
    if b == "Both":
        return "Both"
    return not a or b



class NonEmptyPowersetPriestDoubleSemiGroupBoundedLattice(Aggr2SGrpBLat[NonEmptyPowerset[Priest]]):
    def top(self) -> NonEmptyPowerset[Priest]:
        return singleton("Both")

    def bottom(self) -> NonEmptyPowerset[Priest]:
        return singleton(False)

    def neg(self, a: NonEmptyPowerset[Priest]) -> NonEmptyPowerset[Priest]:
        return from_list([priest_negation(x) for x in a.value])

    def conjunction(self, a: NonEmptyPowerset[Priest], b: NonEmptyPowerset[Priest]) -> NonEmptyPowerset[Priest]:
        return from_list([priest_conjunction(x,y) for x in a.value for y in b.value])

    def disjunction(self, a: NonEmptyPowerset[Priest], b: NonEmptyPowerset[Priest]) -> NonEmptyPowerset[Priest]:
        return from_list([priest_disjunction(x,y) for x in a.value for y in b.value])

class DistributionBoolDoubleSemiGroupBoundedLattice(Aggr2SGrpBLat[Prob[bool]]):
    def top(self) -> Prob[bool]:
        return Prob({True: 1.0, False: 0.0})

    def bottom(self) -> Prob[bool]:
        return Prob({False: 1.0, True: 0.0})

    def neg(self, a: Prob[bool]) -> Prob[bool]:
        return fmapT(a, a, lambda x: not x)

    def conjunction(self, a: Prob[bool], b: Prob[bool]) -> Prob[bool]:
        return bindT(a, a, lambda x: fmapT(b, b, lambda y: x and y))

    def disjunction(self, a: Prob[bool], b: Prob[bool]) -> Prob[bool]:
        return bindT(a, a, lambda x: fmapT(b, b, lambda y: x or y))


def of[O](monad: type[Monad[O]], type: O) -> Aggr2SGrpBLat[Monad[O]]:
    current_module = sys.modules[__name__]
    member = getattr(current_module, "non_empty_set_bool", None)
    if member is not None:
        return member
    raise ValueError(f"No implementation for monad {monad} and truth basis {type} found.")

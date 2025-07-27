from typing import Literal

from muller.aggr2sgrpblat.aggr2sgrpblat import Aggr2SGrpBLat
from muller.monad.non_empty_powerset import NonEmptyPowerset, from_list, singleton


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
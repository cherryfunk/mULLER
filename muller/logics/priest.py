from typing import Literal

from muller.logics.aggr2sgrpblat import Aggr2SGrpBLat, DblSGrpBLat, with_list_structure, with_prob_structure
from muller.monad.identity import Identity
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


class ClassicalPriestLogic(DblSGrpBLat[Identity[Priest]]):
    def top(self) -> Identity[Priest]:
        return Identity.insert("Both")

    def bottom(self) -> Identity[Priest]:
        return Identity.insert(False)

    def neg(self, a: Identity[Priest]) -> Identity[Priest]:
        return Identity.insert(priest_negation(a.value))

    def conjunction(self, a: Identity[Priest], b: Identity[Priest]) -> Identity[Priest]:
        return Identity.insert(priest_conjunction(a.value, b.value))

    def disjunction(self, a: Identity[Priest], b: Identity[Priest]) -> Identity[Priest]:
        return Identity.insert(priest_disjunction(a.value, b.value))
    

ClassicalPriestLogicList = with_list_structure(ClassicalPriestLogic, Identity, Priest)
ClassicalPriestLogicProb = with_prob_structure(ClassicalPriestLogic, Identity, Priest)


class NonDeterministicPriestLogic(DblSGrpBLat[NonEmptyPowerset[Priest]]):
    def top(self) -> NonEmptyPowerset[Priest]:
        return singleton("Both")

    def bottom(self) -> NonEmptyPowerset[Priest]:
        return singleton(False)

    def neg(self, a: NonEmptyPowerset[Priest]) -> NonEmptyPowerset[Priest]:
        return from_list([priest_negation(x) for x in a.value])

    def conjunction(
        self, a: NonEmptyPowerset[Priest], b: NonEmptyPowerset[Priest]
    ) -> NonEmptyPowerset[Priest]:
        return from_list([priest_conjunction(x, y) for x in a.value for y in b.value])

    def disjunction(
        self, a: NonEmptyPowerset[Priest], b: NonEmptyPowerset[Priest]
    ) -> NonEmptyPowerset[Priest]:
        return from_list([priest_disjunction(x, y) for x in a.value for y in b.value])

NonDeterministicPriestLogicList = with_list_structure(NonDeterministicPriestLogic, NonEmptyPowerset, type(Priest))
NonDeterministicPriestLogicProb = with_prob_structure(NonDeterministicPriestLogic, NonEmptyPowerset, Priest)


class ProbabilisticPriestLogic(DblSGrpBLat[NonEmptyPowerset[Priest]]):
    def top(self) -> NonEmptyPowerset[Priest]:
        return singleton("Both")

    def bottom(self) -> NonEmptyPowerset[Priest]:
        return singleton(False)

    def neg(self, a: NonEmptyPowerset[Priest]) -> NonEmptyPowerset[Priest]:
        return from_list([priest_negation(x) for x in a.value])

    def conjunction(
        self, a: NonEmptyPowerset[Priest], b: NonEmptyPowerset[Priest]
    ) -> NonEmptyPowerset[Priest]:
        return from_list([priest_conjunction(x, y) for x in a.value for y in b.value])

    def disjunction(
        self, a: NonEmptyPowerset[Priest], b: NonEmptyPowerset[Priest]
    ) -> NonEmptyPowerset[Priest]:
        return from_list([priest_disjunction(x, y) for x in a.value for y in b.value])


ProbabilisticPriestLogicList = with_list_structure(ProbabilisticPriestLogic, NonEmptyPowerset, Priest)
ProbabilisticPriestLogicProb = with_prob_structure(ProbabilisticPriestLogic, NonEmptyPowerset, Priest)

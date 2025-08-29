from functools import reduce
from typing import Callable
from muller.logics.aggr2sgrpblat import Aggr2SGrpBLat, DblSGrpBLat, NeSyLogicMeta, with_list_structure, with_prob_structure
from muller.monad.distribution import Prob
from muller.monad.identity import Identity
from muller.monad.non_empty_powerset import NonEmptyPowerset, from_list, singleton
from muller.monad.util import bind_T, fmap_T
from muller.monad.giry_pymc import GiryMonadPyMC

class ClassicalBooleanLogic(DblSGrpBLat[Identity[bool]], NeSyLogicMeta[bool]):
    def top(self) -> Identity[bool]:
        return Identity.insert(True)

    def bottom(self) -> Identity[bool]:
        return Identity.insert(False)

    def neg(self, a: Identity[bool]) -> Identity[bool]:
        return Identity.insert(not a.value)

    def conjunction(self, a: Identity[bool], b: Identity[bool]) -> Identity[bool]:
        return Identity.insert(a.value and b.value)

    def disjunction(self, a: Identity[bool], b: Identity[bool]) -> Identity[bool]:
        return Identity.insert(a.value or b.value)

ClassicalBooleanLogicList = with_list_structure(Identity, bool)(ClassicalBooleanLogic)
ClassicalBooleanLogicProb = with_prob_structure(Identity, bool)(ClassicalBooleanLogic)


class NonDeterministicBooleanLogic(
    DblSGrpBLat[NonEmptyPowerset[bool]]
):
    def top(self) -> NonEmptyPowerset[bool]:
        return singleton(True)

    def bottom(self) -> NonEmptyPowerset[bool]:
        return singleton(False)

    def neg(self, a: NonEmptyPowerset[bool]) -> NonEmptyPowerset[bool]:
        return from_list([not x for x in a.value])

    def conjunction(
        self, a: NonEmptyPowerset[bool], b: NonEmptyPowerset[bool]
    ) -> NonEmptyPowerset[bool]:
        return from_list([x and y for x in a.value for y in b.value])

    def disjunction(
        self, a: NonEmptyPowerset[bool], b: NonEmptyPowerset[bool]
    ) -> NonEmptyPowerset[bool]:
        return from_list([x or y for x in a.value for y in b.value])

NonDeterministicBooleanLogicList = with_list_structure(NonEmptyPowerset, bool)(NonDeterministicBooleanLogic)
NonDeterministicBooleanLogicProb = with_prob_structure(NonEmptyPowerset, bool)(NonDeterministicBooleanLogic)


class ProbabilisticBooleanLogic(DblSGrpBLat[Prob[bool]], NeSyLogicMeta[bool]):
    def top(self) -> Prob[bool]:
        return Prob({True: 1.0, False: 0.0})

    def bottom(self) -> Prob[bool]:
        return Prob({False: 1.0, True: 0.0})

    def neg(self, a: Prob[bool]) -> Prob[bool]:
        return fmap_T(a, a, lambda x: not x)

    def conjunction(self, a: Prob[bool], b: Prob[bool]) -> Prob[bool]:
        return bind_T(a, a, lambda x: fmap_T(b, b, lambda y: x and y))

    def disjunction(self, a: Prob[bool], b: Prob[bool]) -> Prob[bool]:
        return bind_T(a, a, lambda x: fmap_T(b, b, lambda y: x or y))

ProbabilisticBooleanLogicList = with_list_structure(Prob, bool)(ProbabilisticBooleanLogic)
ProbabilisticBooleanLogicProb = with_prob_structure(Prob, bool)(ProbabilisticBooleanLogic)

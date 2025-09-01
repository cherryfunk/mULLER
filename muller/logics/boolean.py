from typing import Type, cast

from muller.logics.aggr2sgrpblat import (
    DblSGrpBLat,
    NeSyLogicMeta,
    with_list_structure,
    with_prob_structure,
)
from muller.monad.base import ParametrizedMonad
from muller.monad.distribution import Prob
from muller.monad.giry_sampling import GirySampling
from muller.monad.identity import Identity
from muller.monad.non_empty_powerset import NonEmptyPowerset
from muller.monad.util import bind_T, fmap_T


def boolean_logic[T: ParametrizedMonad[bool]](monad: Type[T]) -> Type[DblSGrpBLat[T]]:
    class _BooleanLogic(DblSGrpBLat[T], NeSyLogicMeta[bool]):
        def top(self) -> T:
            return cast(T, monad.insert(True))

        def bottom(self) -> T:
            return cast(T, monad.insert(False))

        def neg(self, a: T) -> T:
            return fmap_T(a, a, lambda x: not x)

        def conjunction(self, a: T, b: T) -> T:
            return bind_T(a, a, lambda x: self.bottom() if not x else b)

        def disjunction(self, a: T, b: T) -> T:
            return bind_T(a, a, lambda x: self.top() if x else b)

    return _BooleanLogic

ClassicalBooleanLogic = boolean_logic(Identity[bool])
ClassicalBooleanLogicList = with_list_structure(Identity, bool)(ClassicalBooleanLogic)
ClassicalBooleanLogicProb = with_prob_structure(Identity, bool)(ClassicalBooleanLogic)

NonDeterministicBooleanLogic = boolean_logic(NonEmptyPowerset[bool])
NonDeterministicBooleanLogicList = with_list_structure(NonEmptyPowerset, bool)(NonDeterministicBooleanLogic) # noqa: E501
NonDeterministicBooleanLogicProb = with_prob_structure(NonEmptyPowerset, bool)(NonDeterministicBooleanLogic) # noqa: E501

ProbabilisticBooleanLogic = boolean_logic(Prob[bool])
ProbabilisticBooleanLogicList = with_list_structure(Prob, bool)(ProbabilisticBooleanLogic)
ProbabilisticBooleanLogicProb = with_prob_structure(Prob, bool)(ProbabilisticBooleanLogic)

GiryBooleanLogic = boolean_logic(GirySampling[bool])
GiryBooleanLogicList = with_list_structure(GirySampling, bool)(GiryBooleanLogic)
GiryBooleanLogicProb = with_prob_structure(GirySampling, bool)(GiryBooleanLogic)

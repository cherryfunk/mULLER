from typing import Literal, Type, cast

from muller.logics.aggr2sgrpblat import (
    DblSGrpBLat,
    with_list_structure,
    with_prob_structure,
)
from muller.monad.base import ParametrizedMonad
from muller.monad.identity import Identity
from muller.monad.non_empty_powerset import NonEmptyPowerset
from muller.monad.util import bind_T

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


def priest_logic[T: ParametrizedMonad[Priest]](monad: Type[T]) -> Type[DblSGrpBLat[T]]:
    class _PriestLogic(DblSGrpBLat[T]):
        def top(self) -> T:
            return cast(T, monad.insert("Both"))

        def bottom(self) -> T:
            return cast(T, monad.insert(False))

        def neg(self, a: T) -> T:
            return bind_T(a, a, lambda x: monad.insert(priest_negation(x)))

        def conjunction(self, a: T, b: T) -> T:
            return bind_T(a, a, lambda x: a if x != True else b)  # noqa: E712

        def disjunction(self, a: T, b: T) -> T:
            return bind_T(a, a, lambda x: a if x != False else b)  # noqa: E712

    return _PriestLogic


ClassicalPriestLogic = priest_logic(Identity[Priest])
ClassicalPriestLogicList = with_list_structure(Identity, Priest)(ClassicalPriestLogic)
ClassicalPriestLogicProb = with_prob_structure(Identity, Priest)(ClassicalPriestLogic)


NonDeterministicPriestLogic = priest_logic(NonEmptyPowerset[Priest])
NonDeterministicPriestLogicList = with_list_structure(NonEmptyPowerset, Priest)(
    NonDeterministicPriestLogic
)
NonDeterministicPriestLogicProb = with_prob_structure(NonEmptyPowerset, Priest)(
    NonDeterministicPriestLogic
)


ProbabilisticPriestLogic = priest_logic(NonEmptyPowerset[Priest])
ProbabilisticPriestLogicList = with_list_structure(NonEmptyPowerset, Priest)(
    ProbabilisticPriestLogic
)
ProbabilisticPriestLogicProb = with_prob_structure(NonEmptyPowerset, Priest)(
    ProbabilisticPriestLogic
)

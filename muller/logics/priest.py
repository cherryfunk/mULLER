# mypy: disable-error-code="type-arg"
"""
Priest Logic (LP - Logic of Paradox) implementations for various monad types.

Priest logic is a three-valued logic with values True, False, and "Both" (paradox).
"""

from typing import Any, Literal, TypeVar

from returns.interfaces.container import Container1
from returns.primitives.hkt import Kind1

from muller.logics.aggr2sgrpblat import DblSGrpBLat  # GirySamplingAggregationMixin,,

_ObjectType = TypeVar("_ObjectType")
_MonadType = TypeVar("_MonadType", bound=Container1[Any])

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


# =============================================================================
# Identity Monad Priest Logic (Classical/Deterministic)
# =============================================================================


class PriestLogic(DblSGrpBLat[_MonadType, Priest]):
    """Classical Priest logic with Identity monad (deterministic)."""

    def top(self) -> Kind1[_MonadType, Priest]:
        return self.monad_from_value("Both")

    def bottom(self) -> Kind1[_MonadType, Priest]:
        return self.monad_from_value(False)

    def neg(self, a: Kind1[_MonadType, Priest]) -> Kind1[_MonadType, Priest]:
        return a.map(priest_negation)

    def conjunction(
        self, a: Kind1[_MonadType, Priest], b: Kind1[_MonadType, Priest]
    ) -> Kind1[_MonadType, Priest]:
        return a.bind(lambda x: self.bottom() if x is False else b)

    def disjunction(
        self, a: Kind1[_MonadType, Priest], b: Kind1[_MonadType, Priest]
    ) -> Kind1[_MonadType, Priest]:
        return a.bind(lambda x: self.top() if x is True else b)


# class IdentityPriestLogicList(
#     ListAggregationMixin[Identity[Priest], Priest, _ObjectType],
#     PriestLogic[Identity[Priest]],
# ):
#     """Classical Priest logic with list aggregation."""

#     pass


# class IdentityPriestLogicGiry(
#     GirySamplingAggregationMixin[Kind1["Identity", Priest], Priest],
#     IdentityPriestLogic,
# ):
#     """Classical Priest logic with Giry sampling aggregation."""

#     pass


# =============================================================================
# NonEmptyPowerset Monad Priest Logic (Non-deterministic)
# =============================================================================


# class NonEmptyPowersetPriestLogic(
#     DblSGrpBLat[NonEmptyPowerset[Priest], Priest], NeSyLogicMeta[Priest]
# ):
#     """Non-deterministic Priest logic with NonEmptyPowerset monad."""

#     def top(self) -> Kind1["NonEmptyPowerset", Priest]:
#         return NonEmptyPowerset.from_value("Both")

#     def bottom(self) -> Kind1["NonEmptyPowerset", Priest]:
#         return NonEmptyPowerset.from_value(False)

#     def neg(self, a: Kind1["NonEmptyPowerset", Priest]) -> Kind1["NonEmptyPowerset", Priest]:
#         return a.map(priest_negation)

#     def conjunction(
#         self, a: Kind1["NonEmptyPowerset", Priest], b: Kind1["NonEmptyPowerset", Priest]
#     ) -> Kind1["NonEmptyPowerset", Priest]:
#         return a.bind(lambda x: self.bottom() if x is False else b)

#     def disjunction(
#         self, a: Kind1["NonEmptyPowerset", Priest], b: Kind1["NonEmptyPowerset", Priest]
#     ) -> Kind1["NonEmptyPowerset", Priest]:
#         return a.bind(lambda x: self.top() if x is True else b)


# class NonEmptyPowersetPriestLogicList(
#     ListAggregationMixin[Kind1["NonEmptyPowerset", Priest], Priest, _ObjectType],
#     NonEmptyPowersetPriestLogic,
# ):
#     """Non-deterministic Priest logic with list aggregation."""

#     pass


# class NonEmptyPowersetPriestLogicGiry(
#     GirySamplingAggregationMixin[Kind1["NonEmptyPowerset", Priest], Priest],
#     NonEmptyPowersetPriestLogic,
# ):
#     """Non-deterministic Priest logic with Giry sampling aggregation."""

#     pass


# =============================================================================
# Backward compatibility aliases
# =============================================================================

# ClassicalPriestLogic = IdentityPriestLogic
# ClassicalPriestLogicList = IdentityPriestLogicList
# # ClassicalPriestLogicProb = IdentityPriestLogicGiry

# NonDeterministicPriestLogic = NonEmptyPowersetPriestLogic
# NonDeterministicPriestLogicList = NonEmptyPowersetPriestLogicList
# # NonDeterministicPriestLogicProb = NonEmptyPowersetPriestLogicGiry

# ProbabilisticPriestLogic = NonEmptyPowersetPriestLogic
# ProbabilisticPriestLogicList = NonEmptyPowersetPriestLogicList
# # ProbabilisticPriestLogicProb = NonEmptyPowersetPriestLogicGiry

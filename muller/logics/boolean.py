# mypy: disable-error-code="type-arg"
"""
Boolean Logic implementations for various monad types.

Each class is a concrete implementation for a specific monad type,
allowing proper type checking with the returns library's HKT system.
"""

from typing import Any, Generic, TypeVar

from returns.interfaces.container import Container1
from returns.primitives.hkt import Kind1, dekind

from muller.logics.aggr2sgrpblat import (
    DblSGrpBLat,
    # GirySamplingAggregationMixin,
    ListAggregationMixin
)
from muller.monad.distribution import Prob
from muller.monad.giry_sampling import GirySampling
from muller.monad.identity import Identity
from muller.monad.non_empty_powerset import NonEmptyPowerset

# =============================================================================
# Identity Monad Boolean Logic (Classical/Deterministic)
# =============================================================================

_ObjectType = TypeVar("_ObjectType")
_MonadType = TypeVar("_MonadType", bound=Container1[Any])


class BooleanLogic(Generic[_MonadType], DblSGrpBLat[_MonadType, bool]):
    """Classical boolean logic with Identity monad (deterministic)."""

    def top(self) -> Kind1[_MonadType, bool]:
        return self.monad_from_value(True)

    def bottom(self) -> Kind1[_MonadType, bool]:
        return self.monad_from_value(False)

    def neg(self, a: Kind1[_MonadType, bool]) -> Kind1[_MonadType, bool]:
        return a.map(lambda x: not x)

    def conjunction(
        self, a: Kind1[_MonadType, bool], b: Kind1[_MonadType, bool]
    ) -> Kind1[_MonadType, bool]:
        return a.bind(lambda x: self.bottom() if not x else b)

    def disjunction(
        self, a: Kind1[_MonadType, bool], b: Kind1[_MonadType, bool]
    ) -> Kind1[_MonadType, bool]:
        return a.bind(lambda x: self.top() if x else b)


class IdentityBooleanLogicList(
    ListAggregationMixin[Identity[bool], bool, _ObjectType],
    BooleanLogic[Identity[bool]],
):
    """Classical boolean logic with list aggregation."""

    pass


# class IdentityBooleanLogicGiry(
#     GirySamplingAggregationMixin[Identity[bool], bool],
#     IdentityBooleanLogic,
# ):
#     """Classical boolean logic with Giry sampling aggregation."""

#     pass


# =============================================================================
# NonEmptyPowerset Monad Boolean Logic (Non-deterministic)
# =============================================================================


# class NonEmptyPowersetBooleanLogic(
#     DblSGrpBLat[NonEmptyPowerset[bool], bool], NeSyLogicMeta[bool]
# ):
#     """Non-deterministic boolean logic with NonEmptyPowerset monad."""

#     def top(self) -> Kind1["NonEmptyPowerset", bool]:
#         return NonEmptyPowerset.from_value(True)

#     def bottom(self) -> Kind1["NonEmptyPowerset", bool]:
#         return NonEmptyPowerset.from_value(False)

#     def neg(self, a: Kind1["NonEmptyPowerset", bool]) -> Kind1["NonEmptyPowerset", bool]:
#         return a.map(lambda x: not x)

#     def conjunction(
#         self, a: Kind1["NonEmptyPowerset", bool], b: Kind1["NonEmptyPowerset", bool]
#     ) -> Kind1["NonEmptyPowerset", bool]:
#         return a.bind(lambda x: self.bottom() if not x else b)

#     def disjunction(
#         self, a: Kind1["NonEmptyPowerset", bool], b: Kind1["NonEmptyPowerset", bool]
#     ) -> Kind1["NonEmptyPowerset", bool]:
#         return a.bind(lambda x: self.top() if x else b)


# class NonEmptyPowersetBooleanLogicList(
#     ListAggregationMixin[NonEmptyPowerset[bool], bool, _ObjectType],
#     NonEmptyPowersetBooleanLogic,
# ):
#     """Non-deterministic boolean logic with list aggregation."""

#     pass


# class NonEmptyPowersetBooleanLogicGiry(
#     GirySamplingAggregationMixin[NonEmptyPowerset[bool], bool],
#     NonEmptyPowersetBooleanLogic,
# ):
#     """Non-deterministic boolean logic with Giry sampling aggregation."""

#     pass


# # =============================================================================
# # Prob Monad Boolean Logic (Probabilistic)
# # =============================================================================


# class ProbBooleanLogic(DblSGrpBLat[Prob[bool], bool], NeSyLogicMeta[bool]):
#     """Probabilistic boolean logic with Prob monad."""

#     def top(self) -> Kind1["Prob", bool]:
#         return Prob.from_value(True)

#     def bottom(self) -> Kind1["Prob", bool]:
#         return Prob.from_value(False)

#     def neg(self, a: Kind1["Prob", bool]) -> Kind1["Prob", bool]:
#         return a.map(lambda x: not x)

#     def conjunction(
#         self, a: Kind1["Prob", bool], b: Kind1["Prob", bool]
#     ) -> Kind1["Prob", bool]:
#         return a.bind(lambda x: self.bottom() if not x else b)

#     def disjunction(
#         self, a: Kind1["Prob", bool], b: Kind1["Prob", bool]
#     ) -> Kind1["Prob", bool]:
#         return a.bind(lambda x: self.top() if x else b)


class ProbBooleanLogicList(
    ListAggregationMixin[Prob[bool], bool, _ObjectType],
    BooleanLogic[Prob[bool]],
):
    """Probabilistic boolean logic with list aggregation."""

    pass


# # class ProbBooleanLogicGiry(
# #     GirySamplingAggregationMixin[Kind1["Prob", bool], bool],
# #     ProbBooleanLogic,
# # ):
# #     """Probabilistic boolean logic with Giry sampling aggregation."""

# #     pass


# # =============================================================================
# # GirySampling Monad Boolean Logic
# # =============================================================================


# class GirySamplingBooleanLogic(
#     DblSGrpBLat[GirySampling[bool], bool], NeSyLogicMeta[bool]
# ):
#     """Boolean logic with GirySampling monad."""

#     def top(self) -> Kind1["GirySampling", bool]:
#         return GirySampling.from_value(True)

#     def bottom(self) -> Kind1["GirySampling", bool]:
#         return GirySampling.from_value(False)

#     def neg(self, a: Kind1["GirySampling", bool]) -> Kind1["GirySampling", bool]:
#         return a.map(lambda x: not x)

#     def conjunction(
#         self, a: Kind1["GirySampling", bool], b: Kind1["GirySampling", bool]
#     ) -> Kind1["GirySampling", bool]:
#         return a.bind(lambda x: self.bottom() if not x else b)

#     def disjunction(
#         self, a: Kind1["GirySampling", bool], b: Kind1["GirySampling", bool]
#     ) -> Kind1["GirySampling", bool]:
#         return a.bind(lambda x: self.top() if x else b)


# class GirySamplingBooleanLogicList(
#     ListAggregationMixin[Kind1["GirySampling", bool], bool, _ObjectType],
#     BooleanLogic[GirySampling[bool]],
# ):
#     """GirySampling boolean logic with list aggregation."""

#     pass


# class GirySamplingBooleanLogicGiry(
#     GirySamplingAggregationMixin[Kind1["GirySampling", bool], bool, _ObjectType],
#     GirySamplingBooleanLogic,
# ):
#     """GirySampling boolean logic with Giry sampling aggregation."""

#     pass


# =============================================================================
# Backward compatibility aliases
# =============================================================================

# ClassicalBooleanLogic = IdentityBooleanLogic
# ClassicalBooleanLogicList = IdentityBooleanLogicList
# # ClassicalBooleanLogicProb = IdentityBooleanLogicGiry

# NonDeterministicBooleanLogic = NonEmptyPowersetBooleanLogic
# NonDeterministicBooleanLogicList = NonEmptyPowersetBooleanLogicList
# # NonDeterministicBooleanLogicProb = NonEmptyPowersetBooleanLogicGiry

# ProbabilisticBooleanLogic = ProbBooleanLogic
# ProbabilisticBooleanLogicList = ProbBooleanLogicList
# # ProbabilisticBooleanLogicProb = ProbBooleanLogicGiry

# GiryBooleanLogic = GirySamplingBooleanLogic
# GiryBooleanLogicList = GirySamplingBooleanLogicList
# # GiryBooleanLogicProb = GirySamplingBooleanLogicGiry

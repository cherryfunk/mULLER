from abc import ABC, abstractmethod
from functools import reduce
from typing import Any, Callable, Generic, TypeVar

from returns.interfaces.container import Container1
from returns.primitives.hkt import Kind1

from muller.monad.giry_sampling import GirySampling

# Type variables for monads and value types
_TruthType = TypeVar("_TruthType")  # Value type (omega)
_MonadType = TypeVar("_MonadType", bound=Container1[Any])  # Monad type constructor


class DblSGrpBLat(Generic[_MonadType, _TruthType], ABC):
    """
    Double-sided Semigroup Boolean Lattice over Monad M with value type O.

    M is the monad type (e.g., Identity[bool], Powerset[bool])
    O is the value type (e.g., bool, Priest)

    Implementations should use concrete monad types (Identity[bool], etc.)
    which properly support the returns library HKT via SupportsKind1.
    """

    def __init__(
        self, monad_from_value: Callable[[_TruthType], Kind1[_MonadType, _TruthType]]
    ):
        self.monad_from_value = monad_from_value

    # def monad_from_value(self, value: _TruthType) -> Kind1[_MonadType, _TruthType]:
    #     """Helper to lift a pure value into the monad."""
    #     return self._monad_from_value(value)

    @abstractmethod
    def top(self) -> Kind1[_MonadType, _TruthType]: ...

    @abstractmethod
    def bottom(self) -> Kind1[_MonadType, _TruthType]: ...

    @abstractmethod
    def neg(self, a: Kind1[_MonadType, _TruthType]) -> Kind1[_MonadType, _TruthType]: ...

    @abstractmethod
    def conjunction(
        self, a: Kind1[_MonadType, _TruthType], b: Kind1[_MonadType, _TruthType]
    ) -> Kind1[_MonadType, _TruthType]: ...

    @abstractmethod
    def disjunction(
        self, a: Kind1[_MonadType, _TruthType], b: Kind1[_MonadType, _TruthType]
    ) -> Kind1[_MonadType, _TruthType]: ...

    def implies(
        self, a: Kind1[_MonadType, _TruthType], b: Kind1[_MonadType, _TruthType]
    ) -> Kind1[_MonadType, _TruthType]:
        return self.disjunction(self.neg(a), b)


_ObjectType = TypeVar("_ObjectType")
_StructureType = TypeVar("_StructureType")


class Aggr2SGrpBLat(
    DblSGrpBLat[_MonadType, _TruthType],
    Generic[_MonadType, _TruthType, _StructureType, _ObjectType],
):
    """Aggregation-capable extension of DblSGrpBLat."""

    @abstractmethod
    def aggrE(
        self,
        structure: Kind1[_StructureType, _ObjectType],
        f: Callable[[_ObjectType], Kind1[_MonadType, _TruthType]],
    ) -> Kind1[_MonadType, _TruthType]:
        raise NotImplementedError

    @abstractmethod
    def aggrA(
        self,
        structure: Kind1[_StructureType, _ObjectType],
        f: Callable[[_ObjectType], Kind1[_MonadType, _TruthType]],
    ) -> Kind1[_MonadType, _TruthType]:
        raise NotImplementedError


class ListAggregationMixin(Aggr2SGrpBLat[_MonadType, _TruthType, list[Any], _ObjectType]):
    """Mixin that provides list aggregation for any DblSGrpBLat."""

    def aggrE(
        self,
        sort: Kind1[list[Any], _ObjectType],
        f: Callable[[_ObjectType], Kind1[_MonadType, _TruthType]],
    ) -> Kind1[_MonadType, _TruthType]:
        return reduce(
            lambda a, b: self.disjunction(a, b),
            map(f, sort),
            self.bottom(),
        )

    def aggrA(
        self,
        structure: Kind1[list[Any], _ObjectType],
        f: Callable[[_ObjectType], Kind1[_MonadType, _TruthType]],
    ) -> Kind1[_MonadType, _TruthType]:
        return reduce(
            lambda a, b: self.conjunction(a, b),
            map(f, structure),
            self.top(),
        )


class GirySamplingAggregationMixin(
    Aggr2SGrpBLat[_MonadType, _TruthType, GirySampling[Any], _ObjectType]
):
    """Mixin that provides GirySampling aggregation for any DblSGrpBLat."""

    def aggrE(
        self,
        structure: Kind1[GirySampling[Any], _ObjectType],
        f: Callable[[_ObjectType], Kind1[_MonadType, _TruthType]],
    ) -> Kind1[_MonadType, _TruthType]:
        samples = structure.sample(1000)
        return reduce(
            lambda a, b: self.disjunction(a, b),
            map(f, samples),
            self.bottom(),
        )

    def aggrA(
        self,
        structure: Kind1[GirySampling[Any], _ObjectType],
        f: Callable[[_ObjectType], Kind1[_MonadType, _TruthType]],
    ) -> Kind1[_MonadType, _TruthType]:
        samples = structure.sample(1000)
        return reduce(
            lambda a, b: self.conjunction(a, b),
            map(f, samples),
            self.top(),
        )


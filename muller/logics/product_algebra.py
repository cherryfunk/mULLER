# mypy: disable-error-code="type-arg"
from typing import Any, TypeVar

import torch
from returns.interfaces.container import Container1
from returns.primitives.hkt import Kind1

from muller.logics.aggr2sgrpblat import DblSGrpBLat

_MonadType = TypeVar("_MonadType", bound=Container1[Any])


class ProductAlgebraLogic(DblSGrpBLat[_MonadType, float]):
    """
    Product algebra on [0, 1] with:
    - Conjunction: a · b
    - Disjunction: a + b - a · b (probabilistic sum)
    - Negation: 1 - a
    - Implication (residuum of ·): min(1, b / a) with convention 1 if a == 0

    This matches the semantics used in the paper excerpt for the weather examples.
    Truth values are wrapped in the Identity monad to match the existing logic API
    (see boolean.py and priest.py).
    """

    # Lattice bounds ---------------------------------------------------------
    def top(self) -> Kind1[_MonadType, float]:
        return self.monad_from_value(1.0)

    def bottom(self) -> Kind1[_MonadType, float]:
        return self.monad_from_value(0.0)

    # Connectives ------------------------------------------------------------
    def neg(self, a: Kind1[_MonadType, float]) -> Kind1[_MonadType, float]:
        return a.bind(lambda x: self.monad_from_value(1.0 - x))

    def conjunction(
        self, a: Kind1[_MonadType, float], b: Kind1[_MonadType, float]
    ) -> Kind1[_MonadType, float]:
        return a.bind(lambda x: b.bind(lambda y: self.monad_from_value(x * y)))

    def disjunction(
        self, a: Kind1[_MonadType, float], b: Kind1[_MonadType, float]
    ) -> Kind1[_MonadType, float]:
        # probabilistic sum
        return a.bind(lambda x: b.bind(lambda y: self.monad_from_value(x + y - x * y)))

    # Override default material implication with product residuum
    def implies(
        self, a: Kind1[_MonadType, float], b: Kind1[_MonadType, float]
    ) -> Kind1[_MonadType, float]:
        return a.bind(lambda x: b.bind(lambda y: self._residuum(x, y)))

    def _residuum(self, x: float, y: float) -> Kind1[_MonadType, float]:
        if x <= y:
            return self.monad_from_value(1.0)
        if x == 0.0:
            return self.monad_from_value(1.0)
        return self.monad_from_value(min(1.0, y / x))


class ProductTorchAlgebraLogic(DblSGrpBLat[_MonadType, torch.Tensor]):
    """
    Product algebra on [0, 1] with:
    - Conjunction: a · b
    - Disjunction: a + b - a · b (probabilistic sum)
    - Negation: 1 - a
    - Implication (residuum of ·): min(1, b / a) with convention 1 if a == 0

    This matches the semantics used in the paper excerpt for the weather examples.
    Truth values are wrapped in the Identity monad to match the existing logic API
    (see boolean.py and priest.py).
    """

    # Lattice bounds ---------------------------------------------------------
    def top(self) -> Kind1[_MonadType, torch.Tensor]:
        return self.monad_from_value(torch.tensor(1.0))

    def bottom(self) -> Kind1[_MonadType, torch.Tensor]:
        return self.monad_from_value(torch.tensor(0.0))

    # Connectives ------------------------------------------------------------
    def neg(self, a: Kind1[_MonadType, torch.Tensor]) -> Kind1[_MonadType, torch.Tensor]:
        return a.bind(lambda x: self.monad_from_value(torch.tensor(1.0) - x))

    def conjunction(
        self, a: Kind1[_MonadType, torch.Tensor], b: Kind1[_MonadType, torch.Tensor]
    ) -> Kind1[_MonadType, torch.Tensor]:
        return a.bind(lambda x: b.bind(lambda y: self.monad_from_value(x * y)))

    def disjunction(
        self, a: Kind1[_MonadType, torch.Tensor], b: Kind1[_MonadType, torch.Tensor]
    ) -> Kind1[_MonadType, torch.Tensor]:
        # probabilistic sum
        return a.bind(lambda x: b.bind(lambda y: self.monad_from_value(x + y - x * y)))

    # Override default material implication with product residuum
    def implies(
        self, a: Kind1[_MonadType, torch.Tensor], b: Kind1[_MonadType, torch.Tensor]
    ) -> Kind1[_MonadType, torch.Tensor]:
        return a.bind(lambda x: b.bind(lambda y: self._residuum(x, y)))

    def _residuum(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Kind1[_MonadType, torch.Tensor]:
        if x <= y:
            return self.monad_from_value(torch.tensor(1.0))
        if x == 0.0:
            return self.monad_from_value(torch.tensor(1.0))
        return self.monad_from_value(torch.minimum(torch.tensor(1.0), y / x))

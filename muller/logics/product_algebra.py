from muller.logics.aggr2sgrpblat import Aggr2SGrpBLat, NeSyLogicMeta
from muller.monad.identity import Identity
from typing import Iterable


class ProductAlgebraLogic(Aggr2SGrpBLat[Identity[float]], NeSyLogicMeta[float]):
    """
    Product algebra on [0, 1] with:
    - Conjunction: a · b
    - Disjunction: a + b − a · b (probabilistic sum)
    - Negation: 1 − a
    - Implication (residuum of ·): min(1, b / a) with convention 1 if a == 0

    This matches the semantics used in the paper excerpt for the weather examples.
    Truth values are wrapped in the Identity monad to match the existing logic API
    (see boolean.py and priest.py).
    """

    # Lattice bounds ---------------------------------------------------------
    def top(self) -> Identity[float]:
        return Identity.insert(1.0)

    def bottom(self) -> Identity[float]:
        return Identity.insert(0.0)

    # Connectives ------------------------------------------------------------
    def neg(self, a: Identity[float]) -> Identity[float]:
        return Identity.insert(1.0 - a.value)

    def conjunction(self, a: Identity[float], b: Identity[float]) -> Identity[float]:
        return Identity.insert(a.value * b.value)

    def disjunction(self, a: Identity[float], b: Identity[float]) -> Identity[float]:
        # probabilistic sum
        return Identity.insert(a.value + b.value - a.value * b.value)

    # Override default material implication with product residuum
    def implies(self, a: Identity[float], b: Identity[float]) -> Identity[float]:
        if a.value <= b.value:
            return Identity.insert(1.0)
        if a.value == 0.0:
            return Identity.insert(1.0)
        return Identity.insert(min(1.0, b.value / a.value))

    # High-level aggregations on plain floats --------------------------------
    # These helpers expose the product-algebra ∀/∃ fold directly on lists of
    # probabilities without requiring callers to wrap values in Identity.
    def universal(self, probabilities: Iterable[float]) -> float:
        """Universal aggregation (conjunction/product) over probabilities.

        Computes the product-algebra conjunction over the sequence, equivalent
        to iterative application of `conjunction` starting from `top()`.
        """
        acc: Identity[float] = self.top()
        for p in probabilities:
            acc = self.conjunction(acc, Identity.insert(float(p)))
        return acc.value

    def existential(self, probabilities: Iterable[float]) -> float:
        """Existential aggregation (probabilistic sum) over probabilities.

        Computes the product-algebra disjunction over the sequence, equivalent
        to iterative application of `disjunction` starting from `bottom()`.
        """
        acc: Identity[float] = self.bottom()
        for p in probabilities:
            acc = self.disjunction(acc, Identity.insert(float(p)))
        return acc.value



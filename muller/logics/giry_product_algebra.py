from muller.logics.aggr2sgrpblat import Aggr2SGrpBLat, NeSyLogicMeta
from muller.monad.giry import Giry


class GiryProductAlgebraLogic(Aggr2SGrpBLat[list, Giry[float]], NeSyLogicMeta[float]):
    """
    Product algebra on truth values in [0, 1] carried by the Giry monad.

    - Conjunction: a · b
    - Disjunction: a + b − a·b (probabilistic sum)
    - Negation: 1 − a
    - Implication (residuum of ·): min(1, b / a), with convention 1 when a = 0

    All connectives are lifted pointwise to GiryMonad via bind/map, assuming
    independence for combining two monadic truth values.
    """

    # Lattice bounds ---------------------------------------------------------
    def top(self) -> Giry[float]:
        return Giry.insert(1.0)

    def bottom(self) -> Giry[float]:
        return Giry.insert(0.0)

    # Connectives ------------------------------------------------------------
    def neg(self, a: Giry[float]) -> Giry[float]:
        return a.map(lambda x: 1.0 - x)

    def conjunction(self, a: Giry[float], b: Giry[float]) -> Giry[float]:
        # (lifted) product under independence
        return a.bind(lambda x: b.map(lambda y: x * y))

    def disjunction(self, a: Giry[float], b: Giry[float]) -> Giry[float]:
        # probabilistic sum a + b - a*b (lifted)
        return a.bind(lambda x: b.map(lambda y: x + y - x * y))

    def implies(self, a: Giry[float], b: Giry[float]) -> Giry[float]:
        # residuum of product (lifted)
        def residuum(x: float, y: float) -> float:
            if x == 0.0 or x <= y:
                return 1.0
            return min(1.0, y / x)

        return a.bind(lambda x: b.map(lambda y: residuum(x, y)))

    # High-level aggregations on plain floats --------------------------------
    # Provide convenient ∀/∃ folds directly on probabilities (scalars).
    def universal(self, probabilities):  # Iterable[float] -> float
        prod = 1.0
        for p in probabilities:
            prod *= float(p)
        return prod

    def existential(self, probabilities):  # Iterable[float] -> float
        prod_complements = 1.0
        for p in probabilities:
            prod_complements *= (1.0 - float(p))
        return 1.0 - prod_complements



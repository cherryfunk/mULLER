from functools import reduce
from typing import Callable
from muller import nesy, Dist
from muller.logics import Aggr2MonBLat, TwoMonBLat, with_list_structure, with_prob_structure
from muller.monad.util import bind_T, fmap_T


class MyLogicOverwrite(Aggr2MonBLat[list, Dist[bool]]):
    def top(self) -> Dist[bool]:
        return Dist({True: .9, False: 0.1})

    def bottom(self) -> Dist[bool]:
        return Dist({True: 0.1, False: .9})

    def conjunction(self, a: Dist[bool], b: Dist[bool]) -> Dist[bool]:
        return bind_T(a, a, lambda x: fmap_T(b, b, lambda y: x and y))

    def disjunction(self, a: Dist[bool], b: Dist[bool]) -> Dist[bool]:
        return bind_T(a, a, lambda x: fmap_T(b, b, lambda y: x or y))
    
    def aggrA[A](self, structure: list[A], f: Callable[[A], Dist[bool]]) -> Dist[bool]:
        return reduce(
            lambda a, b: self.conjunction(a, b), map(f, structure), self.top()
        )
        
    def aggrE[A](self, structure: list[A], f: Callable[[A], Dist[bool]]) -> Dist[bool]:
        return reduce(
            lambda a, b: self.disjunction(a, b), map(f, structure), self.bottom()
        )


@with_list_structure(Dist[str], str)
class MyCustomLogic(TwoMonBLat[Dist[str]]):
    def top(self) -> Dist[str]:
        return Dist({ "high": 0.9, "low": 0.1 })

    def bottom(self) -> Dist[str]:
        return Dist({ "high": 0.1, "low": 0.9 })

    def conjunction(self, a: Dist[str], b: Dist[str]) -> Dist[str]:
        return bind_T(a, a, lambda x: fmap_T(b, b, lambda y: {
            ("high", "high"): "high",
            ("high", "low"): "low",
            ("low", "high"): "low",
            ("low", "low"): "low"
        }.get((x,y))))

    def disjunction(self, a: Dist[str], b: Dist[str]) -> Dist[str]:
        return bind_T(a, a, lambda x: fmap_T(b, b, lambda y: {
            ("high", "high"): "high",
            ("high", "low"): "high",
            ("low", "high"): "high",
            ("low", "low"): "low"
        }.get((x,y))))


nesy_framework = nesy(Dist, bool)  # Uses `muller.logics.ProbabilisticBooleanLogic`
print(f"Top: {nesy_framework.logic.top()}") # Prints `Prob({True: 1.0, False: 0.0})`
nesy_framework = nesy(MyLogicOverwrite(), bool)  # Uses `MyLogicOverwrite`
print(f"Top: {nesy_framework.logic.top()}") # Prints `Prob({True: 0.9, False: 0.1})`
nesy_framework = nesy(Dist, str)  # Uses `MyCustomLogic`
print(f"Top: {nesy_framework.logic.top()}") # Prints `Prob({ "high": 0.9, "low": 0.1 })`
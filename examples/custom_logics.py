from muller import nesy, Prob
from muller.logics import Aggr2SGrpBLat
from muller.monad.util import bind_T, fmap_T


class MyLogicOverwrite(Aggr2SGrpBLat[Prob[bool]]):
    def top(self) -> Prob[bool]:
        return Prob({True: .9, False: 0.1})

    def bottom(self) -> Prob[bool]:
        return Prob({True: 0.1, False: .9})

    def conjunction(self, a: Prob[bool], b: Prob[bool]) -> Prob[bool]:
        return bind_T(a, a, lambda x: fmap_T(b, b, lambda y: x and y))

    def disjunction(self, a: Prob[bool], b: Prob[bool]) -> Prob[bool]:
        return bind_T(a, a, lambda x: fmap_T(b, b, lambda y: x or y))


class MyCustomLogic(Aggr2SGrpBLat[Prob[str]]):
    def top(self) -> Prob[str]:
        return Prob({ "high": 0.9, "low": 0.1 })

    def bottom(self) -> Prob[str]:
        return Prob({ "high": 0.1, "low": 0.9 })

    def conjunction(self, a: Prob[str], b: Prob[str]) -> Prob[str]:
        return bind_T(a, a, lambda x: fmap_T(b, b, lambda y: {
            ("high", "high"): "high",
            ("high", "low"): "low",
            ("low", "high"): "low",
            ("low", "low"): "low"
        }.get((x,y))))

    def disjunction(self, a: Prob[str], b: Prob[str]) -> Prob[str]:
        return bind_T(a, a, lambda x: fmap_T(b, b, lambda y: {
            ("high", "high"): "high",
            ("high", "low"): "high",
            ("low", "high"): "high",
            ("low", "low"): "low"
        }.get((x,y))))


nesy_framework = nesy(Prob, bool)  # Uses `muller.logics.ProbabilisticBooleanLogic`
print(f"Top: {nesy_framework.logic.top()}") # Prints `Prob({True: 1.0, False: 0.0})`
nesy_framework = nesy(MyLogicOverwrite(), bool)  # Uses `MyLogicOverwrite`
print(f"Top: {nesy_framework.logic.top()}") # Prints `Prob({True: 0.9, False: 0.1})`
nesy_framework = nesy(Prob, str)  # Uses `MyCustomLogic`
print(f"Top: {nesy_framework.logic.top()}") # Prints `Prob({ "high": 0.9, "low": 0.1 })`
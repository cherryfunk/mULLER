from muller.aggr2sgrpblat.aggr2sgrpblat import Aggr2SGrpBLat
from muller.monad.distribution import Prob
from muller.monad.non_empty_powerset import NonEmptyPowerset, from_list, singleton
from muller.monad.util import bindT, fmapT


class NonEmptyPowersetBoolDoubleSemiGroupBoundedLattice(Aggr2SGrpBLat[NonEmptyPowerset[bool]]):
    def top(self) -> NonEmptyPowerset[bool]:
        return singleton(True)

    def bottom(self) -> NonEmptyPowerset[bool]:
        return singleton(False)

    def neg(self, a: NonEmptyPowerset[bool]) -> NonEmptyPowerset[bool]:
        return from_list([not x for x in a.value])

    def conjunction(self, a: NonEmptyPowerset[bool], b: NonEmptyPowerset[bool]) -> NonEmptyPowerset[bool]:
        return from_list([x and y for x in a.value for y in b.value])

    def disjunction(self, a: NonEmptyPowerset[bool], b: NonEmptyPowerset[bool]) -> NonEmptyPowerset[bool]:
        return from_list([x or y for x in a.value for y in b.value])


class ProbBoolDoubleSemiGroupBoundedLattice(Aggr2SGrpBLat[Prob[bool]]):
    def top(self) -> Prob[bool]:
        return Prob({True: 1.0, False: 0.0})

    def bottom(self) -> Prob[bool]:
        return Prob({False: 1.0, True: 0.0})

    def neg(self, a: Prob[bool]) -> Prob[bool]:
        return fmapT(a, a, lambda x: not x)

    def conjunction(self, a: Prob[bool], b: Prob[bool]) -> Prob[bool]:
        return bindT(a, a, lambda x: fmapT(b, b, lambda y: x and y))

    def disjunction(self, a: Prob[bool], b: Prob[bool]) -> Prob[bool]:
        return bindT(a, a, lambda x: fmapT(b, b, lambda y: x or y))
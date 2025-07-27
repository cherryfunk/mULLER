from pymonad.monad import Monad
import sys


from .aggr2sgrpblat import Aggr2SGrpBLat
from .boolean import (
    NonEmptyPowersetBoolDoubleSemiGroupBoundedLattice,
    ProbBoolDoubleSemiGroupBoundedLattice,
)
from .priest import NonEmptyPowersetPriestDoubleSemiGroupBoundedLattice


def get_lattice[O](monad: type[Monad[O]], type: type[O]) -> Aggr2SGrpBLat[Monad[O]]:
    current_module = sys.modules[__name__]
    impl_name = f"{monad.__name__}{type.__name__[0].upper()}{type.__name__[1:]}DoubleSemiGroupBoundedLattice"
    member = getattr(current_module, impl_name, None)
    if member is not None:
        return member()
    raise ValueError(f"No implementation for monad {monad} and truth basis {type} found.")


__all__= [
    "Aggr2SGrpBLat",
    "get_lattice",
    "NonEmptyPowersetBoolDoubleSemiGroupBoundedLattice",
    "ProbBoolDoubleSemiGroupBoundedLattice",
    "NonEmptyPowersetPriestDoubleSemiGroupBoundedLattice",
]
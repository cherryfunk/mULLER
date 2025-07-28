from functools import cache, lru_cache
from itertools import chain
import sys
from types import get_original_bases
from pymonad.monad import Monad

import muller.logics
from .aggr2sgrpblat import Aggr2SGrpBLat
from .boolean import (
    NonDeterministicBooleanLogic,
    ProbabilisticBooleanLogic,
    ClassicalBooleanLogic,
)
from .priest import (
    NonDeterministicPriestLogic,
    ProbabilisticPriestLogic,
    ClassicalPriestLogic,
)


@lru_cache(maxsize=128)
def get_logic[T: Monad, O](
    monad_type: type[T], omega: type[O]
) -> Aggr2SGrpBLat[Monad[O]]:
    """
    Get the logic for a specific monad and type.
    """
    modules = list(sys.modules.values())
    members = (getattr(module, name) for module in modules for name in dir(module))
    members = chain(
        muller.logics.__dict__.values(), members
    )  # Ensure we start with our own logics for efficiency

    # Search through all attributes in the current scope
    for val in members:
        # Check if it's a class and not a built-in or imported type
        if isinstance(val, type) and hasattr(val, "__module__"):
            # Check if it's a subclass of Aggr2SGrpBLat
            if issubclass(val, Aggr2SGrpBLat):
                # Find the class in the inheritance hierarchy that directly inherits from Aggr2SGrpBLat with generic parameters
                def find_aggr2sgrpblat_base(cls) -> type | None:
                    if any(base is Aggr2SGrpBLat for base in cls.__bases__):
                        return cls

                    return next(
                        (
                            find_aggr2sgrpblat_base(base)
                            for base in cls.__bases__
                            if base is not object
                        ),
                        None,
                    )

                # Find the actual class that defines the generic Aggr2SGrpBLat inheritance
                if (generic_base_class := find_aggr2sgrpblat_base(val)) is not None:
                    val = generic_base_class

                # Check if it's a subclass of Aggr2SGrpBLat
                original_bases = get_original_bases(val) if hasattr(val, "__bases__") else []
                base = next(
                    (
                        base
                        for base in original_bases
                        if hasattr(base, "__origin__")
                        and base.__origin__ is Aggr2SGrpBLat
                        and hasattr(base, "__args__")
                        and len(base.__args__) > 0
                    ),
                    None,
                )

                # Check if the base is Aggr2SGrpBLat with a generic parameter
                if base is not None:

                    monad_arg = base.__args__[0]

                    # Check if the monad type matches our target T
                    if (
                        hasattr(monad_arg, "__origin__")
                        and monad_arg.__origin__ is monad_type
                        and hasattr(monad_arg, "__args__")
                        and len(monad_arg.__args__) > 0
                        and monad_arg.__args__[0] is omega
                    ):

                        # Return an instance of the matching class
                        return val()

    # If no matching class is found, raise an exception
    raise ValueError(
        f"No logic class found for monad type {monad_type} with omega type {omega}"
    )


__all__ = [
    "Aggr2SGrpBLat",
    "get_logic",
    "NonDeterministicBooleanLogic",
    "ProbabilisticBooleanLogic",
    "ClassicalBooleanLogic",
    "NonDeterministicPriestLogic",
    "ProbabilisticPriestLogic",
    "ClassicalPriestLogic",
]

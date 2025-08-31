import sys
from functools import lru_cache
from itertools import chain
from types import get_original_bases
from typing import get_origin

import muller.logics
from muller.monad.base import ParametrizedMonad

from .aggr2sgrpblat import Aggr2SGrpBLat
from .boolean import (
    ClassicalBooleanLogic,
    NonDeterministicBooleanLogic,
    ProbabilisticBooleanLogic,
)
from .priest import (
    ClassicalPriestLogic,
    NonDeterministicPriestLogic,
    Priest,
    ProbabilisticPriestLogic,
)
from .product_algebra import ProductAlgebraLogic
from .giry_product_algebra import GiryProductAlgebraLogic

def _get_module_member(module: type, name: str) -> str | None:
    """
    Get the module name for a given class.
    """
    try:
        return getattr(module, name, None)
    except:
        return None
    
def _type_eq(a, b) -> bool:
    return a is b or a == b or (get_origin(a) or a) == (get_origin(b) or b)

@lru_cache(maxsize=128)
def get_logic[T: ParametrizedMonad, O, S](
    monad_type: type[T], omega: type[O], structure: type[S]
) -> Aggr2SGrpBLat[S, ParametrizedMonad[O]]:
    """
    Get the logic for a specific monad and type.
    """
    modules = list(sys.modules.values())
    members = (_get_module_member(module, name) for module in modules for name in dir(module))
    members = chain(
        muller.logics.__dict__.values(), members
    )  # Ensure we start with our own logics for efficiency

    # Search through all attributes in the current scope
    for val in members:
        # Check if it's a class and not a built-in or imported type
        if isinstance(val, type) and hasattr(val, "__module__"):
            # Check if it's a subclass of Aggr2SGrpBLat
            if issubclass(val, Aggr2SGrpBLat):
                # Find the class in the inheritance hierarchy that directly inherits
                # from Aggr2SGrpBLat with generic parameters
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

                if (annotations := getattr(val, "__annotations__", None)) is not None and type(annotations) is dict:
                    structure_arg = annotations.get("S", None)
                    monad_arg = annotations.get("T", None)
                    omega_arg = annotations.get("O", None)

                    if _type_eq(structure, structure_arg) and _type_eq(monad_arg, monad_type) and _type_eq(omega_arg, omega):
                        # Return an instance of the matching class
                        return val()

                # Check if it's a subclass of Aggr2SGrpBLat
                original_bases = (
                    get_original_bases(val) if hasattr(val, "__bases__") else []
                )
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
                if base is not None and len(base.__args__) == 2:
                    structure_arg, monad_arg = base.__args__

                    # Check if the monad type matches our target T
                    if (
                        hasattr(monad_arg, "__origin__")
                        and monad_arg.__origin__ is monad_type
                        and hasattr(monad_arg, "__args__")
                        and len(monad_arg.__args__) > 0
                        and monad_arg.__args__[0] is omega
                        and _type_eq(structure_arg, structure)
                    ):
                        # Return an instance of the matching class
                        return val()

    # If no matching class is found, raise an exception
    raise ValueError(
        f"No logic class found for monad type {monad_type} with omega type {omega} and structure {structure}."
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
    "Priest",
    "ProductAlgebraLogic",
    "GiryProductAlgebraLogic",
]

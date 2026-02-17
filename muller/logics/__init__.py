import sys
from functools import lru_cache
from itertools import chain
from types import ModuleType, get_original_bases
from typing import Any, TypeVar, get_origin

from returns.interfaces.container import Container1
from returns.primitives.hkt import Kind1

import muller.logics
from muller.monad.giry_sampling import GirySampling

from .aggr2sgrpblat import (
    Aggr2SGrpBLat,
    DblSGrpBLat,
    GirySamplingAggregationMixin,
    ListAggregationMixin,
)
from .boolean import (
    BooleanLogic,
    # ClassicalBooleanLogic,
    # ClassicalBooleanLogicList,
    # ClassicalBooleanLogicProb,
    # NonDeterministicBooleanLogic,
    # NonDeterministicBooleanLogicList,
    # NonDeterministicBooleanLogicProb,
    # ProbabilisticBooleanLogic,
    # ProbabilisticBooleanLogicList,
    # ProbabilisticBooleanLogicProb,
)
from .priest import (
    # ClassicalPriestLogic,
    # ClassicalPriestLogicList,
    # ClassicalPriestLogicProb,
    # NonDeterministicPriestLogic,
    # NonDeterministicPriestLogicList,
    # NonDeterministicPriestLogicProb,
    Priest,
    PriestLogic,
    # ProbabilisticPriestLogic,
    # ProbabilisticPriestLogicList,
    # ProbabilisticPriestLogicProb,
)
from .product_algebra import ProductAlgebraLogic


def _get_module_member(module: ModuleType, name: str) -> Any:
    """
    Get the module name for a given class.
    """
    try:
        return getattr(module, name, None)
    except:  # noqa e722
        return None


def _monad_type_eq(
    requested_monad: type, requested_omega: type, candidate_m: Any
) -> bool:
    """Check if the candidate M type matches the requested monad and omega types.

    M is a fully parameterized type like Prob[bool], Identity[bool], etc.
    requested_monad might be Prob[bool] or just Prob.
    requested_omega is the expected omega type like bool.
    """
    # Get the origin of M (e.g., Prob from Prob[bool])
    m_origin = get_origin(candidate_m)

    # Get the origin of requested_monad if it's parameterized
    req_origin = get_origin(requested_monad) or requested_monad

    # Check if the monad base types match
    if m_origin is None or m_origin is not req_origin:
        return False

    # Check if M has type arguments and the first one matches omega
    if hasattr(candidate_m, "__args__") and len(candidate_m.__args__) > 0:
        m_omega = candidate_m.__args__[0]
        return m_omega is requested_omega or m_omega == requested_omega

    return False


def _structure_matches(cls: type, structure: type) -> bool:
    """Check if the class has the appropriate aggregation mixin for the structure."""
    structure_origin = get_origin(structure) or structure

    # List structure -> must have ListAggregationMixin
    if structure_origin is list:
        return issubclass(cls, ListAggregationMixin)
    # GirySampling structure -> must have GirySamplingAggregationMixin
    if structure_origin is GirySampling:
        return issubclass(cls, GirySamplingAggregationMixin)

    return False


def _get_truth_type_from_class(cls: type) -> type | None:
    if any(c is DblSGrpBLat for c in cls.__bases__):
        try:
            _monad, dblsgrpblat = get_original_bases(cls)
            args = getattr(dblsgrpblat, "__args__", None)

            if args and len(args) == 2 and isinstance(args[1], type):
                return args[1]  # M, O
        except (TypeError, AttributeError):
            pass

    return None


_TruthValueType = TypeVar("_TruthValueType")
_ObjectType = TypeVar("_ObjectType")
_StructureType = TypeVar("_StructureType")
_MonadType = TypeVar("_MonadType", bound=Container1[Any])


def get_logic(
    monad_type: type[_MonadType],
    truth_type: type[_TruthValueType],
    structure_type: type[_StructureType],
) -> Aggr2SGrpBLat[_MonadType, _TruthValueType, _StructureType, _ObjectType]:
    mixin: type[Aggr2SGrpBLat[_MonadType, _TruthValueType, Any, _ObjectType]] | None
    if issubclass(structure_type, list):
        mixin = ListAggregationMixin
    elif structure_type is GirySampling:
        mixin = GirySamplingAggregationMixin
    elif issubclass(structure_type, Aggr2SGrpBLat):
        mixin = structure_type
    else:
        raise ValueError(
            f"Unsupported structure type {structure_type}. Supported types are list and GirySampling."
        )

    modules = list(sys.modules.values())
    all_members = (
        _get_module_member(module, name) for module in modules for name in dir(module)
    )
    members = chain(
        muller.logics.__dict__.values(), all_members
    )  # Ensure we start with our own logics for efficiency

    seen: set[type] = set()

    # Search through all attributes in the current scope
    for val in members:
        # Check if it's a class and not a built-in or imported type
        if not isinstance(val, type) or not hasattr(val, "__module__"):
            continue
        if val in seen:
            continue
        seen.add(val)

        # Skip abstract base classes
        if val in (
            DblSGrpBLat,
            Aggr2SGrpBLat,
        ):
            continue

        # Check if it's a subclass of DblSGrpBLat
        try:
            if not issubclass(val, DblSGrpBLat):
                continue
        except TypeError:
            continue

        # # Check structure compatibility
        # if not _structure_matches(val, object_type):
        #     continue

        # Extract truth type from the class
        candidate_truth_type = _get_truth_type_from_class(val)
        if candidate_truth_type is None or candidate_truth_type is not truth_type:
            continue

        # create instance
        _LogicInstance: type[
            Aggr2SGrpBLat[_MonadType, _TruthValueType, _StructureType, _ObjectType]
        ] = type(f"LogicInstance_{val.__name__}", (mixin, val), {})

        # Found a match - return an instance
        return _LogicInstance(monad_from_value=monad_type.from_value)

    # If no matching class is found, raise an exception
    raise ValueError(f"No logic class found for truth type {truth_type}.")


__all__ = [
    "Aggr2SGrpBLat",
    "get_logic",
    "BooleanLogic",
    # "NonDeterministicBooleanLogic",
    # "ProbabilisticBooleanLogic",
    # "ClassicalBooleanLogic",
    # "ClassicalBooleanLogicList",
    # "ClassicalBooleanLogicProb",
    # "NonDeterministicBooleanLogic",
    # "NonDeterministicBooleanLogicList",
    # "NonDeterministicBooleanLogicProb",
    # "ProbabilisticBooleanLogic",
    # "ProbabilisticBooleanLogicList",
    # "ProbabilisticBooleanLogicProb",
    "Priest",
    "PriestLogic",
    # "ClassicalPriestLogic",
    # "ClassicalPriestLogicList",
    # "ClassicalPriestLogicProb",
    # "NonDeterministicPriestLogic",
    # "NonDeterministicPriestLogicList",
    # "NonDeterministicPriestLogicProb",
    # "ProbabilisticPriestLogic",
    # "ProbabilisticPriestLogicList",
    # "ProbabilisticPriestLogicProb",
    "ProductAlgebraLogic",
    "GiryProductAlgebraLogic",
    "DblSGrpBLat",
    "Aggr2SGrpBLat",
    "with_prob_structure",
    "with_list_structure",
]

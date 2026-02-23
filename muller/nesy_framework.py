from typing import (
    Any,
    Callable,
    Mapping,
    Type,
    TypeVar,
)

from returns.interfaces.container import Container1
from returns.interfaces.mappable import Mappable1

from muller.framework.nesy import BaseNeSyFramework
from muller.logics import Aggr2MonBLat, get_logic
from muller.parser import (
    Ident,
)

_InType = TypeVar("_InType")
_OutType = TypeVar("_OutType")
SingleArgumentTypeFunction = Callable[[list[_InType]], _OutType]
_SingleArgumentTypeFunction = (
    Callable[[], _OutType]
    | Callable[[_InType], _OutType]
    | Callable[[_InType, _InType], _OutType]
    | Callable[[_InType, _InType, _InType], _OutType]
    | Callable[[_InType, _InType, _InType, _InType], _OutType]
    | Callable[[_InType, _InType, _InType, _InType, _InType], _OutType]
)


_MonadType = TypeVar("_MonadType", bound=Container1[Any])
_OtherMonadType = TypeVar("_OtherMonadType", bound=Container1[Any])
_TruthType = TypeVar("_TruthType")
_StructureType = TypeVar("_StructureType", bound=Mappable1[Any])
_OtherStructureType = TypeVar("_OtherStructureType")
_OtherTruthType = TypeVar("_OtherTruthType")
_ObjectType = TypeVar("_ObjectType")
_InterpretationType = TypeVar(
    "_InterpretationType",
    bound="Interpretation[Any, Any, Any, Any]"
    " | EmbeddingInterpretation[Any, Any, Any, Any]",
)
_LogicType = TypeVar("_LogicType", bound=Aggr2MonBLat[Any, Any, Any])


type Valuation[A] = Mapping[Ident, A]


nesy_for_logic = BaseNeSyFramework.from_logic
"""
Create a NeSyFramework instance for the given logic. See `nesy` for more details.

Args:
    logic: An instance of Aggr2SGrpBLat representing the logic.

Returns:
    An instance of NeSyFramework with the logic's monad and omega type.
"""


def nesy(
    monad_type: Type[_MonadType],
    truth_type: Type[_TruthType],
    structure_type: Type[_StructureType],
    mixins: list[type] = [],
) -> BaseNeSyFramework[_MonadType, _TruthType, _StructureType]:
    logic = get_logic(monad_type, truth_type, structure_type, mixins)
    return BaseNeSyFramework.from_logic(logic)


# TODO: SubClass TorchNesyFramework: TorchAggr2MonBLat and two interpretations

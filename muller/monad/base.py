from typing import Any, Callable, TypeVar

from returns.interfaces.container import Container1
from returns.primitives.hkt import Kind1

_ValueType = TypeVar("_ValueType")
_MonadType = TypeVar("_MonadType", bound=Container1[Any])
_NewValueType = TypeVar("_NewValueType")


def monad_apply(
    self: Kind1[_MonadType, _ValueType],
    container: Kind1[_MonadType, Callable[[_ValueType], _NewValueType]],
) -> Kind1[_MonadType, _NewValueType]:
    return container.bind(lambda f: self.map(f))

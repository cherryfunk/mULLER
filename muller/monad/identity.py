from __future__ import annotations

from typing import Callable, TypeVar, final

from returns.interfaces.container import Container1
from returns.primitives.container import BaseContainer
from returns.primitives.hkt import Kind1, SupportsKind1

from muller.monad.base import monad_apply

_ValueType = TypeVar("_ValueType")
_NewValueType = TypeVar("_NewValueType")


@final
class Identity(
    BaseContainer,
    SupportsKind1["Identity", _ValueType],  # type: ignore[type-arg]
    Container1[_ValueType],
):
    def __init__(self, value: _ValueType) -> None:
        super().__init__(value)

    def map(
        self, function: Callable[[_ValueType], _NewValueType]
    ) -> Kind1["Identity", _NewValueType]:  # type: ignore[type-arg]
        return Identity(function(self._inner_value))

    def bind(
        self, function: Callable[[_ValueType], Kind1["Identity", _NewValueType]]  # type: ignore[type-arg]
    ) -> Kind1["Identity", _NewValueType]:  # type: ignore[type-arg]
        return function(self._inner_value)

    @classmethod
    def from_value(cls, inner_value: _NewValueType) -> Kind1["Identity", _NewValueType]:  # type: ignore[type-arg]
        return Identity(inner_value)

    apply = monad_apply

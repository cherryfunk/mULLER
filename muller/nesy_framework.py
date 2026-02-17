from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Mapping, Type, TypeVar, overload

from returns.interfaces.container import Container1
from returns.primitives.hkt import Kind1

from muller.logics import Aggr2SGrpBLat, get_logic
from muller.parser import (
    Computation,
    Conjunction,
    Disjunction,
    ExistentialQuantification,
    FalseFormula,
    Formula,
    FunctionApplication,
    Ident,
    Implication,
    MonadicPredicate,
    Negation,
    Predicate,
    Term,
    TrueFormula,
    UniversalQuantification,
    Variable,
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
_StructureType = TypeVar("_StructureType")
_OtherStructureType = TypeVar("_OtherStructureType")
_OtherTruthType = TypeVar("_OtherTruthType")
_ObjectType = TypeVar("_ObjectType")


type Valuation[A] = Mapping[Ident, A]


class Interpretation(Generic[_MonadType, _TruthType, _StructureType, _ObjectType], ABC):
    @abstractmethod
    def __init__(
        self,
        sort: Kind1[_StructureType, _ObjectType],
        functions: dict[Ident, SingleArgumentTypeFunction[_ObjectType, _ObjectType]] = {},
        mfunctions: dict[
            Ident, SingleArgumentTypeFunction[_ObjectType, Kind1[_MonadType, _ObjectType]]
        ] = {},
        predicates: dict[Ident, SingleArgumentTypeFunction[_ObjectType, _TruthType]] = {},
        mpredicates: dict[
            Ident, SingleArgumentTypeFunction[_ObjectType, Kind1[_MonadType, _TruthType]]
        ] = {},
    ) -> None: ...

    sort: Kind1[_StructureType, _ObjectType]
    functions: dict[Ident, SingleArgumentTypeFunction[_ObjectType, _ObjectType]] = {}
    mfunctions: dict[
        Ident, SingleArgumentTypeFunction[_ObjectType, Kind1[_MonadType, _ObjectType]]
    ] = {}
    predicates: dict[Ident, SingleArgumentTypeFunction[_ObjectType, _TruthType]] = {}
    mpredicates: dict[
        Ident, SingleArgumentTypeFunction[_ObjectType, Kind1[_MonadType, _TruthType]]
    ] = {}

    def fn(
        self,
    ) -> Callable[
        [_SingleArgumentTypeFunction[_ObjectType, _ObjectType]],
        _SingleArgumentTypeFunction[_ObjectType, _ObjectType],
    ]:
        def wrapper(
            f: _SingleArgumentTypeFunction[_ObjectType, _ObjectType],
        ) -> _SingleArgumentTypeFunction[_ObjectType, _ObjectType]:
            self.functions[f.__name__] = lambda args: f(*args)
            return f

        return wrapper

    def comp_fn(
        self,
    ) -> Callable[
        [_SingleArgumentTypeFunction[Kind1[_MonadType, _ObjectType], _ObjectType]],
        _SingleArgumentTypeFunction[Kind1[_MonadType, _ObjectType], _ObjectType],
    ]:
        def wrapper(
            f: _SingleArgumentTypeFunction[Kind1[_MonadType, _ObjectType], _ObjectType],
        ) -> _SingleArgumentTypeFunction[Kind1[_MonadType, _ObjectType], _ObjectType]:
            self.mfunctions[f.__name__] = lambda args: f(*args)
            return f

        return wrapper

    def pred(
        self,
    ) -> Callable[
        [_SingleArgumentTypeFunction[_TruthType, _ObjectType]],
        _SingleArgumentTypeFunction[_TruthType, _ObjectType],
    ]:
        def wrapper(
            f: _SingleArgumentTypeFunction[_TruthType, _ObjectType],
        ) -> _SingleArgumentTypeFunction[_TruthType, _ObjectType]:
            self.predicates[f.__name__] = lambda args: f(*args)
            return f

        return wrapper

    def comp_pred(
        self,
    ) -> Callable[
        [_SingleArgumentTypeFunction[Kind1[_MonadType, _TruthType], _ObjectType]],
        _SingleArgumentTypeFunction[Kind1[_MonadType, _TruthType], _ObjectType],
    ]:
        def wrapper(
            f: _SingleArgumentTypeFunction[Kind1[_MonadType, _TruthType], _ObjectType],
        ) -> _SingleArgumentTypeFunction[Kind1[_MonadType, _TruthType], _ObjectType]:
            self.mpredicates[f.__name__] = lambda args: f(*args)
            return f

        return wrapper


class _Interpretation(
    Generic[_MonadType, _TruthType, _StructureType, _ObjectType],
    Interpretation[_MonadType, _TruthType, _StructureType, _ObjectType],
):
    def __init__(
        self,
        sort: Kind1[_StructureType, _ObjectType],
        functions: dict[Ident, SingleArgumentTypeFunction[_ObjectType, _ObjectType]] = {},
        mfunctions: dict[
            Ident, SingleArgumentTypeFunction[_ObjectType, Kind1[_MonadType, _ObjectType]]
        ] = {},
        predicates: dict[Ident, SingleArgumentTypeFunction[_ObjectType, _TruthType]] = {},
        mpredicates: dict[
            Ident, SingleArgumentTypeFunction[_ObjectType, Kind1[_MonadType, _TruthType]]
        ] = {},
    ) -> None:
        self.sort = sort
        self.functions = functions
        self.mfunctions = mfunctions
        self.predicates = predicates
        self.mpredicates = mpredicates


class NeSyFramework(Generic[_MonadType, _TruthType, _StructureType, _ObjectType]):
    """
    Class to represent a monadic NeSy framework consisting of a monad (T),
    a set Ω acting as truth basis (O),
    and an aggregated double semigroup bounded lettice (R).

    This class ensures the following runtime constraint which is not
    representable in Pythons type system:
    - _R: Aggr2SGrpBLat[T, O] where T is the monad type parameterized by O

    """

    _monad_from_value: Callable[[_TruthType], Kind1[_MonadType, _TruthType]]
    _logic: Aggr2SGrpBLat[_MonadType, _TruthType, _StructureType, _ObjectType]

    @property
    def logic(self) -> Aggr2SGrpBLat[_MonadType, _TruthType, _StructureType, _ObjectType]:
        """
        Returns the logic used in this NeSy framework typed with the
        generic monad but specific truth basis.
        """

        return self._logic

    def create_interpretation(
        self,
        sort: Kind1[_StructureType, _ObjectType],
        functions: dict[Ident, SingleArgumentTypeFunction[_ObjectType, _ObjectType]] = {},
        mfunctions: dict[
            Ident, SingleArgumentTypeFunction[_ObjectType, Kind1[_MonadType, _ObjectType]]
        ] = {},
        predicates: dict[Ident, SingleArgumentTypeFunction[_ObjectType, _TruthType]] = {},
        mpredicates: dict[
            Ident, SingleArgumentTypeFunction[_ObjectType, Kind1[_MonadType, _TruthType]]
        ] = {},
    ) -> Interpretation[_MonadType, _TruthType, _StructureType, _ObjectType]:
        return _Interpretation(sort, functions, mfunctions, predicates, mpredicates)

    def __init__(
        self,
        monad_from_value: Callable[[_TruthType], Kind1[_MonadType, _TruthType]],
        logic: Aggr2SGrpBLat[_MonadType, _TruthType, _StructureType, _ObjectType],
    ) -> None:
        """
        Initialize the NeSy framework with a monad, truth basis, and logic.
        """
        self._logic = logic
        self._monad_from_value = monad_from_value

    def monad_from_value(self, value: _TruthType) -> Kind1[_MonadType, _TruthType]:
        """
        Convert a truth value to the monad type using the logic's from_value method.
        """
        return self._monad_from_value(value)

    def eval_formula(
        self,
        formula: Formula,
        interpretation: Interpretation[_MonadType, _TruthType, _StructureType, _ObjectType],
        valuation: Valuation[_ObjectType] = {},
    ) -> Kind1[_MonadType, _TruthType]:
        """
        Evaluate a formula in the NeSy framework using the provided interpretation and valuation.
        """
        return self._eval(formula, interpretation, valuation)

    @overload
    def _eval(
        self,
        val: Formula,
        interpretation: Interpretation[_MonadType, _TruthType, _StructureType, _ObjectType],
        valuation: Valuation[_ObjectType] = {},
    ) -> Kind1[_MonadType, _TruthType]: ...

    @overload
    def _eval(
        self,
        val: Term,
        interpretation: Interpretation[_MonadType, _TruthType, _StructureType, _ObjectType],
        valuation: Valuation[_ObjectType] = {},
    ) -> _ObjectType: ...

    def _eval(
        self,
        val: Formula | Term,
        interpretation: Interpretation[_MonadType, _TruthType, _StructureType, _ObjectType],
        valuation: Valuation[_ObjectType] = {},
    ) -> Kind1[_MonadType, _TruthType] | _ObjectType:
        match val:
            case Variable(ident):
                return valuation[ident]
            case FunctionApplication(function, arguments):
                func = interpretation.functions[function]
                args = [self._eval(arg, interpretation, valuation) for arg in arguments]
                return func(args)
            case TrueFormula():
                return self.logic.top()
            case FalseFormula():
                return self.logic.bottom()
            case Predicate(name, arguments):
                pred_fn = interpretation.predicates[name]
                args = [self._eval(arg, interpretation, valuation) for arg in arguments]
                return self.monad_from_value(pred_fn(args))
            case MonadicPredicate(name, arguments):
                mpred_fn = interpretation.mpredicates[name]
                args = [self._eval(arg, interpretation, valuation) for arg in arguments]
                return mpred_fn(args)
            case Negation(formula):
                return self.logic.neg(self._eval(formula, interpretation, valuation))
            case Conjunction(left, right):
                return self.logic.conjunction(
                    self._eval(left, interpretation, valuation),
                    self._eval(right, interpretation, valuation),
                )
            case Disjunction(left, right):
                return self.logic.disjunction(
                    self._eval(left, interpretation, valuation),
                    self._eval(right, interpretation, valuation),
                )
            case Implication(antecedent, consequent):
                return self.logic.implies(
                    self._eval(antecedent, interpretation, valuation),
                    self._eval(consequent, interpretation, valuation),
                )
            case UniversalQuantification(variable, formula):
                return self.logic.aggrA(
                    interpretation.sort,
                    lambda x: self._eval(
                        formula, interpretation, {**valuation, variable: x}
                    ),
                )
            case ExistentialQuantification(variable, formula):
                return self.logic.aggrE(
                    interpretation.sort,
                    lambda x: self._eval(
                        formula, interpretation, {**valuation, variable: x}
                    ),
                )
            case Computation(variable, function, arguments, formula):
                mfunc = interpretation.mfunctions[function]
                args = [self._eval(arg, interpretation, valuation) for arg in arguments]
                return mfunc(args).bind(
                    lambda result: self._eval(
                        formula, interpretation, {**valuation, variable: result}
                    )
                )

        raise NotImplementedError(f"Evaluation of {val} is not implemented yet.")

    @classmethod
    def from_logic(
        cls,
        logic: Aggr2SGrpBLat[_MonadType, _TruthType, _StructureType, _ObjectType],
    ) -> "NeSyFramework[_MonadType, _TruthType, _StructureType, _ObjectType]":
        """
        Create a NeSyFramework instance from a given logic.

        Args:
            logic: An instance of Aggr2SGrpBLat representing the logic.

        Returns:
            An instance of NeSyFramework with the logic's monad and truth value type.
        """
        return NeSyFramework(logic.top().from_value, logic)

    # @classmethod
    # @overload
    # def from_monad_and_logic(
    #     cls, monad_type: Type[Kind1[_MonadType, bool]], *, structure: Type[_StructureType]
    # ) -> "NeSyFramework[_MonadType, bool, _StructureType]":
    #     """
    #     Create a NeSyFramework instance with the given monad type and optional
    #     truth value type. See `nesy` for more details.

    #     Args:
    #         monad_type: The type of the monad to use.
    #         structure: The type of aggregation structure.

    #     Returns:
    #         An instance of NeSyFramework with the specified monad and truth value types.
    #     """

    #     logic: Aggr2SGrpBLat[_MonadType, bool, _StructureType] = get_logic(
    #         monad_type, bool, structure
    #     )

    #     return NeSyFramework(logic.top().from_value, logic)

    # @classmethod
    # @overload
    # def from_monad_and_logic(
    #     cls,
    #     monad_type: Type[Kind1[_MonadType, _TruthType]],
    #     *,
    #     omega: Type[_TruthType],
    # ) -> "NeSyFramework[_MonadType, _TruthType, List[Any]]":
    #     """
    #     Create a NeSyFramework instance with the given monad type and optional
    #     truth value type. See `nesy` for more details.

    #     Args:
    #         monad_type: The type of the monad to use.
    #         omega: The type of truth values.

    #     Returns:
    #         An instance of NeSyFramework with the specified monad and truth value types.
    #     """

    #     logic: Aggr2SGrpBLat[_MonadType, _TruthType, List[Any]] = get_logic(
    #         monad_type, omega, List[Any]
    #     )
    #     return NeSyFramework(logic.top().from_value, logic)

    # @classmethod
    # def from_monad_and_logic(
    #     cls,
    #     monad_type: Type[Kind1[_MonadType, _TruthType]] | Type[Kind1[_MonadType, bool]],
    #     *,
    #     omega: Type[_TruthType] | Type[bool] = bool,
    #     structure: Type[_StructureType] | Type[List[Any]] = List[Any],
    # ) -> (
    #     "NeSyFramework[_MonadType, _TruthType, _StructureType]"
    #     | "NeSyFramework[_MonadType, bool, _StructureType]"
    #     | "NeSyFramework[_MonadType, _TruthType, List[Any]]"
    # ):
    #     """
    #     Create a NeSyFramework instance with the given monad type and optional
    #     truth value type. See `nesy` for more details.

    #     Args:
    #         monad_type: The type of the monad to use.
    #         omega: The type of truth values.
    #         structure: The type of aggregation structure.

    #     Returns:
    #         An instance of NeSyFramework with the specified monad and truth value types.
    #     """

    #     # Ignore type in the following. The overloads are correct
    #     logic: Aggr2SGrpBLat[_MonadType, _TruthType, _StructureType] = get_logic(
    #         monad_type,  # type: ignore
    #         omega,  # type: ignore
    #         structure,  # type: ignore
    #     )
    #     return NeSyFramework(logic.top().from_value, logic)


nesy_for_logic = NeSyFramework.from_logic
"""
Create a NeSyFramework instance for the given logic. See `nesy` for more details.

Args:
    logic: An instance of Aggr2SGrpBLat representing the logic.

Returns:
    An instance of NeSyFramework with the logic's monad and omega type.
"""


# nesy_framework_for_monad = NeSyFramework.from_monad_and_logic


# @overload
# def nesy(
#     logic: Aggr2SGrpBLat[_MonadType, _TruthType, _StructureType],
# ) -> NeSyFramework[_MonadType, _TruthType, _StructureType]:
#     """
#     Create a NeSyFramework instance for the given logic.

#     Args:
#         logic: An instance of Aggr2SGrpBLat representing the logic.

#     Returns:
#         An instance of NeSyFramework with the logic's monad and truth value type.
#     """
#     ...


# @overload
# def nesy(
#     monad_type: Type[Kind1[_MonadType, _TruthType]],
#     *,
#     truths: Type[_TruthType],
#     structure: Type[_StructureType],
# ) -> NeSyFramework[_MonadType, _TruthType, _StructureType]:
#     """
#     Creates a NeSy framework instance from a monad type.

#     The function will search all loaded modules for a subclass of `Aggr2SGrpBLat` that
#     matches the provided monad and truth value type and returns a corresponding
#     `NeSyFramework`. To extend the built-in logics, you can create a new logic class
#     that inherits from `Aggr2SGrpBLat` and implements the required methods. The search
#     stops at the first matching logic class found and starts with the built-in logics.
#     To overwrite a builtin implementation with a custom implementation, it has to be
#     instantiated (second overload of `nesy` function).

#     Args:
#         monad_type: monad type.
#         truths: The type of truth values. Defaults to `bool`.
#         structure: The type of structure. Defaults to `list`.

#     Returns:
#         An instance of `NeSyFramework` with the specified monad and truth value types.

#     Example:
#         ::

#             class MyLogicOverwrite(Aggr2SGrpBLat[Prob[bool]]):
#                 ...

#             class MyCustomLogic(Aggr2SGrpBLat[Prob[str]]):
#                 ...


#             nesy_framework = nesy(Prob, truths=bool, structure=list)  # Uses `ProbabilisticBooleanLogic`
#             nesy_framework = nesy(MyLogicOverwrite(), truths=bool, structure=list) # Uses `MyLogicOverwrite`
#             nesy_framework = nesy(Prob, truths=Priest, structure=list)  # Uses `ProbabilisticPriestLogic`
#     """
#     ...

# @overload
# def nesy(
#     monad_type: Type[Kind1[_MonadType, _TruthType]],
#     *,
#     structure: Type[_StructureType],
# ) -> NeSyFramework[_MonadType, bool, _StructureType]: ...


# @overload
# def nesy(
#     monad_type: Type[Kind1[_MonadType, _TruthType]],
#     *,
#     truths: Type[_TruthType],
# ) -> NeSyFramework[_MonadType, _TruthType, List[Any]]: ...


# @overload
# def nesy(
#     monad_type: Type[Kind1[_MonadType, _TruthType]],
# ) -> NeSyFramework[_MonadType, bool, List[Any]]: ...


# def nesy(arg1: Any, *, truths: Any, structure: Any) -> NeSyFramework[Any, Any, Any]:
#     """
#     Create a NeSyFramework instance based on the provided argument.

#     Args:
#         arg1: Either an instance of Aggr2SGrpBLat or a monad type.
#         truths: The type of truth values. Defaults to `bool`.
#         structure: The type of structure. Defaults to `list`.

#     Returns:
#         An instance of NeSyFramework.
#     """
#     if isinstance(arg1, Aggr2SGrpBLat):
#         return nesy_for_logic(arg1)
#     else:
#         return nesy_framework_for_monad(arg1, truths, structure)


def nesy(
    monad_type: Type[_MonadType],
    truth_type: Type[_TruthType],
    structure_type: Type[_StructureType],
    object_type: Type[_ObjectType] | None = None,
) -> NeSyFramework[_MonadType, _TruthType, _StructureType, _ObjectType]:
    logic: Aggr2SGrpBLat[_MonadType, _TruthType, _StructureType, _ObjectType] = get_logic(
        monad_type, truth_type, structure_type
    )
    return NeSyFramework.from_logic(logic)


# def nesy(
#     logic: Aggr2SGrpBLat[_MonadType, _TruthType, _ObjectType],
# ) -> NeSyFramework[_MonadType, _TruthType, _ObjectType]:
#     return nesy_for_logic(logic)

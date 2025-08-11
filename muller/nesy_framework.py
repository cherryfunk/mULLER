import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Type, cast, overload

from muller.logics import Aggr2SGrpBLat, get_logic
from muller.monad.base import ParametrizedMonad
from muller.monad.util import bind

Ident = str


def _format_function_ident(ident: str) -> str:
    """
    Format a function identifier with quotes if needed.
    Functions can start with numbers or letters.
    """
    # Check if it's a simple lowercase identifier
    if re.match(r"^[a-z][a-zA-Z0-9_]*$", ident):
        return ident
    # Check if it's a numeric identifier - functions can be numeric
    if re.match(r"^[0-9]+$", ident):
        return ident
    # Otherwise, it needs quotes
    return f"'{ident}'"


def _format_predicate_ident(ident: str) -> str:
    """
    Format a predicate identifier with quotes if needed.
    Predicates must start with letters, NOT numbers.
    """
    # Check if it's a simple lowercase identifier (predicates must start with letter)
    if re.match(r"^[a-z][a-zA-Z0-9_]*$", ident):
        return ident
    # Predicates cannot be purely numeric - they need quotes
    # Otherwise, it needs quotes
    return f"'{ident}'"


type SingleArgumentTypeFunction[I, O] = (
    Callable[[], O]
    | Callable[[I], O]
    | Callable[[I, I], O]
    | Callable[[I, I, I], O]
    | Callable[[I, I, I, I], O]
)


class NeSyTransformer[A, O1, O2](ABC):
    """
    Base class for transformations in the NeSy framework.
    Transformations can be applied to interpretations to produce new interpretations.
    """

    @abstractmethod
    def __call__(
        self, interpretation: "Interpretation[A, O1]"
    ) -> "Interpretation[A, O2]":
        """
        Apply the transformation to the given interpretation.

        Args:
            interpretation: The interpretation to transform.

        Returns:
            A new interpretation after applying the transformation.
        """
        raise NotImplementedError()


NeSyTransformation = NeSyTransformer


@dataclass
class Interpretation[A, O]:
    universe: list[A]
    functions: dict[Ident, SingleArgumentTypeFunction[A, A]]
    mfunctions: dict[Ident, SingleArgumentTypeFunction[A, ParametrizedMonad[A]]]
    preds: dict[Ident, SingleArgumentTypeFunction[A, O]]
    mpreds: dict[Ident, SingleArgumentTypeFunction[A, ParametrizedMonad[O]]]

    def transform[P](
        self, transformation: NeSyTransformation[A, O, P]
    ) -> "Interpretation[A, P]":
        return transformation(self)


type Valuation[A] = dict[Ident, A]


class NeSyFramework[_T: ParametrizedMonad, _O, _R: Aggr2SGrpBLat]:
    """
    Class to represent a monadic NeSy framework consisting of a monad (T), a set Ω acting as truth basis (O), and an aggregated double semigroup bounded lettice (R).

    This class ensures the following runtime constraint which is not representable in Pythons type system:
    - _R: Aggr2SGrpBLat[_T[_O]]


    """

    _monad: Type[_T]
    _logic: _R

    @property
    def M(self) -> Type[ParametrizedMonad[_O]]:
        """
        Returns the monad type used in this NeSy framework.
        """
        return self._monad

    @property
    def logic(self) -> Aggr2SGrpBLat[ParametrizedMonad[_O]]:
        """
        Returns the logic used in this NeSy framework typed with the generic monad but specific truth basis.
        """

        return self._logic

    @property
    def logic_T(self) -> Aggr2SGrpBLat[_T]:
        """
        Returns the logic used in this NeSy framework typed with the specific monad but generic truth basis.
        """

        return self._logic

    def __init__(
        self,
        monad: type[_T],
        logic: Aggr2SGrpBLat[ParametrizedMonad[_O]],
    ) -> None:
        """
        Initialize the NeSy framework with a monad, truth basis, and logic.
        """
        self._monad = monad

        if monad != type(logic.top()):
            raise ValueError(
                f"Monad type {monad} must match logic type {type(logic.top())}"
            )

        self._logic = cast(_R, logic)

    def unitT(self, value: _O) -> _T:
        """Returns the value 'value' in the monadic context of 'T' typed with the specific monad but generic truth basis."""
        return cast(_T, self.M.insert(value))

    @overload
    def castT(self, m: _T) -> ParametrizedMonad[_O]:
        """Cast a monadic value to the generic monad type with specific truth basis."""
        return cast(_T, m)

    @overload
    def castT(self, m: ParametrizedMonad[_O]) -> _T:
        """Cast a monadic value to the specific monad type with generic truth basis."""
        return cast(_T, m)

    def castT(self, m) -> ParametrizedMonad[_O] | _T:
        """Cast a monadic value to the appropriate type based on the monad."""
        return m


class Term(ABC):
    """
    Base class for all terms in the logic system.
    Terms can be variables or applications of functions to arguments.
    """

    def eval[T: ParametrizedMonad, O, R: Aggr2SGrpBLat, A](
        self,
        nesy: "NeSyFramework[T, O, R]",
        interpretation: "Interpretation[A, O]",
        valuation: "Valuation[A]",
    ) -> A:
        raise NotImplementedError()


@dataclass
class Variable(Term):
    ident: Ident

    def eval[T: ParametrizedMonad, O, R: Aggr2SGrpBLat, A](
        self,
        nesy: "NeSyFramework[T, O, R]",
        interpretation: "Interpretation[A, O]",
        valuation: "Valuation[A]",
    ) -> A:
        return valuation[self.ident]

    def __repr__(self) -> str:
        return self.ident


@dataclass
class FunctionApplication(Term):
    function: Ident
    arguments: list[Term]

    def eval[T: ParametrizedMonad, O, R: Aggr2SGrpBLat, A](
        self,
        nesy: "NeSyFramework[T, O, R]",
        interpretation: "Interpretation[A, O]",
        valuation: "Valuation[A]",
    ) -> A:
        f = interpretation.functions[self.function]
        args = [a.eval(nesy, interpretation, valuation) for a in self.arguments]
        return f(*args)

    def __repr__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.arguments)
        formatted_function = _format_function_ident(self.function)
        if not args_str:
            return formatted_function
        return f"{formatted_function}({args_str})"


class Formula(ABC):
    """
    Base class for all formulas in the logic system.
    """

    @abstractmethod
    def eval[T: ParametrizedMonad, O, R: Aggr2SGrpBLat, A](
        self,
        nesy: "NeSyFramework[T, O, R]",
        interpretation: "Interpretation[A, O]",
        valuation: "Valuation[A]",
    ) -> T:
        raise NotImplementedError()


@dataclass
class TrueFormula(Formula):
    """
    Represents the logical constant True.
    """

    def __repr__(self):
        return "True"

    def eval[T: ParametrizedMonad, O, R: Aggr2SGrpBLat, A](
        self,
        nesy: "NeSyFramework[T, O, R]",
        interpretation: "Interpretation[A, O]",
        valuation: "Valuation[A]",
    ) -> T:
        return nesy.logic_T.top()


@dataclass
class FalseFormula(Formula):
    """
    Represents the logical constant False.
    """

    def __repr__(self):
        return "False"

    def eval[T: ParametrizedMonad, O, R: Aggr2SGrpBLat, A](
        self,
        nesy: "NeSyFramework[T, O, R]",
        interpretation: "Interpretation[A, O]",
        valuation: "Valuation[A]",
    ) -> T:
        return nesy.logic_T.bottom()


@dataclass
class Predicate(Formula):
    """
    Represents a predicate with a name and arguments.
    """

    name: Ident
    arguments: list[Term]

    def __repr__(self):
        # Special case for equality operator: use infix notation
        if self.name == "==" and len(self.arguments) == 2:
            left, right = self.arguments
            # Add parentheses for clarity, similar to other binary operators
            return f"({left} == {right})"

        # Standard predicate representation
        args_str = ", ".join(str(arg) for arg in self.arguments)
        formatted_name = _format_predicate_ident(self.name)
        if not args_str:
            return formatted_name
        return f"{formatted_name}({args_str})"

    def eval[T: ParametrizedMonad, O, R: Aggr2SGrpBLat, A](
        self,
        nesy: "NeSyFramework[T, O, R]",
        interpretation: "Interpretation[A, O]",
        valuation: "Valuation[A]",
    ) -> T:
        p = interpretation.preds[self.name]
        args = [a.eval(nesy, interpretation, valuation) for a in self.arguments]
        return nesy.unitT(p(*args))


@dataclass
class MonadicPredicate(Formula):
    """
    Represents a monadic predicate (a predicate with a single argument).
    """

    name: Ident
    arguments: list[Term]

    def __repr__(self):
        args_str = ", ".join(str(arg) for arg in self.arguments)
        formatted_name = _format_predicate_ident(self.name)
        if not args_str:
            return formatted_name
        return f"{formatted_name}({args_str})"

    def eval[T: ParametrizedMonad, O, R: Aggr2SGrpBLat, A](
        self,
        nesy: "NeSyFramework[T, O, R]",
        interpretation: "Interpretation[A, O]",
        valuation: "Valuation[A]",
    ) -> T:
        p = interpretation.mpreds[self.name]
        args = [a.eval(nesy, interpretation, valuation) for a in self.arguments]
        return nesy.castT(p(*args))


@dataclass
class Negation(Formula):
    """
    Represents the negation of a formula.
    """

    formula: Formula

    def __repr__(self):
        return f"¬{self.formula}"

    def eval[T: ParametrizedMonad, O, R: Aggr2SGrpBLat, A](
        self,
        nesy: "NeSyFramework[T, O, R]",
        interpretation: "Interpretation[A, O]",
        valuation: "Valuation[A]",
    ) -> T:
        return nesy.logic_T.neg(self.formula.eval(nesy, interpretation, valuation))


@dataclass
class Conjunction(Formula):
    """
    Represents the conjunction (AND) of two formulas.
    """

    left: Formula
    right: Formula

    def __repr__(self):
        return f"({self.left} ∧ {self.right})"

    def eval[T: ParametrizedMonad, O, R: Aggr2SGrpBLat, A](
        self,
        nesy: "NeSyFramework[T, O, R]",
        interpretation: "Interpretation[A, O]",
        valuation: "Valuation[A]",
    ) -> T:
        result_left = self.left.eval(nesy, interpretation, valuation)
        result_right = self.right.eval(nesy, interpretation, valuation)
        return nesy.logic_T.conjunction(result_left, result_right)


@dataclass
class Disjunction(Formula):
    """
    Represents the disjunction (OR) of two formulas.
    """

    left: Formula
    right: Formula

    def __repr__(self):
        return f"({self.left} ∨ {self.right})"

    def eval[T: ParametrizedMonad, O, R: Aggr2SGrpBLat, A](
        self,
        nesy: "NeSyFramework[T, O, R]",
        interpretation: "Interpretation[A, O]",
        valuation: "Valuation[A]",
    ) -> T:
        result_left = self.left.eval(nesy, interpretation, valuation)
        result_right = self.right.eval(nesy, interpretation, valuation)
        return nesy.logic_T.disjunction(result_left, result_right)


@dataclass
class Implication(Formula):
    """
    Represents the implication (IF...THEN) of two formulas.
    """

    antecedent: Formula
    consequent: Formula

    def __repr__(self):
        return f"({self.antecedent} -> {self.consequent})"

    def eval[T: ParametrizedMonad, O, R: Aggr2SGrpBLat, A](
        self,
        nesy: "NeSyFramework[T, O, R]",
        interpretation: "Interpretation[A, O]",
        valuation: "Valuation[A]",
    ) -> T:
        result_left = self.antecedent.eval(nesy, interpretation, valuation)
        result_right = self.consequent.eval(nesy, interpretation, valuation)
        return nesy.logic_T.implies(result_left, result_right)


@dataclass
class UniversalQuantification(Formula):
    """
    Represents a universally quantified formula.
    """

    variable: Ident
    formula: Formula

    def __repr__(self):
        return f"∀{self.variable} ({self.formula})"

    def eval[T: ParametrizedMonad, O, R: Aggr2SGrpBLat, A](
        self,
        nesy: "NeSyFramework[T, O, R]",
        interpretation: "Interpretation[A, O]",
        valuation: "Valuation[A]",
    ) -> T:
        universe = interpretation.universe
        results = (
            self.formula.eval(nesy, interpretation, {**valuation, self.variable: a})
            for a in universe
        )
        return nesy.logic_T.aggrA(results)


@dataclass
class ExistentialQuantification(Formula):
    """
    Represents an existentially quantified formula.
    """

    variable: Ident
    formula: Formula

    def __repr__(self):
        return f"∃{self.variable} ({self.formula})"

    def eval[T: ParametrizedMonad, O, R: Aggr2SGrpBLat, A](
        self,
        nesy: "NeSyFramework[T, O, R]",
        interpretation: "Interpretation[A, O]",
        valuation: "Valuation[A]",
    ) -> T:
        universe = interpretation.universe
        results = (
            self.formula.eval(nesy, interpretation, {**valuation, self.variable: a})
            for a in universe
        )
        return nesy.logic_T.aggrE(results)


@dataclass
class Computation(Formula):
    variable: Ident
    function: Ident
    arguments: list[Term]
    formula: Formula

    def __repr__(self):
        args_str = ", ".join(str(arg) for arg in self.arguments)
        formatted_function = _format_function_ident(self.function)
        return f"{self.variable} := {formatted_function}({args_str})({self.formula})"

    def eval[T: ParametrizedMonad, O, R: Aggr2SGrpBLat, A](
        self,
        nesy: "NeSyFramework[T, O, R]",
        interpretation: "Interpretation[A, O]",
        valuation: "Valuation[A]",
    ) -> T:
        f = interpretation.mfunctions[self.function]
        args = [a.eval(nesy, interpretation, valuation) for a in self.arguments]
        result = f(*args)
        x = bind(
            result,
            lambda res: nesy.castT(
                self.formula.eval(
                    nesy, interpretation, {**valuation, self.variable: res}
                )
            ),
        )
        return nesy.castT(x)


def nesy_for_logic[T: ParametrizedMonad, O](
    logic: Aggr2SGrpBLat[T],
) -> NeSyFramework[ParametrizedMonad[O], O, Aggr2SGrpBLat[ParametrizedMonad[O]]]:
    """
    Create a NeSyFramework instance for the given logic. See `nesy` for more details.

    Args:
        logic: An instance of Aggr2SGrpBLat representing the logic.

    Returns:
        An instance of NeSyFramework with the logic's monad and omega type.
    """
    monad_type = type(logic.top())

    # Check if the logic is compatible with the monad type is done in `NeSyFramework.__init__`
    return NeSyFramework(monad_type, cast(Aggr2SGrpBLat[ParametrizedMonad[O]], logic))


def nesy_framework_for_monad[O](
    monad_type: Type[ParametrizedMonad], omega: Type[O] = Type[bool]
) -> NeSyFramework[ParametrizedMonad, Any, Aggr2SGrpBLat[ParametrizedMonad]]:
    """
    Create a NeSyFramework instance with the given monad type and optional truth value type. See `nesy` for more details.

    Args:
        monad_type: The type of the monad to use.
        omega: The type of truth values. Defaults to `bool`.

    Returns:
        An instance of NeSyFramework with the specified monad and truth value types.
    """

    return NeSyFramework(monad_type, get_logic(monad_type, omega))


@overload
def nesy[T: ParametrizedMonad, O](
    logic: Aggr2SGrpBLat[T],
    omega: Type[O] = Type[bool]
) -> NeSyFramework[ParametrizedMonad[O], O, Aggr2SGrpBLat[ParametrizedMonad[O]]]:
    """
    Create a NeSyFramework instance for the given logic.

    Args:
        logic: An instance of Aggr2SGrpBLat representing the logic.

    Returns:
        An instance of NeSyFramework with the logic's monad and truth value type.
    """
    ...


@overload
def nesy[O](
    monad_type: Type[ParametrizedMonad], omega: Type[O] = Type[bool]
) -> NeSyFramework[ParametrizedMonad[O], O, Aggr2SGrpBLat[ParametrizedMonad[O]]]:
    """
    Creates a NeSy framework instance from a monad type.

    The function will search all loaded modules for a subclass of `Aggr2SGrpBLat` that matches the provided monad and truth value type and returns a corresponding `NeSyFramework`. To extend the built-in logics, you can create a new logic class that inherits from `Aggr2SGrpBLat` and implements the required methods. The search stops at the first matching logic class found and starts with the built-in logics. To overwrite a builtin implementation with a custom implementation, it has to be instantiated (second overload of `nesy` function).

    Args:
        monad_type: monad type.
        omega: The type of truth values. Defaults to `bool`.

    Returns:
        An instance of `NeSyFramework` with the specified monad and truth value types.

    Example:
        ::

            from muller import nesy, Prob
            from muller.logics import Aggr2SGrpBLat

            class MyLogicOverwrite(Aggr2SGrpBLat[Prob[bool]]):
                ...

            class MyCustomLogic(Aggr2SGrpBLat[Prob[str]]):
                ...

            nesy_framework = nesy(Prob, bool) # Uses `muller.logics.ProbabilisticBooleanLogic`
            nesy_framework = nesy(MyLogicOverwrite()) # Uses `MyLogicOverwrite`
            nesy_framework = nesy(Prob, str) # Uses `MyCustomLogic`
    """
    ...


def nesy[O](  # pyright: ignore[reportInconsistentOverload]
    arg1: Aggr2SGrpBLat[ParametrizedMonad[O]] | Type[ParametrizedMonad],
    omega: Type[O] = Type[bool],
) -> NeSyFramework[ParametrizedMonad[O], O, Aggr2SGrpBLat[ParametrizedMonad[O]]]:
    """
    Create a NeSyFramework instance based on the provided argument.

    Args:
        arg1: Either an instance of Aggr2SGrpBLat or a monad type.
        omega: The type of truth values. Defaults to `bool`.

    Returns:
        An instance of NeSyFramework.
    """
    if isinstance(arg1, Aggr2SGrpBLat):
        return nesy_for_logic(arg1)
    else:
        return nesy_framework_for_monad(arg1, omega)

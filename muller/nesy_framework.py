from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Type, cast, overload
import re

from pymonad.monad import Monad

from muller.monad.util import bind


from muller.logics import Aggr2SGrpBLat, get_logic


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


type SingleArgumentTypeFunction[I, O] = Callable[[], O] | Callable[[I], O] | Callable[
    [I, I], O
] | Callable[[I, I, I], O] | Callable[[I, I, I, I], O]


@dataclass
class Interpretation[A, O]:
    universe: list[A]
    functions: dict[Ident, SingleArgumentTypeFunction[A, A]]
    mfunctions: dict[Ident, SingleArgumentTypeFunction[A, Monad[A]]]
    preds: dict[Ident, SingleArgumentTypeFunction[A, O]]
    mpreds: dict[Ident, SingleArgumentTypeFunction[A, Monad[O]]]


type Valuation[A] = dict[Ident, A]


class NeSyFramework[_T: Monad, _O, _R: Aggr2SGrpBLat]:
    _monad: Type[_T]
    _logic: _R

    @property
    def M(self) -> Type[Monad[_O]]:
        return self._monad

    @property
    def logic(self) -> Aggr2SGrpBLat[Monad[_O]]:
        return self._logic

    @property
    def logic_T(self) -> Aggr2SGrpBLat[_T]:
        return self._logic

    def __init__(
        self,
        monad: type[_T],
        logic: Aggr2SGrpBLat[Monad[_O]],
    ) -> None:
        self._monad = monad

        self._logic = cast(_R, logic)

    def unitT(self, value: _O) -> _T:
        """Returns a monadic value of type 'T' with the value 'value' in it."""
        return cast(_T, self.M.insert(value))

    @overload
    def castT(self, m: _T) -> Monad[_O]:
        return cast(_T, m)

    @overload
    def castT(self, m: Monad[_O]) -> _T:
        return cast(_T, m)

    def castT(self, m) -> Monad[_O] | _T:
        return m


class Term(ABC):
    """
    Base class for all terms in the logic system.
    Terms can be variables or applications of functions to arguments.
    """

    def eval[T: Monad, O, R: Aggr2SGrpBLat, A](
        self,
        nesy: "NeSyFramework[T, O, R]",
        interpretation: "Interpretation[A, O]",
        valuation: "Valuation[A]",
    ) -> A:
        raise NotImplementedError()


@dataclass
class Variable(Term):
    ident: Ident

    def eval[T: Monad, O, R: Aggr2SGrpBLat, A](
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

    def eval[T: Monad, O, R: Aggr2SGrpBLat, A](
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
    def eval[T: Monad, O, R: Aggr2SGrpBLat, A](
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

    def eval[T: Monad, O, R: Aggr2SGrpBLat, A](
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

    def eval[T: Monad, O, R: Aggr2SGrpBLat, A](
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
        args_str = ", ".join(str(arg) for arg in self.arguments)
        formatted_name = _format_predicate_ident(self.name)
        if not args_str:
            return formatted_name
        return f"{formatted_name}({args_str})"

    def eval[T: Monad, O, R: Aggr2SGrpBLat, A](
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

    def eval[T: Monad, O, R: Aggr2SGrpBLat, A](
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

    def eval[T: Monad, O, R: Aggr2SGrpBLat, A](
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

    def eval[T: Monad, O, R: Aggr2SGrpBLat, A](
        self,
        nesy: "NeSyFramework[T, O, R]",
        interpretation: "Interpretation[A, O]",
        valuation: "Valuation[A]",
    ) -> T:
        l = self.left.eval(nesy, interpretation, valuation)
        r = self.right.eval(nesy, interpretation, valuation)
        return nesy.logic_T.conjunction(l, r)


@dataclass
class Disjunction(Formula):
    """
    Represents the disjunction (OR) of two formulas.
    """

    left: Formula
    right: Formula

    def __repr__(self):
        return f"({self.left} ∨ {self.right})"

    def eval[T: Monad, O, R: Aggr2SGrpBLat, A](
        self,
        nesy: "NeSyFramework[T, O, R]",
        interpretation: "Interpretation[A, O]",
        valuation: "Valuation[A]",
    ) -> T:
        l = self.left.eval(nesy, interpretation, valuation)
        r = self.right.eval(nesy, interpretation, valuation)
        return nesy.logic_T.disjunction(l, r)


@dataclass
class Implication(Formula):
    """
    Represents the implication (IF...THEN) of two formulas.
    """

    antecedent: Formula
    consequent: Formula

    def __repr__(self):
        return f"({self.antecedent} -> {self.consequent})"

    def eval[T: Monad, O, R: Aggr2SGrpBLat, A](
        self,
        nesy: "NeSyFramework[T, O, R]",
        interpretation: "Interpretation[A, O]",
        valuation: "Valuation[A]",
    ) -> T:
        l = self.antecedent.eval(nesy, interpretation, valuation)
        r = self.consequent.eval(nesy, interpretation, valuation)
        return nesy.logic_T.implies(l, r)


@dataclass
class UniversalQuantification(Formula):
    """
    Represents a universally quantified formula.
    """

    variable: Ident
    formula: Formula

    def __repr__(self):
        return f"∀{self.variable} ({self.formula})"

    def eval[T: Monad, O, R: Aggr2SGrpBLat, A](
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

    def eval[T: Monad, O, R: Aggr2SGrpBLat, A](
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

    def eval[T: Monad, O, R: Aggr2SGrpBLat, A](
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


def nesy_for_logic[O](
    logic: Aggr2SGrpBLat[Monad],
    omega: Type[O] = Type[bool]
) -> NeSyFramework[Monad[O], O, Aggr2SGrpBLat[Monad[O]]]:
    """
    Create a NeSyFramework instance for the given logic.

    Args:
        logic: An instance of Aggr2SGrpBLat representing the logic.

    Returns:
        An instance of NeSyFramework with the logic's monad and omega type.
    """
    monad_type = type(logic.top())
    assert issubclass(monad_type, Monad), "Logic must be over a Monad type"

    return NeSyFramework(monad_type, logic)


def nesy_framework_for_monad[O](
    monad_type: Type[Monad], omega: Type[O] = Type[bool]
) -> NeSyFramework[Monad, Any, Aggr2SGrpBLat[Monad]]:
    """
    Create a NeSyFramework instance with the given monad type and optional omega type.

    Args:
        monad_type: The type of the monad to use.
        omega: Optional type for the outcomes of the monadic computations.

    Returns:
        An instance of NeSyFramework with the specified monad and omega types.
    """

    return NeSyFramework(monad_type, get_logic(monad_type, omega))


@overload
def nesy[O](
    logic: Aggr2SGrpBLat[Monad],
    omega: Type[O] = Type[bool]
) -> NeSyFramework[Monad[O], O, Aggr2SGrpBLat[Monad[O]]]: ...
@overload
def nesy[O](
    monad_type: Type[Monad], omega: Type[O] = Type[bool]
) -> NeSyFramework[Monad, Any, Aggr2SGrpBLat[Monad]]: ...
def nesy[O](  # pyright: ignore[reportInconsistentOverload]
    arg1: Aggr2SGrpBLat[Monad[O]] | Type[Monad], omega: Type[O] = Type[bool]
) -> NeSyFramework[Monad, Any, Aggr2SGrpBLat[Monad]]:
    """
    Create a NeSyFramework instance based on the provided argument.

    Args:
        arg1: Either an instance of Aggr2SGrpBLat or a monad type.
        omega: Optional type for the outcomes of the monadic computations.

    Returns:
        An instance of NeSyFramework.
    """
    if isinstance(arg1, Aggr2SGrpBLat):
        return nesy_for_logic(arg1)
    else:
        return nesy_framework_for_monad(arg1, omega)

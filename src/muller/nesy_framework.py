from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Type, TypeVar, assert_type, cast, overload

from pymonad.monad import Monad

from .aggr2sgrpblat.aggr2sgrpblat import non_empty_set_bool

from .monad.util import bind

from .non_empty_powerset_monad import NonEmptyPowerset

from .aggr2sgrpblat.aggr2sgrpblat import (
    Aggr2SGrpBLat,
)



Ident = str

type SingleArgumentTypeFunction[I, O] = Callable[[], O] | Callable[[I], O] | Callable[[I, I], O] | Callable[[I, I, I], O]  | Callable[[I, I, I, I], O]


@dataclass
class Interpretation[A, O]:
    universe: list[A]
    functions: dict[Ident, SingleArgumentTypeFunction[A, A]]
    mfunctions: dict[Ident, SingleArgumentTypeFunction[A, Monad[A]]]
    preds: dict[Ident, SingleArgumentTypeFunction[A, O]]
    mpreds: dict[Ident, SingleArgumentTypeFunction[A, Monad[O]]]


type Valuation[A] = dict[Ident, A]


class NeSyFramework[_T: Monad, _O, _R: Aggr2SGrpBLat]:
    _t: type[_T]
    _omega: type[_O]
    _r: _R


    @property
    def T(self) -> Type[_T]:
        return self._t

    @property
    def T_O(self) -> Type[Monad[_O]]:
        return self._t

    @property
    def R(self) -> _R:
        return self._r

    @property
    def R_O(self) -> Aggr2SGrpBLat[Monad[_O]]:
        return self._r

    @property
    def R_T(self) -> Aggr2SGrpBLat[_T]:
        return self._r

    def __init__(
        self,
        monad_type: type[_T],
        omega: type[_O],
        r: Aggr2SGrpBLat[Monad[_O]],
    ) -> None:
        self._t = monad_type
        self._omega = omega
        self._r = cast(_R, r)

    def unitT(self, value: _O) -> _T:
        """Returns a monadic value of type 'T' with the value 'value' in it."""
        return cast(_T, self.T.insert(value))

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


@dataclass
class Application(Term):
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
        return nesy.R_T.top()


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
        return nesy.R_T.bottom()


@dataclass
class Predicate(Formula):
    """
    Represents a predicate with a name and arguments.
    """

    name: Ident
    arguments: list[Term]

    def __repr__(self):
        args_str = ", ".join(str(arg) for arg in self.arguments)
        return f"{self.name}({args_str})"

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
        return f"{self.name}({self.arguments})"

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
        return nesy.R_T.neg(self.formula.eval(nesy, interpretation, valuation))


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
        return nesy.R_T.conjunction(l, r)


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
        return nesy.R_T.disjunction(l, r)


@dataclass
class Implication(Formula):
    """
    Represents the implication (IF...THEN) of two formulas.
    """

    antecedent: Formula
    consequent: Formula

    def __repr__(self):
        return f"({self.antecedent} → {self.consequent})"

    def eval[T: Monad, O, R: Aggr2SGrpBLat, A](
        self,
        nesy: "NeSyFramework[T, O, R]",
        interpretation: "Interpretation[A, O]",
        valuation: "Valuation[A]",
    ) -> T:
        l = self.antecedent.eval(nesy, interpretation, valuation)
        r = self.consequent.eval(nesy, interpretation, valuation)
        return nesy.R_T.implies(l, r)


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
        results = (self.formula.eval(nesy, interpretation, {**valuation, self.variable: a}) for a in universe)
        return nesy.R_T.aggrA(results)


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
        results = (self.formula.eval(nesy, interpretation, {**valuation, self.variable: a}) for a in universe)
        return nesy.R_T.aggrE(results)


@dataclass
class Computation(Formula):
    variable: Ident
    function: Ident
    arguments: list[Term]
    formula: Formula

    def __repr__(self):
        args_str = ", ".join(str(arg) for arg in self.arguments)
        return f"{self.variable} := {self.function}({args_str})({self.formula})"
    


    def eval[T: Monad, O, R: Aggr2SGrpBLat, A](
        self,
        nesy: "NeSyFramework[T, O, R]",
        interpretation: "Interpretation[A, O]",
        valuation: "Valuation[A]",
    ) -> T:
        f = interpretation.mfunctions[self.function]
        args = [a.eval(nesy, interpretation, valuation) for a in self.arguments]
        result = f(*args)
        x = bind(result, lambda res: nesy.castT( self.formula.eval(nesy, interpretation, {**valuation, self.variable: res})))
        return nesy.castT(x)

 # pyright: ignore[reportArgumentType]
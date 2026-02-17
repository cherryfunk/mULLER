from abc import ABC
from dataclasses import dataclass
import re
from typing import Any, Never
from lark import Lark, Transformer
parser = Lark(
    r"""
%import common.WS
%ignore WS

%import common.ESCAPED_STRING   -> STRING

var_ident: /[A-Z][a-zA-Z0-9_]*/
func_ident: /[a-z][a-zA-Z0-9_]*/ | /[0-9]+/ | /'[^']*'/
mfunc_ident: /\$[a-z][a-zA-Z0-9_]*/ | /\$[0-9]+/ | /\$'[^']*'/
pred_ident: /[a-z][a-zA-Z0-9_]*/ | /'[^']*'/
mpred_ident: /\$[a-z][a-zA-Z0-9_]*/ | /\$'[^']*'/


?formula : implication

?implication : disjunction
             | disjunction "->" implication -> implication

?disjunction : conjunction
             | disjunction "or" conjunction -> disjunction
             | disjunction "∨" conjunction -> disjunction
             | disjunction "|" conjunction -> disjunction

?conjunction : negation
             | conjunction "and" negation -> conjunction
             | conjunction "∧" negation -> conjunction
             | conjunction "&" negation -> conjunction

?negation : "not" negation -> negation
          | "~" negation -> negation
          | quantified

?quantified : "?[" var_ident "]:" quantified -> exists
            | "![" var_ident "]:" quantified -> forall
            | computation

?computation : computation_sequence
             | atom

?computation_sequence : computation_assignment ("," computation_assignment)* "(" formula ")" -> computation_sequence
                     | computation_assignment

?computation_assignment : var_ident ":=" mfunc_ident "(" [term ("," term)*] ")" -> computation_assignment

?atom : "T" -> true
      | "F" -> false
      | "(" formula ")"
      | "(" computation_assignment ")" "(" formula ")" -> parenthesized_computation
      | term "==" term -> equality
      | term "!=" term -> not_equal
      | term "<" term -> lessthan
      | term "<=" term -> less_equal
      | term ">" term -> greaterthan
      | term ">=" term -> greater_equal
      | pred_ident "(" [term ("," term)*] ")" -> predicate
      | pred_ident -> predicate
      | mpred_ident "(" [term ("," term)*] ")" -> computational_predicate
      | mpred_ident -> computational_predicate

term : func_ident -> constant
     | term "." func_ident -> property
     | func_ident "(" [term ("," term)*] ")" -> function
     | var_ident -> variable
""",  # noqa: E501
    start="formula",
)

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

class Term(ABC):
    """
    Base class for all terms in the logic system.
    Terms can be variables or applications of functions to arguments.
    """


@dataclass
class Variable(Term):
    ident: Ident

    def __repr__(self) -> str:
        return self.ident


@dataclass
class FunctionApplication(Term):
    function: Ident
    arguments: list[Term]

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


@dataclass
class TrueFormula(Formula):
    """
    Represents the logical constant True.
    """

    def __repr__(self) -> str:
        return "True"


@dataclass
class FalseFormula(Formula):
    """
    Represents the logical constant False.
    """

    def __repr__(self) -> str:
        return "False"

@dataclass
class Predicate(Formula):
    """
    Represents a predicate with a name and arguments.
    """

    name: Ident
    arguments: list[Term]

    def __repr__(self) -> str:
        # Special case for equality operator: use infix notation
        if self.name in {"==", "<", ">", "<=", ">="} and len(self.arguments) == 2:
            left, right = self.arguments
            # Add parentheses for clarity, similar to other binary operators
            return f"({left} {self.name} {right})"

        # Standard predicate representation
        args_str = ", ".join(str(arg) for arg in self.arguments)
        formatted_name = _format_predicate_ident(self.name)
        if not args_str:
            return formatted_name
        return f"{formatted_name}({args_str})"


@dataclass
class MonadicPredicate(Formula):
    """
    Represents a monadic predicate (a predicate with a single argument).
    """

    name: Ident
    arguments: list[Term]

    def __repr__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.arguments)
        formatted_name = _format_predicate_ident(self.name)
        if not args_str:
            return formatted_name
        return f"${formatted_name}({args_str})"


@dataclass
class Negation(Formula):
    """
    Represents the negation of a formula.
    """

    formula: Formula

    def __repr__(self) -> str:
        return f"¬{self.formula}"


@dataclass
class Conjunction(Formula):
    """
    Represents the conjunction (AND) of two formulas.
    """

    left: Formula
    right: Formula

    def __repr__(self) -> str:
        return f"({self.left} ∧ {self.right})"


@dataclass
class Disjunction(Formula):
    """
    Represents the disjunction (OR) of two formulas.
    """

    left: Formula
    right: Formula

    def __repr__(self) -> str:
        return f"({self.left} ∨ {self.right})"


@dataclass
class Implication(Formula):
    """
    Represents the implication (IF...THEN) of two formulas.
    """

    antecedent: Formula
    consequent: Formula

    def __repr__(self) -> str:
        return f"({self.antecedent} -> {self.consequent})"


@dataclass
class UniversalQuantification(Formula):
    """
    Represents a universally quantified formula.
    """

    variable: Ident
    formula: Formula

    def __repr__(self) -> str:
        return f"∀{self.variable} ({self.formula})"


@dataclass
class ExistentialQuantification(Formula):
    """
    Represents an existentially quantified formula.
    """

    variable: Ident
    formula: Formula

    def __repr__(self) -> str:
        return f"∃{self.variable} ({self.formula})"


@dataclass
class Computation(Formula):
    variable: Ident
    function: Ident
    arguments: list[Term]
    formula: Formula

    def __repr__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.arguments)
        formatted_function = _format_function_ident(self.function)
        return f"{self.variable} := ${formatted_function}({args_str})({self.formula})"




class NeSyTransformer(Transformer[Any, Formula]):
    def true(self, _: Never) -> TrueFormula:
        """Transform 'T' into a TrueFormula."""
        return TrueFormula()

    def false(self, _: Never) -> FalseFormula:
        """Transform 'F' into a FalseFormula."""
        return FalseFormula()

    def _remove_quotes(self, ident: str) -> str:
        """Remove surrounding single quotes if present."""
        if ident.startswith("'") and ident.endswith("'"):
            return ident[1:-1]
        return ident

    def _remove_dollar_prefix(self, ident: str) -> str:
        """Remove $ prefix if present."""
        if ident.startswith("$"):
            return ident[1:]
        return ident

    def _process_monadic_ident(self, ident: str) -> str:
        """Process monadic identifier by removing $ prefix and quotes."""
        ident = self._remove_dollar_prefix(ident)
        return self._remove_quotes(ident)

    # Handle different identifier types
    def var_ident(self, items: list[str]) -> str:
        return items[0]

    def func_ident(self, items: list[str]) -> str:
        return self._remove_quotes(items[0])

    def mfunc_ident(self, items: list[str]) -> str:
        return self._process_monadic_ident(items[0])

    def pred_ident(self, items: list[str]) -> str:
        return self._remove_quotes(items[0])

    def mpred_ident(self, items: list[str]) -> str:
        return self._process_monadic_ident(items[0])

    def variable(self, items: list[str]) -> Variable:
        return Variable(items[0])

    def constant(self, children: list[str]) -> Term:
        [name] = children
        # Constants are function names used as nullary functions
        return FunctionApplication(name, [])

    def property(self, children: list[Any]) -> Term:
        [obj, prop] = children
        # Property access: obj.prop becomes prop(obj)
        return FunctionApplication(prop, [obj])

    def function(self, children: list[Any]) -> Term:
        [name, *args] = children
        # Filter out None values that come from empty optional groups
        args = [arg for arg in args if arg is not None]
        return FunctionApplication(name, args)

    def negation(self, children: list[Any]) -> Negation:
        [f] = children
        return Negation(f)

    def predicate(self, children: list[Any]) -> Formula:
        [name, *args] = children
        # Filter out None values that come from empty optional groups
        args = [arg for arg in args if arg is not None]
        return Predicate(name, args)

    def computational_predicate(self, children: list[Any]) -> Formula:
        [name, *args] = children
        # Filter out None values that come from empty optional groups
        args = [arg for arg in args if arg is not None]
        # Use MonadicPredicate for computational predicates
        return MonadicPredicate(name, args)

    def equality(self, children: list[Any]) -> Predicate:
        [left, right] = children
        return Predicate("==", [left, right])

    def lessthan(self, children: list[Any]) -> Predicate:
        [left, right] = children
        return Predicate("<", [left, right])

    def greaterthan(self, children: list[Any]) -> Predicate:
        [left, right] = children
        return Predicate(">", [left, right])

    def not_equal(self, children: list[Any]) -> Predicate:
        [left, right] = children
        return Predicate("!=", [left, right])

    def less_equal(self, children: list[Any]) -> Predicate:
        [left, right] = children
        return Predicate("<=", [left, right])

    def greater_equal(self, children: list[Any]) -> Predicate:
        [left, right] = children
        return Predicate(">=", [left, right])

    def conjunction(self, children: list[Any]) -> Conjunction:
        [left, right] = children
        return Conjunction(left, right)

    def disjunction(self, children: list[Any]) -> Disjunction:
        [left, right] = children
        return Disjunction(left, right)

    def implication(self, children: list[Any]) -> Implication:
        [antecedent, consequent] = children
        return Implication(antecedent, consequent)

    def forall(self, children: list[Any]) -> UniversalQuantification:
        [variable, formula] = children
        return UniversalQuantification(variable, formula)

    def exists(self, children: list[Any]) -> ExistentialQuantification:
        [variable, formula] = children
        return ExistentialQuantification(variable, formula)

    def computation_assignment(self, children: list[Any]) -> tuple[Any, Any, list[Any]]:
        """Handle a single computation assignment like 'X := $f(args)'."""
        [variable, function, *args] = children
        # Filter out None values that come from empty optional groups
        args = [arg for arg in args if arg is not None]
        return (variable, function, args)

    def computation_sequence(self, children: list[Any]) -> Computation:
        """Handle a sequence of computation assignments followed by a formula."""
        # The last element is always the formula
        # All other elements are computation assignments (tuples)
        [*assignments, formula] = children

        # Build nested computations from right to left
        result_formula: Computation = formula
        for variable, function, args in reversed(assignments):
            result_formula = Computation(variable, function, args, result_formula)

        return result_formula


def parse(formula: str) -> Formula:
    """
    Parse a NeSy formula and return a Formula object.
    """
    tree = parser.parse(formula)
    return NeSyTransformer().transform(tree)

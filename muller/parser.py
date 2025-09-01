from lark import Lark, Transformer

from muller.nesy_framework import (
    Computation,
    Conjunction,
    Disjunction,
    ExistentialQuantification,
    FalseFormula,
    Formula,
    FunctionApplication,
    Implication,
    MonadicPredicate,
    Negation,
    Predicate,
    Term,
    TrueFormula,
    UniversalQuantification,
    Variable,
)

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


class NeSyTransformer(Transformer):
    def true(self, _) -> TrueFormula:
        """Transform 'T' into a TrueFormula."""
        return TrueFormula()

    def false(self, _) -> FalseFormula:
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
    def var_ident(self, items):
        return items[0]

    def func_ident(self, items):
        return self._remove_quotes(items[0])

    def mfunc_ident(self, items):
        return self._process_monadic_ident(items[0])

    def pred_ident(self, items):
        return self._remove_quotes(items[0])

    def mpred_ident(self, items):
        return self._process_monadic_ident(items[0])

    def variable(self, items):
        return Variable(items[0])

    def constant(self, children: list) -> Term:
        [name] = children
        # Constants are function names used as nullary functions
        return FunctionApplication(name, [])

    def property(self, children: list) -> Term:
        [obj, prop] = children
        # Property access: obj.prop becomes prop(obj)
        return FunctionApplication(prop, [obj])

    def function(self, children: list) -> Term:
        [name, *args] = children
        # Filter out None values that come from empty optional groups
        args = [arg for arg in args if arg is not None]
        return FunctionApplication(name, args)

    def negation(self, children: list) -> Negation:
        [f] = children
        return Negation(f)

    def predicate(self, children: list) -> Formula:
        [name, *args] = children
        # Filter out None values that come from empty optional groups
        args = [arg for arg in args if arg is not None]
        return Predicate(name, args)

    def computational_predicate(self, children: list) -> Formula:
        [name, *args] = children
        # Filter out None values that come from empty optional groups
        args = [arg for arg in args if arg is not None]
        # Use MonadicPredicate for computational predicates
        return MonadicPredicate(name, args)

    def equality(self, children: list) -> Predicate:
        [left, right] = children
        return Predicate("==", [left, right])

    def lessthan(self, children: list) -> Predicate:
        [left, right] = children
        return Predicate("<", [left, right])

    def greaterthan(self, children: list) -> Predicate:
        [left, right] = children
        return Predicate(">", [left, right])

    def not_equal(self, children: list) -> Predicate:
        [left, right] = children
        return Predicate("!=", [left, right])

    def less_equal(self, children: list) -> Predicate:
        [left, right] = children
        return Predicate("<=", [left, right])

    def greater_equal(self, children: list) -> Predicate:
        [left, right] = children
        return Predicate(">=", [left, right])

    def conjunction(self, children: list) -> Conjunction:
        [left, right] = children
        return Conjunction(left, right)

    def disjunction(self, children: list) -> Disjunction:
        [left, right] = children
        return Disjunction(left, right)

    def implication(self, children: list) -> Implication:
        [antecedent, consequent] = children
        return Implication(antecedent, consequent)

    def forall(self, children: list) -> UniversalQuantification:
        [variable, formula] = children
        return UniversalQuantification(variable, formula)

    def exists(self, children: list) -> ExistentialQuantification:
        [variable, formula] = children
        return ExistentialQuantification(variable, formula)

    def computation_assignment(self, children: list) -> tuple:
        """Handle a single computation assignment like 'X := $f(args)'."""
        [variable, function, *args] = children
        # Filter out None values that come from empty optional groups
        args = [arg for arg in args if arg is not None]
        return (variable, function, args)

    def computation_sequence(self, children: list) -> Computation:
        """Handle a sequence of computation assignments followed by a formula."""
        # The last element is always the formula
        # All other elements are computation assignments (tuples)
        [*assignments, formula] = children

        # Build nested computations from right to left
        result_formula = formula
        for variable, function, args in reversed(assignments):
            result_formula = Computation(variable, function, args, result_formula)

        return result_formula


def parse(formula: str) -> Formula:
    """
    Parse a NeSy formula and return a Formula object.
    """
    tree = parser.parse(formula)
    return NeSyTransformer().transform(tree)

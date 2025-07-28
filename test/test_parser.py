from typing import cast
import unittest
from muller.nesy_framework import (
    FalseFormula,
    Predicate,
    TrueFormula,
    Negation,
    Variable,
    Conjunction,
    Disjunction,
    Implication,
    UniversalQuantification,
    ExistentialQuantification,
    FunctionApplication,
    Computation,
    MonadicPredicate,
)
from muller.parser import parse
from unittest import TestCase
from lark import LarkError


class ParserTestCase(TestCase):

    def test_true(self):
        f = parse("T")
        self.assertEqual(f, TrueFormula())

    def test_false(self):
        f = parse("F")
        self.assertEqual(f, FalseFormula())

    def test_predicate_1(self):
        f = parse("human(X)")
        self.assertEqual(f, Predicate("human", [Variable("X")]))

    def test_predicate_no_args(self):
        f = parse("raining()")
        self.assertEqual(f, Predicate("raining", []))

    def test_predicate_multiple_args(self):
        f = parse("likes(John, Mary)")
        self.assertEqual(f, Predicate("likes", [Variable("John"), Variable("Mary")]))

    def test_predicate_with_function_args(self):
        f = parse("taller(X, father(X))")
        self.assertEqual(
            f,
            Predicate(
                "taller",
                [Variable("X"), FunctionApplication("father", [Variable("X")])],
            ),
        )

    def test_negation(self):
        f = parse("not T")
        self.assertEqual(f, Negation(TrueFormula()))

    def test_negation_complex(self):
        f = parse("not human(X)")
        self.assertEqual(f, Negation(Predicate("human", [Variable("X")])))

    def test_parentheses(self):
        f = parse("(T)")
        self.assertEqual(f, TrueFormula())

    def test_nested_parentheses(self):
        f = parse("((human(X)))")
        self.assertEqual(f, Predicate("human", [Variable("X")]))

    def test_variable_term(self):
        f = parse("human(X)")
        self.assertIsInstance(f, Predicate)
        predicate = cast(Predicate, f)
        expected_var = Variable("X")
        self.assertEqual(predicate.arguments[0], expected_var)

    def test_function_term(self):
        f = parse("human(father(X, Y))")
        self.assertIsInstance(f, Predicate)
        predicate = cast(Predicate, f)
        expected_func = FunctionApplication("father", [Variable("X"), Variable("Y")])
        self.assertEqual(predicate.arguments[0], expected_func)

    def test_nested_function_term(self):
        f = parse("human(father(mother(X)))")
        self.assertIsInstance(f, Predicate)
        predicate = cast(Predicate, f)
        expected_func = FunctionApplication(
            "father", [FunctionApplication("mother", [Variable("X")])]
        )
        self.assertEqual(predicate.arguments[0], expected_func)

    # Test new identifier types
    def test_constant_term(self):
        f = parse("human(john)")  # lowercase function as constant
        self.assertIsInstance(f, Predicate)
        predicate = cast(Predicate, f)
        expected_const = FunctionApplication("john", [])  # nullary function
        self.assertEqual(predicate.arguments[0], expected_const)

    def test_property_access(self):
        f = parse("human(Person.name)")
        self.assertIsInstance(f, Predicate)
        predicate = cast(Predicate, f)
        expected_prop = FunctionApplication("name", [Variable("Person")])
        self.assertEqual(predicate.arguments[0], expected_prop)

    def test_monadic_predicate(self):
        f = parse("$probability(X)")
        self.assertEqual(
            f, MonadicPredicate("probability", [Variable("X")])
        )  # Test logical connectives

    def test_conjunction(self):
        f = parse("T and F")
        self.assertEqual(f, Conjunction(TrueFormula(), FalseFormula()))

    def test_disjunction(self):
        f = parse("T or F")
        self.assertEqual(f, Disjunction(TrueFormula(), FalseFormula()))

    def test_implication(self):
        f = parse("T -> F")
        self.assertEqual(f, Implication(TrueFormula(), FalseFormula()))

    # Test equality operator
    def test_equality_variables(self):
        f = parse("X == Y")
        self.assertEqual(f, Predicate("==", [Variable("X"), Variable("Y")]))

    def test_equality_variable_constant(self):
        f = parse("X == 5")
        expected = Predicate("==", [Variable("X"), FunctionApplication("5", [])])
        self.assertEqual(f, expected)

    def test_equality_functions(self):
        f = parse("f(X) == g(Y)")
        expected = Predicate("==", [
            FunctionApplication("f", [Variable("X")]),
            FunctionApplication("g", [Variable("Y")])
        ])
        self.assertEqual(f, expected)

    def test_equality_with_conjunction(self):
        f = parse("X == Y and Y == Z")
        expected = Conjunction(
            Predicate("==", [Variable("X"), Variable("Y")]),
            Predicate("==", [Variable("Y"), Variable("Z")])
        )
        self.assertEqual(f, expected)

    def test_equality_with_negation(self):
        f = parse("not (X == Y)")
        expected = Negation(Predicate("==", [Variable("X"), Variable("Y")]))
        self.assertEqual(f, expected)

    def test_complex_logical_expression(self):
        f = parse("human(X) and mortal(Y)")
        expected = Conjunction(
            Predicate("human", [Variable("X")]), Predicate("mortal", [Variable("Y")])
        )
        self.assertEqual(f, expected)

    def test_negation_of_conjunction(self):
        f = parse("not (human(X) and mortal(Y))")
        expected = Negation(
            Conjunction(
                Predicate("human", [Variable("X")]),
                Predicate("mortal", [Variable("Y")]),
            )
        )
        self.assertEqual(f, expected)

    # Test quantifiers
    def test_universal_quantification(self):
        f = parse("forall X human(X)")
        self.assertEqual(
            f, UniversalQuantification("X", Predicate("human", [Variable("X")]))
        )

    def test_existential_quantification(self):
        f = parse("exists X human(X)")
        self.assertEqual(
            f, ExistentialQuantification("X", Predicate("human", [Variable("X")]))
        )

    def test_nested_quantifiers(self):
        f = parse("forall X exists Y likes(X, Y)")
        expected = UniversalQuantification(
            "X",
            ExistentialQuantification(
                "Y", Predicate("likes", [Variable("X"), Variable("Y")])
            ),
        )
        self.assertEqual(f, expected)

    # Test computation with monadic functions
    def test_computation(self):
        f = parse("X := $compute() (valid(X))")
        expected = Computation("X", "compute", [], Predicate("valid", [Variable("X")]))
        self.assertEqual(f, expected)

    def test_computation_with_args(self):
        f = parse("Result := $process(Input) (valid(Result))")
        expected = Computation(
            "Result",
            "process",
            [Variable("Input")],
            Predicate("valid", [Variable("Result")]),
        )
        self.assertEqual(f, expected)

    # Test edge cases
    def test_predicate_with_special_symbols(self):
        f = parse("equals(X, Y)")
        self.assertEqual(f, Predicate("equals", [Variable("X"), Variable("Y")]))

    def test_predicate_with_comparison(self):
        f = parse("greater(X, Y)")
        self.assertEqual(f, Predicate("greater", [Variable("X"), Variable("Y")]))

    # Test operator precedence
    def test_precedence_and_or(self):
        # Should parse as: human(X) and (mortal(Y) or divine(Z))
        f = parse("human(X) and mortal(Y) or divine(Z)")
        expected = Disjunction(
            Conjunction(
                Predicate("human", [Variable("X")]),
                Predicate("mortal", [Variable("Y")]),
            ),
            Predicate("divine", [Variable("Z")]),
        )
        self.assertEqual(f, expected)

    def test_precedence_implication_and(self):
        # Should parse as: (human(X) and mortal(Y)) -> dead(Z)
        f = parse("human(X) and mortal(Y) -> dead(Z)")
        expected = Implication(
            Conjunction(
                Predicate("human", [Variable("X")]),
                Predicate("mortal", [Variable("Y")]),
            ),
            Predicate("dead", [Variable("Z")]),
        )
        self.assertEqual(f, expected)

    def test_precedence_negation_and(self):
        # Should parse as: (not human(X)) and mortal(Y)
        f = parse("not human(X) and mortal(Y)")
        expected = Conjunction(
            Negation(Predicate("human", [Variable("X")])),
            Predicate("mortal", [Variable("Y")]),
        )
        self.assertEqual(f, expected)

    def test_double_negation(self):
        f = parse("not not T")
        expected = Negation(Negation(TrueFormula()))
        self.assertEqual(f, expected)

    # Test complex nested expressions
    def test_complex_nested_expression(self):
        f = parse("forall X (human(X) -> exists Y (parent(X, Y) and loves(Y)))")
        expected = UniversalQuantification(
            "X",
            Implication(
                Predicate("human", [Variable("X")]),
                ExistentialQuantification(
                    "Y",
                    Conjunction(
                        Predicate("parent", [Variable("X"), Variable("Y")]),
                        Predicate("loves", [Variable("Y")]),
                    ),
                ),
            ),
        )
        self.assertEqual(f, expected)

    def test_computation_in_quantifier(self):
        f = parse("exists X (X := $transform(Y) (valid(X)))")
        expected = ExistentialQuantification(
            "X",
            Computation(
                "X", "transform", [Variable("Y")], Predicate("valid", [Variable("X")])
            ),
        )
        self.assertEqual(f, expected)

    # Additional tests for new grammar features
    def test_monadic_function_in_computation(self):
        f = parse("Result := $neural_network(Input) (confidence(Result))")
        expected = Computation(
            "Result",
            "neural_network",
            [Variable("Input")],
            Predicate("confidence", [Variable("Result")]),
        )
        self.assertEqual(f, expected)

    def test_multiple_property_access(self):
        f = parse("human(Person.address.city)")
        expected = Predicate(
            "human",
            [
                FunctionApplication(
                    "city", [FunctionApplication("address", [Variable("Person")])]
                )
            ],
        )
        self.assertEqual(f, expected)

    # Test identifier type enforcement
    def test_variable_must_be_uppercase(self):
        # Variables must start with uppercase
        f = parse("human(Person)")
        self.assertIsInstance(f, Predicate)
        predicate = cast(Predicate, f)
        self.assertIsInstance(predicate.arguments[0], Variable)
        var = cast(Variable, predicate.arguments[0])
        self.assertEqual(var.ident, "Person")

    def test_function_must_be_lowercase(self):
        # Functions must start with lowercase
        f = parse("human(father(X))")
        self.assertIsInstance(f, Predicate)
        predicate = cast(Predicate, f)
        self.assertIsInstance(predicate.arguments[0], FunctionApplication)
        func_app = cast(FunctionApplication, predicate.arguments[0])
        self.assertEqual(func_app.function, "father")

    def test_predicate_must_be_lowercase(self):
        # Predicates must start with lowercase
        f = parse("human(X)")
        self.assertIsInstance(f, Predicate)
        predicate = cast(Predicate, f)
        self.assertEqual(predicate.name, "human")

    def test_monadic_predicate_dollar_prefix(self):
        # Monadic predicates must have $ prefix
        f = parse("$neural(X)")
        self.assertIsInstance(f, MonadicPredicate)
        mpred = cast(MonadicPredicate, f)
        self.assertEqual(mpred.name, "neural")  # $ is stripped

    def test_monadic_function_dollar_prefix(self):
        # Monadic functions must have $ prefix
        f = parse("X := $compute(Y) (valid(X))")
        self.assertIsInstance(f, Computation)
        comp = cast(Computation, f)
        self.assertEqual(comp.function, "compute")  # $ is stripped

    # Test string identifiers - now we support complex identifiers in single quotes
    def test_string_identifier(self):
        f = parse("'is greater than'(X, Y)")
        self.assertEqual(
            f, Predicate("is greater than", [Variable("X"), Variable("Y")])
        )

    def test_complex_predicate_names(self):
        f = parse("'has property'(X)")
        self.assertEqual(f, Predicate("has property", [Variable("X")]))

    def test_complex_function_names(self):
        f = parse("human('best friend'(X))")
        expected = Predicate(
            "human", [FunctionApplication("best friend", [Variable("X")])]
        )
        self.assertEqual(f, expected)

    def test_complex_monadic_predicate(self):
        f = parse("$'neural network probability'(X)")
        self.assertEqual(
            f, MonadicPredicate("neural network probability", [Variable("X")])
        )

    def test_complex_computation(self):
        f = parse("Result := $'deep learning model'(Input) (valid(Result))")
        expected = Computation(
            "Result",
            "deep learning model",
            [Variable("Input")],
            Predicate("valid", [Variable("Result")]),
        )
        self.assertEqual(f, expected)

    # Test numeric function symbols
    def test_numeric_constant(self):
        f = parse("human(42)")
        expected = Predicate("human", [FunctionApplication("42", [])])
        self.assertEqual(f, expected)

    def test_numeric_function(self):
        f = parse("human(42(X))")
        expected = Predicate("human", [FunctionApplication("42", [Variable("X")])])
        self.assertEqual(f, expected)

    def test_numeric_function_multiple_args(self):
        f = parse("process(123(X, Y))")
        expected = Predicate(
            "process", [FunctionApplication("123", [Variable("X"), Variable("Y")])]
        )
        self.assertEqual(f, expected)

    def test_mixed_numeric_and_text_args(self):
        f = parse("relation(john, 456, mary)")
        expected = Predicate(
            "relation",
            [
                FunctionApplication("john", []),
                FunctionApplication("456", []),
                FunctionApplication("mary", []),
            ],
        )
        self.assertEqual(f, expected)

    # Test error handling and edge cases
    def test_empty_predicate_args(self):
        f = parse("nullary()")
        self.assertEqual(f, Predicate("nullary", []))

    def test_deeply_nested_parentheses(self):
        f = parse("((((T))))")
        self.assertEqual(f, TrueFormula())

    def test_mixed_operators(self):
        # Test: forall X (p(X) -> (q(X) and not r(X)))
        f = parse("forall X (p(X) -> q(X) and not r(X))")
        expected = UniversalQuantification(
            "X",
            Implication(
                Predicate("p", [Variable("X")]),
                Conjunction(
                    Predicate("q", [Variable("X")]),
                    Negation(Predicate("r", [Variable("X")])),
                ),
            ),
        )
        self.assertEqual(f, expected)

    def test_computation_with_multiple_args(self):
        f = parse("Result := $combine(X, Y, Z) (success(Result))")
        expected = Computation(
            "Result",
            "combine",
            [Variable("X"), Variable("Y"), Variable("Z")],
            Predicate("success", [Variable("Result")]),
        )
        self.assertEqual(f, expected)

    # Negative tests - verify grammar rejects invalid patterns
    def test_reject_lowercase_variable(self):
        # Variables must start with uppercase, lowercase should be parsed as constants
        f = parse(
            "human(person)"
        )  # 'person' should be parsed as constant, not variable
        expected = Predicate("human", [FunctionApplication("person", [])])
        self.assertEqual(f, expected)

    def test_reject_uppercase_function(self):
        # Functions must start with lowercase, uppercase should fail
        with self.assertRaises(LarkError):
            parse("human(Father(X))")  # 'Father' should be 'father'

    def test_reject_uppercase_predicate(self):
        # Predicates must start with lowercase, uppercase should fail
        with self.assertRaises(LarkError):
            parse("Human(X)")  # 'Human' should be 'human'

    def test_reject_monadic_predicate_without_dollar(self):
        # Monadic predicates in the context require $ prefix
        # Regular predicates with the same name should parse as regular predicates
        f = parse("neural(X)")
        # This should parse as a regular predicate, not a monadic one
        self.assertIsInstance(f, Predicate)
        self.assertNotIsInstance(f, MonadicPredicate)

    def test_reject_monadic_function_without_dollar(self):
        # Monadic functions in computation require $ prefix
        with self.assertRaises(LarkError):
            parse("X := compute(Y) (valid(X))")  # 'compute' should be '$compute'

    def test_reject_invalid_variable_pattern(self):
        # Variables can't start with numbers or special characters
        with self.assertRaises(LarkError):
            parse("human(1Person)")
        with self.assertRaises(LarkError):
            parse("human(_Person)")

    def test_reject_invalid_function_pattern(self):
        # Functions can't start with numbers or special characters
        with self.assertRaises(LarkError):
            parse("human(1father(X))")
        with self.assertRaises(LarkError):
            parse("human(_father(X))")

    def test_reject_invalid_predicate_pattern(self):
        # Predicates can't start with numbers or special characters
        with self.assertRaises(LarkError):
            parse("1human(X)")
        with self.assertRaises(LarkError):
            parse("_human(X)")

    def test_reject_malformed_monadic_identifiers(self):
        # Monadic identifiers must have proper $ prefix format
        with self.assertRaises(LarkError):
            parse("$1neural(X)")  # Can't start with number after $
        with self.assertRaises(LarkError):
            parse("$$neural(X)")  # Double $ not allowed
        with self.assertRaises(LarkError):
            parse("$Neural(X)")  # Must be lowercase after $

    def test_reject_empty_identifiers(self):
        # Empty identifiers should be rejected
        with self.assertRaises(LarkError):
            parse("(X)")  # Missing predicate name
        with self.assertRaises(LarkError):
            parse("human((X))")  # Empty function name

    def test_reject_mismatched_parentheses(self):
        # Mismatched parentheses should be rejected
        with self.assertRaises(LarkError):
            parse("human(X")  # Missing closing parenthesis
        with self.assertRaises(LarkError):
            parse("human X)")  # Missing opening parenthesis
        with self.assertRaises(LarkError):
            parse("human((X)")  # Mismatched nested parentheses

    def test_reject_invalid_operator_combinations(self):
        # Invalid operator combinations should be rejected
        with self.assertRaises(LarkError):
            parse("human(X) and and mortal(X)")  # Double 'and'
        with self.assertRaises(LarkError):
            parse("human(X) -> -> mortal(X)")  # Double '->'
        # Triple negation is actually valid, so we'll test something else
        with self.assertRaises(LarkError):
            parse("human(X) or or mortal(X)")  # Double 'or'

    def test_reject_invalid_quantifier_usage(self):
        # Quantifiers must be followed by variable then formula
        with self.assertRaises(LarkError):
            parse("forall human(X)")  # Missing variable after forall
        with self.assertRaises(LarkError):
            parse("exists mortal(X)")  # Missing variable after exists
        with self.assertRaises(LarkError):
            parse("forall x")  # Missing formula after variable

    def test_reject_invalid_computation_syntax(self):
        # Computation must have proper := syntax
        with self.assertRaises(LarkError):
            parse("X = $compute(Y) (valid(X))")  # Should be := not =
        with self.assertRaises(LarkError):
            parse("X := (valid(X))")  # Missing function call
        with self.assertRaises(LarkError):
            parse("X := $compute(Y) valid(X)")  # Missing parentheses around formula

    def test_reject_invalid_property_access(self):
        # Property access must follow proper syntax
        with self.assertRaises(LarkError):
            parse("human(.name)")  # Missing object before .
        with self.assertRaises(LarkError):
            parse("human(Person.)")  # Missing property after .
        with self.assertRaises(LarkError):
            parse("human(Person..name)")  # Double dots not allowed

    def test_reject_mixed_case_keywords(self):
        # Keywords must be exact case
        with self.assertRaises(LarkError):
            parse("NOT T")  # Should be 'not'
        with self.assertRaises(LarkError):
            parse("AND T F")  # Should be 'and'
        with self.assertRaises(LarkError):
            parse("FORALL X human(X)")  # Should be 'forall'
        with self.assertRaises(LarkError):
            parse("EXISTS X human(X)")  # Should be 'exists'

    def test_traffic_light_example1(self):
        # Test a more complex example with traffic light logic
        f = parse("L := $light()(D := $driveF(L) (eval(D) -> equals(L, green)))")

        expected = Computation(
            "L",
            "light",
            [],
            Computation(
                "D",
                "driveF",
                [Variable("L")],
                Implication(
                    Predicate("eval", [Variable("D")]),
                    Predicate(
                        "equals", [Variable("L"), FunctionApplication("green", [])]
                    ),
                ),
            ),
        )
        self.assertEqual(f, expected)

    def test_traffic_light_example2(self):
        # Test a more complex example with traffic light logic
        f = parse("L := $light() ($driveP(L) -> equals(L, green))")

        expected = Computation(
            "L",
            "light",
            [],
            Implication(
                MonadicPredicate("driveP", [Variable("L")]),
                Predicate("equals", [Variable("L"), FunctionApplication("green", [])]),
            ),
        )
        self.assertEqual(f, expected)

    # Test syntactic sugar for consequent computations
    def test_computation_sequence_formal(self):
        """Test the formal syntax with nested parentheses."""
        f = parse("X := $a()(Y := $b(X)(p(X, Y)))")
        expected = Computation(
            "X", "a", [],
            Computation(
                "Y", "b", [Variable("X")],
                Predicate("p", [Variable("X"), Variable("Y")])
            )
        )
        self.assertEqual(f, expected)

    def test_computation_sequence_syntactic_sugar(self):
        """Test the syntactic sugar form with comma separation."""
        f = parse("X := $a(), Y := $b(X)(p(X, Y))")
        expected = Computation(
            "X", "a", [],
            Computation(
                "Y", "b", [Variable("X")],
                Predicate("p", [Variable("X"), Variable("Y")])
            )
        )
        self.assertEqual(f, expected)

    def test_computation_sequence_equivalence(self):
        """Test that syntactic sugar and formal forms are equivalent."""
        formal = parse("X := $a()(Y := $b(X)(p(X, Y)))")
        sugar = parse("X := $a(), Y := $b(X)(p(X, Y))")
        self.assertEqual(formal, sugar)

    def test_computation_sequence_three_computations(self):
        """Test three consecutive computations in formal syntax."""
        f = parse("X := $a()(Y := $b(X)(Z := $c(X, Y)(p(X, Y, Z))))")
        expected = Computation(
            "X", "a", [],
            Computation(
                "Y", "b", [Variable("X")],
                Computation(
                    "Z", "c", [Variable("X"), Variable("Y")],
                    Predicate("p", [Variable("X"), Variable("Y"), Variable("Z")])
                )
            )
        )
        self.assertEqual(f, expected)

    def test_computation_sequence_with_complex_formula(self):
        """Test formal syntax with complex logical formula."""
        f = parse("X := $a()(Y := $b(X)(p(X) and q(Y)))")
        expected = Computation(
            "X", "a", [],
            Computation(
                "Y", "b", [Variable("X")],
                Conjunction(
                    Predicate("p", [Variable("X")]),
                    Predicate("q", [Variable("Y")])
                )
            )
        )
        self.assertEqual(f, expected)

    def test_computation_sequence_three_computations_sugar(self):
        """Test three consecutive computations in syntactic sugar form."""
        f = parse("X := $a(), Y := $b(X), Z := $c(X, Y)(p(X, Y, Z))")
        expected = Computation(
            "X", "a", [],
            Computation(
                "Y", "b", [Variable("X")],
                Computation(
                    "Z", "c", [Variable("X"), Variable("Y")],
                    Predicate("p", [Variable("X"), Variable("Y"), Variable("Z")])
                )
            )
        )
        self.assertEqual(f, expected)


if __name__ == "__main__":
    unittest.main()

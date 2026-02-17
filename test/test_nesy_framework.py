import unittest
from typing import Literal, Mapping

from common import nf as prob_nf, traffic_light_model

from muller.hkt import List
from muller.monad.distribution import Prob, uniform
from muller.monad.non_empty_powerset import NonEmptyPowerset, from_list, singleton
from muller.nesy_framework import (
    Interpretation,
    NeSyFramework,
    nesy,
)
from muller.parser import (
    Computation,
    Conjunction,
    Disjunction,
    ExistentialQuantification,
    FalseFormula,
    FunctionApplication,
    MonadicPredicate,
    Negation,
    Predicate,
    TrueFormula,
    UniversalQuantification,
    Variable,
    parse,
)

Universe = Literal["alice", "bob", "charlie"]


class TestNeSyFramework(unittest.TestCase):
    """Test suite for different NeSy systems."""

    universe: list[Universe]

    def setUp(self):
        """Set up common test data."""
        self.universe = ["alice", "bob", "charlie"]

        # Probabilistic NeSy system
        self.prob_nesy = nesy(Prob, bool, List)

        # Non-deterministic NeSy system
        self.nondet_nesy = nesy(NonEmptyPowerset, bool, List)

    def test_probabilistic_basic_formulas(self):
        """Test basic formulas in probabilistic logic."""
        interpretation = self.prob_nesy.create_interpretation(
            sort=List(self.universe),
            functions={
                "alice": lambda _args: "alice",
                "bob": lambda _args: "bob",
            },
            predicates={
                "human": lambda args: args[0]
                in ["alice", "bob", "charlie"],
                "likes": lambda args: (args[0], args[1])
                in [("alice", "bob"), ("bob", "alice")],
            },
        )

        valuation: Mapping[str, Universe] = {"X": "alice", "Y": "bob"}

        # Test True formula
        true_formula = TrueFormula()
        result = self.prob_nesy.eval(true_formula, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertEqual(result._inner_value[True], 1.0)

        # Test False formula
        false_formula = FalseFormula()
        result = self.prob_nesy.eval(
            false_formula, interpretation, valuation
        )
        self.assertIsInstance(result, Prob)
        self.assertEqual(result._inner_value[False], 1.0)

        # Test predicate
        human_formula = Predicate("human", [Variable("X")])
        result = self.prob_nesy.eval(
            human_formula, interpretation, valuation
        )
        self.assertIsInstance(result, Prob)
        self.assertEqual(
            result._inner_value[True], 1.0
        )  # alice is human

    def test_probabilistic_complex_formulas(self):
        """Test complex formulas in probabilistic logic."""
        interpretation = self.prob_nesy.create_interpretation(
            sort=List(self.universe),
            functions={
                "father": lambda args: "bob"
                if args[0] == "alice"
                else "charlie",
            },
            predicates={
                "parent": lambda args: (args[0], args[1])
                in [("bob", "alice"), ("charlie", "bob")],
                "male": lambda args: args[0] in ["bob", "charlie"],
            },
        )

        valuation: Mapping[str, Universe] = {"X": "alice", "Y": "bob"}

        # Test conjunction: parent(Y, X) and male(Y)
        conj_formula = Conjunction(
            Predicate("parent", [Variable("Y"), Variable("X")]),
            Predicate("male", [Variable("Y")]),
        )
        result = self.prob_nesy.eval(
            conj_formula, interpretation, valuation
        )
        self.assertIsInstance(result, Prob)
        self.assertEqual(
            result._inner_value[True], 1.0
        )  # Both parent(bob, alice) and male(bob) are true

        # Test negation
        not_male = Negation(Predicate("male", [Variable("X")]))
        result = self.prob_nesy.eval(not_male, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertEqual(
            result._inner_value[True], 1.0
        )  # alice is not male

    def test_probabilistic_quantification(self):
        """Test quantification in probabilistic logic."""
        interpretation = self.prob_nesy.create_interpretation(
            sort=List(["alice", "bob"]),
            functions={
                "alice": lambda _args: "alice",
            },
            predicates={
                "human": lambda _args: True,  # Everyone is human
                "likes": lambda args: args[0]
                == "alice",  # Only alice likes anyone
            },
        )

        valuation: Mapping[str, str] = {}

        # Test universal quantification: forall X human(X)
        forall_formula = UniversalQuantification(
            "X", Predicate("human", [Variable("X")])
        )
        result = self.prob_nesy.eval(
            forall_formula, interpretation, valuation
        )
        self.assertIsInstance(result, Prob)
        self.assertEqual(result._inner_value[True], 1.0)  # All are human

        # Test existential quantification: exists X likes(X alice)
        exists_formula = ExistentialQuantification(
            "X",
            Predicate(
                "likes",
                [Variable("X"), FunctionApplication("alice", [])],
            ),
        )
        result = self.prob_nesy.eval(
            exists_formula, interpretation, valuation
        )
        self.assertIsInstance(result, Prob)
        self.assertEqual(
            result._inner_value[True], 1.0
        )  # alice likes alice

    def test_probabilistic_monadic_predicates(self):
        """Test monadic predicates in probabilistic system."""
        interpretation = self.prob_nesy.create_interpretation(
            sort=List(["alice", "bob"]),
            mfunctions={
                "coin_flip": lambda _args: Prob(
                    {"alice": 0.5, "bob": 0.5}
                ),
            },
            mpredicates={
                "probably_tall": lambda args: (
                    Prob({True: 0.8, False: 0.2})
                    if args[0] == "alice"
                    else Prob({True: 0.3, False: 0.7})
                )
            },
        )

        valuation = {"X": "alice"}

        # Test monadic predicate
        prob_formula = MonadicPredicate(
            "probably_tall", [Variable("X")]
        )
        result = self.prob_nesy.eval(
            prob_formula, interpretation, valuation
        )
        self.assertIsInstance(result, Prob)
        self.assertAlmostEqual(
            result._inner_value[True], 0.8, places=2
        )

        # Test computation
        comp_formula = Computation(
            "Y",
            "coin_flip",
            [],
            MonadicPredicate("probably_tall", [Variable("Y")]),
        )
        result = self.prob_nesy.eval(
            comp_formula, interpretation, valuation
        )
        self.assertIsInstance(result, Prob)
        # Expected: 0.5 * 0.8 + 0.5 * 0.3 = 0.55
        expected_true_prob = 0.5 * 0.8 + 0.5 * 0.3
        self.assertAlmostEqual(
            result._inner_value[True], expected_true_prob, places=2
        )

    def test_nondeterministic_system(self):
        """Test non-deterministic NeSy system."""
        interpretation = self.nondet_nesy.create_interpretation(
            sort=List(["alice", "bob"]),
            mfunctions={
                "choose_person": lambda _args: from_list(
                    ["alice", "bob"]
                ),
            },
            mpredicates={
                "might_be_tall": lambda args: (
                    from_list([True, False])
                    if args[0] == "alice"
                    else singleton(False)
                )
            },
        )

        valuation = {"X": "alice"}

        # Test monadic predicate
        nondet_formula = MonadicPredicate(
            "might_be_tall", [Variable("X")]
        )
        result = self.nondet_nesy.eval(
            nondet_formula, interpretation, valuation
        )
        self.assertIsInstance(result, NonEmptyPowerset)
        self.assertEqual(
            result._inner_value, frozenset([True, False])
        )

        # Test computation
        comp_formula = Computation(
            "Y",
            "choose_person",
            [],
            MonadicPredicate("might_be_tall", [Variable("Y")]),
        )
        result = self.nondet_nesy.eval(
            comp_formula, interpretation, valuation
        )
        self.assertIsInstance(result, NonEmptyPowerset)
        # alice can be True or False, bob can only be False
        self.assertEqual(
            result._inner_value, frozenset([True, False])
        )

    def test_parser_integration_probabilistic(self):
        """Test parser integration with probabilistic NeSy system."""
        interpretation = self.prob_nesy.create_interpretation(
            sort=List(["alice", "bob"]),
            mfunctions={
                "random_person": lambda _args: Prob(
                    {"alice": 0.6, "bob": 0.4}
                ),
            },
            mpredicates={
                "tall": lambda args: (
                    Prob({True: 0.7, False: 0.3})
                    if args[0] == "alice"
                    else Prob({True: 0.4, False: 0.6})
                )
            },
        )

        valuation = {"X": "alice"}

        # Test monadic predicate parsing (using $ prefix)
        formula = parse("$tall(X)")
        result = self.prob_nesy.eval(formula, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertAlmostEqual(
            result._inner_value[True], 0.7, places=2
        )

        # Test computation parsing
        formula = parse("Y := $random_person()($tall(Y))")
        result = self.prob_nesy.eval(formula, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        expected = 0.6 * 0.7 + 0.4 * 0.4  # 0.58
        self.assertAlmostEqual(
            result._inner_value[True], expected, places=2
        )

    def test_parser_integration_classical_style(self):
        """Test parser with classical predicates in probabilistic system."""
        interpretation = self.prob_nesy.create_interpretation(
            sort=List(["alice", "bob", "charlie"]),
            functions={
                "alice": lambda _args: "alice",
                "father": lambda args: "bob"
                if args[0] == "alice"
                else "charlie",
            },
            predicates={
                "human": lambda args: args[0]
                in ["alice", "bob", "charlie"],
                "parent": lambda args: (args[0], args[1])
                in [("bob", "alice"), ("charlie", "bob")],
            },
        )

        valuation = {"X": "alice", "Y": "bob", "Z": "charlie"}

        # Test simple predicate parsing
        formula = parse("human(alice)")
        result = self.prob_nesy.eval(formula, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertEqual(result._inner_value[True], 1.0)

        # Test complex formula parsing
        formula = parse("human(X) and parent(Y, X)")
        result = self.prob_nesy.eval(formula, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertEqual(result._inner_value[True], 1.0)

        # Test implication
        formula = parse("human(X) -> parent(Y, X)")
        result = self.prob_nesy.eval(formula, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertEqual(result._inner_value[True], 1.0)

    def test_parser_integration_quantifiers(self):
        """Test parser integration with quantifiers."""
        interpretation = self.prob_nesy.create_interpretation(
            sort=List(["alice", "bob"]),
            predicates={
                "human": lambda _args: True,
                "mortal": lambda _args: True,
            },
        )

        valuation: Mapping[str, str] = {}

        # Test universal quantification
        formula = parse("![X]: (human(X) -> mortal(X))")
        result = self.prob_nesy.eval(formula, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertEqual(result._inner_value[True], 1.0)

        # Test existential quantification
        formula = parse("?[X]: human(X)")
        result = self.prob_nesy.eval(formula, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertEqual(result._inner_value[True], 1.0)

    def test_function_applications(self):
        """Test function applications in terms."""
        interpretation = self.prob_nesy.create_interpretation(
            sort=List(["alice", "bob", "charlie"]),
            functions={
                "father": lambda args: {
                    "alice": "bob",
                    "bob": "charlie",
                }.get(args[0], "alice"),
                "mother": lambda args: {
                    "alice": "charlie",
                    "bob": "alice",
                }.get(args[0], "bob"),
            },
            predicates={
                "human": lambda args: args[0]
                in ["alice", "bob", "charlie"],
            },
        )

        valuation = {"X": "alice"}

        # Test function application in predicate
        formula = parse("human(father(X))")
        result = self.prob_nesy.eval(formula, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertEqual(
            result._inner_value[True], 1.0
        )  # human(father(alice)) = human(bob) = True

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        interpretation = self.prob_nesy.create_interpretation(
            sort=List([]),  # Empty universe
        )

        valuation: Mapping[str, str] = {}

        # Test with empty universe - quantifiers should handle gracefully
        forall_formula = UniversalQuantification("X", TrueFormula())
        result = self.prob_nesy.eval(
            forall_formula, interpretation, valuation
        )
        self.assertIsInstance(result, Prob)
        self.assertEqual(
            result._inner_value[True], 1.0
        )  # Vacuously true

        exists_formula = ExistentialQuantification("X", TrueFormula())
        result = self.prob_nesy.eval(
            exists_formula, interpretation, valuation
        )
        self.assertIsInstance(result, Prob)
        self.assertEqual(
            result._inner_value[False], 1.0
        )  # No elements to satisfy

    def test_mixed_probabilistic_computations(self):
        """Test mixed deterministic and probabilistic computations."""
        interpretation = self.prob_nesy.create_interpretation(
            sort=List(["good", "bad"]),
            functions={
                "outcome": lambda _args: "good",
            },  # Deterministic function
            mfunctions={
                "random_outcome": lambda _args: Prob(
                    {"good": 0.8, "bad": 0.2}
                ),
            },
            predicates={
                "positive": lambda args: args[0] == "good",
            },
            mpredicates={
                "likely_positive": lambda args: (
                    Prob({True: 0.9, False: 0.1})
                    if args[0] == "good"
                    else Prob({True: 0.1, False: 0.9})
                )
            },
        )

        valuation: Mapping[str, str] = {}

        # Test deterministic function with probabilistic predicate
        formula = parse("positive(outcome)")
        result = self.prob_nesy.eval(formula, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertEqual(result._inner_value[True], 1.0)

        # Test probabilistic computation with deterministic predicate
        comp_formula = Computation(
            "X",
            "random_outcome",
            [],
            Predicate("positive", [Variable("X")]),
        )
        result = self.prob_nesy.eval(
            comp_formula, interpretation, valuation
        )
        self.assertIsInstance(result, Prob)
        # Expected: 0.8 * 1.0 + 0.2 * 0.0 = 0.8
        self.assertAlmostEqual(
            result._inner_value[True], 0.8, places=2
        )

    def test_nested_computations(self):
        """Test nested computations and complex monadic operations."""
        interpretation = self.prob_nesy.create_interpretation(
            sort=List(["low", "medium", "high"]),
            mfunctions={
                "confidence_level": lambda _args: Prob(
                    {"low": 0.3, "medium": 0.5, "high": 0.2}
                ),
                "adjust_confidence": lambda args: (
                    Prob(
                        {"low": 0.8, "medium": 0.15, "high": 0.05}
                    )
                    if args[0] == "low"
                    else (
                        Prob(
                            {"low": 0.1, "medium": 0.3, "high": 0.6}
                        )
                        if args[0] == "medium"
                        else Prob(
                            {"low": 0.05, "medium": 0.25, "high": 0.7}
                        )
                    )
                ),
            },
            mpredicates={
                "high_confidence": lambda args: (
                    Prob({True: 0.9, False: 0.1})
                    if args[0] == "high"
                    else (
                        Prob({True: 0.5, False: 0.5})
                        if args[0] == "medium"
                        else Prob({True: 0.1, False: 0.9})
                    )
                )
            },
        )

        valuation: Mapping[str, str] = {}

        # Test sequential computations:
        # X := confidence_level(),
        # Y := adjust_confidence(X), high_confidence(Y)
        nested_formula = Computation(
            "X",
            "confidence_level",
            [],
            Computation(
                "Y",
                "adjust_confidence",
                [Variable("X")],
                MonadicPredicate(
                    "high_confidence", [Variable("Y")]
                ),
            ),
        )

        result = self.prob_nesy.eval(
            nested_formula, interpretation, valuation
        )
        self.assertIsInstance(result, Prob)
        # This should compute the probability through all
        # the nested transformations
        self.assertTrue(0.0 <= result._inner_value[True] <= 1.0)
        self.assertTrue(0.0 <= result._inner_value[False] <= 1.0)
        self.assertAlmostEqual(
            result._inner_value[True]
            + result._inner_value[False],
            1.0,
            places=6,
        )

    def test_non_deterministic_logical_operations(self):
        """Test logical operations in non-deterministic system."""
        interpretation = self.nondet_nesy.create_interpretation(
            sort=List(["alice", "bob"]),
            mpredicates={
                "maybe_smart": lambda _args: from_list(
                    [True, False]
                ),
                "maybe_kind": lambda args: (
                    from_list([True])
                    if args[0] == "alice"
                    else from_list([False])
                ),
            },
        )

        valuation = {"X": "alice", "Y": "bob"}

        # Test conjunction of non-deterministic predicates
        conj_formula = Conjunction(
            MonadicPredicate("maybe_smart", [Variable("X")]),
            MonadicPredicate("maybe_kind", [Variable("X")]),
        )
        result = self.nondet_nesy.eval(
            conj_formula, interpretation, valuation
        )
        self.assertIsInstance(result, NonEmptyPowerset)
        # alice: {T,F} ∧ {T} = {T∧T, F∧T} = {T, F}
        self.assertEqual(
            result._inner_value, frozenset([True, False])
        )

        # Test disjunction
        disj_formula = Disjunction(
            MonadicPredicate("maybe_smart", [Variable("Y")]),
            MonadicPredicate("maybe_kind", [Variable("Y")]),
        )
        result = self.nondet_nesy.eval(
            disj_formula, interpretation, valuation
        )
        self.assertIsInstance(result, NonEmptyPowerset)
        # bob: {T,F} ∨ {F} = {T∨F, F∨F} = {T, F}
        self.assertEqual(
            result._inner_value, frozenset([True, False])
        )

    def test_dice_example2(self):
        """Test a dice-style example similar to the Haskell code."""
        interpretation = self.prob_nesy.create_interpretation(
            sort=List(list(range(7))),
            functions={
                str(i): lambda _args, i=i: i for i in range(1, 7)
            },
            mfunctions={
                "die": lambda _args: uniform(list(range(1, 7))),
            },
            predicates={
                "equals": lambda args: args[0] == args[1],
                "even": lambda args: args[0] % 2 == 0,
            },
        )

        valuation: Mapping[str, int] = {}

        dice_formula = parse(
            "X := $die() (equals(X, 6) and even(X))"
        )

        result = self.prob_nesy.eval(
            dice_formula, interpretation, valuation
        )
        self.assertIsInstance(result, Prob)
        # P(X == 6 ∧ even(X))  = 1/6 * 1 = 1/6
        expected_prob = 1 / 6
        self.assertAlmostEqual(
            result._inner_value[True], expected_prob, places=6
        )

    def test_dice_example_style(self):
        """Test a dice-style example similar to the Haskell code."""
        interpretation = self.prob_nesy.create_interpretation(
            sort=List(list(range(7))),
            functions={
                str(i): lambda _args, i=i: i for i in range(1, 7)
            },
            mfunctions={
                "die": lambda _args: uniform(list(range(1, 7))),
            },
            predicates={
                "==": lambda args: args[0] == args[1],
                "even": lambda args: args[0] % 2 == 0,
            },
        )

        valuation: Mapping[str, int] = {}

        dice_formula = parse(
            "X := $die() (X == 6) & X := $die() (even(X))"
        )

        result = self.prob_nesy.eval(
            dice_formula, interpretation, valuation
        )
        self.assertIsInstance(result, Prob)
        # P(X == 6) * P(even(X)) = 1/6 * 3/6 = 1/12
        expected_prob = 1 / 12
        self.assertAlmostEqual(
            result._inner_value[True], expected_prob, places=6
        )

    def test_traffic_light_example1(self):
        """Test a traffic light example."""

        valuation: Mapping[str, str] = {}

        # Test monadic predicate for traffic light
        light_formula = parse(
            "L := $light()"
            "(D := $driveF(L) (eval(D) -> equals(L, green)))"
        )
        result = prob_nf.eval(
            light_formula, traffic_light_model, valuation
        )
        self.assertIsInstance(result, Prob)
        self.assertAlmostEqual(
            result._inner_value[True], 0.92, places=5
        )

    def test_traffic_light_example2(self):
        """Test a traffic light example."""
        valuation: Mapping[str, str] = {}

        # Test monadic predicate for traffic light
        light_formula = parse(
            "L := $light() ($driveP(L) -> equals(L, green))"
        )
        result = prob_nf.eval(
            light_formula, traffic_light_model, valuation
        )
        self.assertIsInstance(result, Prob)
        self.assertAlmostEqual(
            result._inner_value[True], 0.92, places=5
        )


if __name__ == "__main__":
    unittest.main()

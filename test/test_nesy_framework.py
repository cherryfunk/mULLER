import unittest
from typing import Callable, Literal, Mapping

from common import traffic_light_model

from muller.logics.aggr2sgrpblat import Aggr2SGrpBLat, NeSyLogicMeta
from muller.logics.boolean import (
    NonDeterministicBooleanLogic,
    NonDeterministicBooleanLogicList,
)
from muller.monad.base import ParametrizedMonad
from muller.monad.distribution import Prob, uniform
from muller.monad.non_empty_powerset import NonEmptyPowerset, from_list, singleton
from muller.nesy_framework import (
    Computation,
    Conjunction,
    Disjunction,
    ExistentialQuantification,
    FalseFormula,
    FunctionApplication,
    Interpretation,
    MonadicPredicate,
    Negation,
    NeSyFramework,
    Predicate,
    TrueFormula,
    UniversalQuantification,
    Variable,
    nesy,
)
from muller.parser import parse

Universe = Literal["alice", "bob", "charlie"]
class TestNeSyFramework(unittest.TestCase):
    """Test suite for different NeSy systems."""

    universe: list[Universe]

    def setUp(self):
        """Set up common test data."""
        # Simple universe for testing
        self.universe = ["alice", "bob", "charlie"]

        # Probabilistic NeSy system
        self.prob_nesy = nesy(Prob[bool], bool)

        # Non-deterministic NeSy system
        self.nondet_nesy = nesy(NonEmptyPowerset[bool], bool)

    def test_probabilistic_basic_formulas(self):
        """Test basic formulas in probabilistic logic."""
        interpretation = Interpretation(
            universe=self.universe,
            functions={"alice": lambda: "alice", "bob": lambda: "bob"},
            mfunctions={},
            preds={
                "human": lambda x: x in ["alice", "bob", "charlie"],
                "likes": lambda x, y: (x, y) in [("alice", "bob"), ("bob", "alice")],
            },
            mpreds={},
        )

        valuation: Mapping[str, Universe] = {"X": "alice", "Y": "bob"}

        # Test True formula
        true_formula = TrueFormula()
        result = true_formula.eval(self.prob_nesy, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertEqual(result.value[True], 1.0)

        # Test False formula
        false_formula = FalseFormula()
        result = false_formula.eval(self.prob_nesy, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertEqual(result.value[False], 1.0)

        # Test predicate
        human_formula = Predicate("human", [Variable("X")])
        result = human_formula.eval(self.prob_nesy, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertEqual(result.value[True], 1.0)  # alice is human

    def test_probabilistic_complex_formulas(self):
        """Test complex formulas in probabilistic logic."""
        interpretation: Interpretation[Universe, bool, list[Universe]] = Interpretation(
            universe=self.universe,
            functions={
                "father": lambda x: "bob" if x == "alice" else "charlie",
            },
            mfunctions={},
            preds={
                "parent": lambda x, y: (x, y) in [("bob", "alice"), ("charlie", "bob")],
                "male": lambda x: x in ["bob", "charlie"],
            },
            mpreds={},
        )

        valuation: Mapping[str, Universe] = {"X": "alice", "Y": "bob"}

        # Test conjunction: parent(Y, X) and male(Y)
        conj_formula = Conjunction(
            Predicate("parent", [Variable("Y"), Variable("X")]),
            Predicate("male", [Variable("Y")]),
        )
        result = conj_formula.eval(self.prob_nesy, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertEqual(
            result.value[True], 1.0
        )  # Both parent(bob, alice) and male(bob) are true

        # Test negation
        not_male = Negation(Predicate("male", [Variable("X")]))
        result = not_male.eval(self.prob_nesy, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertEqual(result.value[True], 1.0)  # alice is not male

    def test_probabilistic_quantification(self):
        """Test quantification in probabilistic logic."""
        interpretation = Interpretation(
            universe=["alice", "bob"],
            functions={"alice": lambda: "alice"},  # Add alice as a constant function
            mfunctions={},
            preds={
                "human": lambda x: True,  # Everyone is human
                "likes": lambda x, y: x == "alice",  # Only alice likes anyone
            },
            mpreds={},
        )

        valuation = {}

        # Test universal quantification: forall X human(X)
        forall_formula = UniversalQuantification(
            "X", Predicate("human", [Variable("X")])
        )
        result = forall_formula.eval(self.prob_nesy, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertEqual(result.value[True], 1.0)  # All are human

        # Test existential quantification: exists X likes(X alice)
        exists_formula = ExistentialQuantification(
            "X", Predicate("likes", [Variable("X"), FunctionApplication("alice", [])])
        )
        result = exists_formula.eval(self.prob_nesy, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertEqual(result.value[True], 1.0)  # alice likes alice

    def test_probabilistic_monadic_predicates(self):
        """Test monadic predicates in probabilistic system."""
        interpretation = Interpretation(
            universe=["alice", "bob"],
            functions={},
            mfunctions={"coin_flip": lambda: Prob({"alice": 0.5, "bob": 0.5})},
            preds={},
            mpreds={
                "probably_tall": lambda x: (
                    Prob({True: 0.8, False: 0.2})
                    if x == "alice"
                    else Prob({True: 0.3, False: 0.7})
                )
            },
        )

        valuation = {"X": "alice"}

        # Test monadic predicate
        prob_formula = MonadicPredicate("probably_tall", [Variable("X")])
        result = prob_formula.eval(self.prob_nesy, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertAlmostEqual(result.value[True], 0.8, places=2)

        # Test computation
        comp_formula = Computation(
            "Y", "coin_flip", [], MonadicPredicate("probably_tall", [Variable("Y")])
        )
        result = comp_formula.eval(self.prob_nesy, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        # Expected: 0.5 * 0.8 + 0.5 * 0.3 = 0.55
        expected_true_prob = 0.5 * 0.8 + 0.5 * 0.3
        self.assertAlmostEqual(result.value[True], expected_true_prob, places=2)

    def test_nondeterministic_system(self):
        """Test non-deterministic NeSy system."""
        interpretation = Interpretation(
            universe=["alice", "bob"],
            functions={},
            mfunctions={"choose_person": lambda: from_list(["alice", "bob"])},
            preds={},
            mpreds={
                "might_be_tall": lambda x: (
                    from_list([True, False]) if x == "alice" else singleton(False)
                )
            },
        )

        valuation = {"X": "alice"}

        # Test monadic predicate
        nondet_formula = MonadicPredicate("might_be_tall", [Variable("X")])
        result = nondet_formula.eval(self.nondet_nesy, interpretation, valuation)
        self.assertIsInstance(result, NonEmptyPowerset)
        self.assertEqual(result.value, frozenset([True, False]))

        # Test computation
        comp_formula = Computation(
            "Y", "choose_person", [], MonadicPredicate("might_be_tall", [Variable("Y")])
        )
        result = comp_formula.eval(self.nondet_nesy, interpretation, valuation)
        self.assertIsInstance(result, NonEmptyPowerset)
        # alice can be True or False, bob can only be False
        self.assertEqual(result.value, frozenset([True, False]))

    def test_parser_integration_probabilistic(self):
        """Test parser integration with probabilistic NeSy system."""
        interpretation = Interpretation(
            universe=["alice", "bob"],
            functions={},
            mfunctions={"random_person": lambda: Prob({"alice": 0.6, "bob": 0.4})},
            preds={},
            mpreds={
                "tall": lambda x: (
                    Prob({True: 0.7, False: 0.3})
                    if x == "alice"
                    else Prob({True: 0.4, False: 0.6})
                )
            },
        )

        valuation = {"X": "alice"}

        # Test monadic predicate parsing (using $ prefix)
        formula = parse("$tall(X)")
        result = formula.eval(self.prob_nesy, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertAlmostEqual(result.value[True], 0.7, places=2)

        # Test computation parsing
        formula = parse("Y := $random_person()($tall(Y))")
        result = formula.eval(self.prob_nesy, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        expected = 0.6 * 0.7 + 0.4 * 0.4  # 0.58
        self.assertAlmostEqual(result.value[True], expected, places=2)

    def test_parser_integration_classical_style(self):
        """Test parser with classical predicates in probabilistic system."""
        interpretation = Interpretation(
            universe=["alice", "bob", "charlie"],
            functions={
                "alice": lambda: "alice",
                "father": lambda x: "bob" if x == "alice" else "charlie",
            },
            mfunctions={},
            preds={
                "human": lambda x: x in ["alice", "bob", "charlie"],
                "parent": lambda x, y: (x, y) in [("bob", "alice"), ("charlie", "bob")],
            },
            mpreds={},
        )

        valuation = {"X": "alice", "Y": "bob", "Z": "charlie"}

        # Test simple predicate parsing
        formula = parse("human(alice)")
        result = formula.eval(self.prob_nesy, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertEqual(result.value[True], 1.0)

        # Test complex formula parsing
        formula = parse("human(X) and parent(Y, X)")
        result = formula.eval(self.prob_nesy, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertEqual(result.value[True], 1.0)

        # Test implication
        formula = parse("human(X) -> parent(Y, X)")
        result = formula.eval(self.prob_nesy, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertEqual(result.value[True], 1.0)

    def test_parser_integration_quantifiers(self):
        """Test parser integration with quantifiers."""
        interpretation = Interpretation(
            universe=["alice", "bob"],
            functions={},
            mfunctions={},
            preds={"human": lambda x: True, "mortal": lambda x: True},
            mpreds={},
        )

        valuation = {}

        # Test universal quantification
        formula = parse("![X]: (human(X) -> mortal(X))")
        result = formula.eval(self.prob_nesy, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertEqual(result.value[True], 1.0)

        # Test existential quantification
        formula = parse("?[X]: human(X)")
        result = formula.eval(self.prob_nesy, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertEqual(result.value[True], 1.0)

    def test_function_applications(self):
        """Test function applications in terms."""
        interpretation = Interpretation(
            universe=["alice", "bob", "charlie"],
            functions={
                "father": lambda x: {"alice": "bob", "bob": "charlie"}.get(x, "alice"),
                "mother": lambda x: {"alice": "charlie", "bob": "alice"}.get(x, "bob"),
            },
            mfunctions={},
            preds={"human": lambda x: x in ["alice", "bob", "charlie"]},
            mpreds={},
        )

        valuation = {"X": "alice"}

        # Test function application in predicate
        formula = parse("human(father(X))")
        result = formula.eval(self.prob_nesy, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertEqual(
            result.value[True], 1.0
        )  # human(father(alice)) = human(bob) = True

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        interpretation = Interpretation(
            universe=[],  # Empty universe
            functions={},
            mfunctions={},
            preds={},
            mpreds={},
        )

        valuation = {}

        # Test with empty universe - quantifiers should handle gracefully
        forall_formula = UniversalQuantification("X", TrueFormula())
        result = forall_formula.eval(self.prob_nesy, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertEqual(result.value[True], 1.0)  # Vacuously true

        exists_formula = ExistentialQuantification("X", TrueFormula())
        result = exists_formula.eval(self.prob_nesy, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertEqual(result.value[False], 1.0)  # No elements to satisfy

    def test_mixed_probabilistic_computations(self):
        """Test mixed deterministic and probabilistic computations."""
        interpretation = Interpretation(
            universe=["good", "bad"],
            functions={"outcome": lambda: "good"},  # Deterministic function
            mfunctions={"random_outcome": lambda: Prob({"good": 0.8, "bad": 0.2})},
            preds={"positive": lambda x: x == "good"},
            mpreds={
                "likely_positive": lambda x: (
                    Prob({True: 0.9, False: 0.1})
                    if x == "good"
                    else Prob({True: 0.1, False: 0.9})
                )
            },
        )

        valuation = {}

        # Test deterministic function with probabilistic predicate
        formula = parse("positive(outcome)")
        result = formula.eval(self.prob_nesy, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        self.assertEqual(result.value[True], 1.0)

        # Test probabilistic computation with deterministic predicate
        comp_formula = Computation(
            "X", "random_outcome", [], Predicate("positive", [Variable("X")])
        )
        result = comp_formula.eval(self.prob_nesy, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        # Expected: 0.8 * 1.0 + 0.2 * 0.0 = 0.8
        self.assertAlmostEqual(result.value[True], 0.8, places=2)

    def test_nested_computations(self):
        """Test nested computations and complex monadic operations."""
        interpretation = Interpretation(
            universe=["low", "medium", "high"],
            functions={},
            mfunctions={
                "confidence_level": lambda: Prob(
                    {"low": 0.3, "medium": 0.5, "high": 0.2}
                ),
                "adjust_confidence": lambda x: (
                    Prob({"low": 0.8, "medium": 0.15, "high": 0.05})
                    if x == "low"
                    else (
                        Prob({"low": 0.1, "medium": 0.3, "high": 0.6})
                        if x == "medium"
                        else Prob({"low": 0.05, "medium": 0.25, "high": 0.7})
                    )
                ),
            },
            preds={},
            mpreds={
                "high_confidence": lambda x: (
                    Prob({True: 0.9, False: 0.1})
                    if x == "high"
                    else (
                        Prob({True: 0.5, False: 0.5})
                        if x == "medium"
                        else Prob({True: 0.1, False: 0.9})
                    )
                )
            },
        )

        valuation = {}

        # Test sequential computations:
        # X := confidence_level(), Y := adjust_confidence(X), high_confidence(Y)
        nested_formula = Computation(
            "X",
            "confidence_level",
            [],
            Computation(
                "Y",
                "adjust_confidence",
                [Variable("X")],
                MonadicPredicate("high_confidence", [Variable("Y")]),
            ),
        )

        result = nested_formula.eval(self.prob_nesy, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        # This should compute the probability through all the nested transformations
        self.assertTrue(0.0 <= result.value[True] <= 1.0)
        self.assertTrue(0.0 <= result.value[False] <= 1.0)
        self.assertAlmostEqual(result.value[True] + result.value[False], 1.0, places=6)

    def test_non_deterministic_logical_operations(self):
        """Test logical operations in non-deterministic system."""
        interpretation = Interpretation(
            universe=["alice", "bob"],
            functions={},
            mfunctions={},
            preds={},
            mpreds={
                "maybe_smart": lambda x: from_list([True, False]),
                "maybe_kind": lambda x: (
                    from_list([True]) if x == "alice" else from_list([False])
                ),
            },
        )

        valuation = {"X": "alice", "Y": "bob"}

        # Test conjunction of non-deterministic predicates
        conj_formula = Conjunction(
            MonadicPredicate("maybe_smart", [Variable("X")]),
            MonadicPredicate("maybe_kind", [Variable("X")]),
        )
        result = conj_formula.eval(self.nondet_nesy, interpretation, valuation)
        self.assertIsInstance(result, NonEmptyPowerset)
        # alice: {T,F} ∧ {T} = {T∧T, F∧T} = {T, F}
        self.assertEqual(result.value, frozenset([True, False]))

        # Test disjunction
        disj_formula = Disjunction(
            MonadicPredicate("maybe_smart", [Variable("Y")]),
            MonadicPredicate("maybe_kind", [Variable("Y")]),
        )
        result = disj_formula.eval(self.nondet_nesy, interpretation, valuation)
        self.assertIsInstance(result, NonEmptyPowerset)
        # bob: {T,F} ∨ {F} = {T∨F, F∨F} = {T, F}
        self.assertEqual(result.value, frozenset([True, False]))

    def test_dice_example2(self):
        """Test a dice-style example similar to the Haskell code."""
        interpretation = Interpretation(
            universe=list(range(7)),
            functions={str(i): lambda i=i: i for i in range(1, 7)},
            mfunctions={"die": lambda: uniform(list(range(1, 7)))},
            preds={"equals": lambda x, y: x == y, "even": lambda x: x % 2 == 0},
            mpreds={},
        )

        valuation = {}

        dice_formula = parse("X := $die() (equals(X, 6) and even(X))")

        result = dice_formula.eval(self.prob_nesy, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        # P(X == 6 ∧ even(X))  = 1/6 * 1 = 1/6
        expected_prob = 1 / 6
        self.assertAlmostEqual(result.value[True], expected_prob, places=6)

    def test_dice_example_style(self):
        """Test a dice-style example similar to the Haskell code."""
        interpretation = Interpretation(
            universe=list(range(7)),
            functions={str(i): lambda i=i: i for i in range(1, 7)},
            mfunctions={"die": lambda: uniform(list(range(1, 7)))},
            preds={"==": lambda x, y: x == y, "even": lambda x: x % 2 == 0},
            mpreds={},
        )

        valuation = {}

        dice_formula = parse("X := $die() (X == 6) & X := $die() (even(X))")

        result = dice_formula.eval(self.prob_nesy, interpretation, valuation)
        self.assertIsInstance(result, Prob)
        # P(X == 6) * P(even(X)) = P(X == 6) * P(even(X)) = 1/6 * 3/6 = 1/12
        expected_prob = 1 / 12
        self.assertAlmostEqual(result.value[True], expected_prob, places=6)

    def test_traffic_light_example1(self):
        """Test a traffic light example."""

        valuation = {}

        # Test monadic predicate for traffic light
        light_formula = parse(
            "L := $light()(D := $driveF(L) (eval(D) -> equals(L, green)))"
        )
        result = light_formula.eval(self.prob_nesy, traffic_light_model, valuation)
        self.assertIsInstance(result, Prob)
        self.assertAlmostEqual(result.value[True], 0.92, places=5)

    def test_traffic_light_example2(self):
        """Test a traffic light example."""
        valuation = {}

        # Test monadic predicate for traffic light
        light_formula = parse("L := $light() ($driveP(L) -> equals(L, green))")
        result = light_formula.eval(self.prob_nesy, traffic_light_model, valuation)
        self.assertIsInstance(result, Prob)
        self.assertAlmostEqual(result.value[True], 0.92, places=5)

    def test_get_nesy_for_logic(self):
        """Test getting NeSy framework for a specific logic."""
        logic = NonDeterministicBooleanLogicList()
        nesy_framework = nesy(logic)
        self.assertIsInstance(nesy_framework, NeSyFramework)
        self.assertEqual(nesy_framework.M, NonEmptyPowerset)

    def test_get_nesy_for_monad_and_omega(self):
        """Test getting NeSy framework for a specific monad and omega."""

        nesy_framework = nesy(MyMonad, bool)
        self.assertIsInstance(nesy_framework, NeSyFramework)
        self.assertEqual(nesy_framework.M, MyMonad)


class MyMonad[T](ParametrizedMonad[T]): ...


class MyProbabilityLogic(Aggr2SGrpBLat[list, MyMonad[bool]], NeSyLogicMeta[bool]):
    """Custom logic for testing"""

    def top(self) -> MyMonad[bool]:
        return MyMonad(True, None)

    def bottom(self) -> MyMonad[bool]:
        return MyMonad(False, None)

    def conjunction(self, a: MyMonad[bool], b: MyMonad[bool]) -> MyMonad[bool]:
        return MyMonad(a.value and b.value, None)

    def disjunction(self, a: MyMonad[bool], b: MyMonad[bool]) -> MyMonad[bool]:
        return MyMonad(a.value or b.value, None)
    
    def aggrA[A](self, structure: list, f: Callable[[A], MyMonad[bool]]) -> MyMonad[bool]:
        return MyMonad(all(f(x).value for x in structure), None)
    
    def aggrE[A](self, structure: list, f: Callable[[A], MyMonad[bool]]) -> MyMonad[bool]:
        return MyMonad(any(f(x).value for x in structure), None)


if __name__ == "__main__":
    unittest.main()

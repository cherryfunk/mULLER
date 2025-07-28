from muller.monad.distribution import Prob
from muller.monad.non_empty_powerset import NonEmptyPowerset
from muller.nesy_framework import Interpretation, nesy
from muller.parser import parse
from muller.transformation import argmax

from common import traffic_light_model


import unittest


class TestTransformation(unittest.TestCase):
    """Test suite for the argmax transformation."""

    def setUp(self):
        """Set up common test data."""
        self.universe = ["alice", "bob", "charlie"]
        self.prob_nesy = nesy(Prob, bool)
        self.nondet_nesy = nesy(NonEmptyPowerset, bool)

    def test_argmax_basic_functionality(self):
        """Test basic argmax transformation functionality."""
        # Create a probabilistic interpretation with monadic predicates
        prob_interpretation = Interpretation(
            universe=self.universe,
            functions={},
            mfunctions={},
            preds={},
            mpreds={
                "uncertain_pred": lambda x: (
                    Prob({True: 0.7, False: 0.3})
                    if x == "alice"
                    else Prob({True: 0.2, False: 0.8})
                ),
                "tie_pred": lambda x: Prob(
                    {True: 0.5, False: 0.5}
                ),  # Equal probabilities
            },
        )

        # Apply argmax transformation
        nondet_interpretation = prob_interpretation.transform(argmax())

        # Test that the transformation returns a non-deterministic interpretation
        self.assertIsInstance(nondet_interpretation, Interpretation)
        self.assertEqual(nondet_interpretation.universe, self.universe)
        self.assertEqual(nondet_interpretation.functions, {})
        self.assertEqual(nondet_interpretation.preds, {})

        # Test monadic predicates transformation
        self.assertIn("uncertain_pred", nondet_interpretation.mpreds)
        self.assertIn("tie_pred", nondet_interpretation.mpreds)

        # Test argmax on uncertain_pred for alice (should return {True})
        result_alice = nondet_interpretation.mpreds["uncertain_pred"](*["alice"])
        self.assertIsInstance(result_alice, NonEmptyPowerset)
        self.assertEqual(set(result_alice.value), {True})

        # Test argmax on uncertain_pred for bob (should return {False})
        result_bob = nondet_interpretation.mpreds["uncertain_pred"](*["bob"])
        self.assertIsInstance(result_bob, NonEmptyPowerset)
        self.assertEqual(set(result_bob.value), {False})

        # Test argmax on tie_pred (should return {True, False})
        result_tie = nondet_interpretation.mpreds["tie_pred"](*["alice"])
        self.assertIsInstance(result_tie, NonEmptyPowerset)
        self.assertEqual(set(result_tie.value), {True, False})

    def test_argmax_with_monadic_functions(self):
        """Test argmax transformation with monadic functions."""
        # Create interpretation with monadic functions
        prob_interpretation = Interpretation(
            universe=self.universe,
            functions={},
            mfunctions={
                "random_choice": lambda: Prob({"alice": 0.6, "bob": 0.4}),
                "uniform_choice": lambda: Prob(
                    {"alice": 0.33, "bob": 0.33, "charlie": 0.34}
                ),  # charlie has slight edge
            },
            preds={},
            mpreds={},
        )

        # Apply argmax transformation
        argmax_transformer = argmax()
        nondet_interpretation = argmax_transformer(prob_interpretation)

        # Test monadic functions transformation
        self.assertIn("random_choice", nondet_interpretation.mfunctions)
        self.assertIn("uniform_choice", nondet_interpretation.mfunctions)

        # Test argmax on random_choice (should return {"alice"})
        result_random = nondet_interpretation.mfunctions["random_choice"](*[])
        self.assertIsInstance(result_random, NonEmptyPowerset)
        self.assertEqual(set(result_random.value), {"alice"})

        # Test argmax on uniform_choice (should return {"charlie"})
        result_uniform = nondet_interpretation.mfunctions["uniform_choice"](*[])
        self.assertIsInstance(result_uniform, NonEmptyPowerset)
        self.assertEqual(set(result_uniform.value), {"charlie"})

    def test_argmax_preserves_regular_predicates_and_functions(self):
        """Test that argmax transformation preserves regular predicates and functions."""
        prob_interpretation = Interpretation(
            universe=self.universe,
            functions={
                "identity": lambda x: x,
                "father": lambda x: {"alice": "bob"}.get(x, "unknown"),
            },
            mfunctions={"m_func": lambda: Prob({"result": 1.0})},
            preds={
                "human": lambda x: x in self.universe,
                "parent": lambda x, y: x == "bob" and y == "alice",
            },
            mpreds={"m_pred": lambda x: Prob({True: 0.8, False: 0.2})},
        )

        # Apply argmax transformation
        argmax_transformer = argmax()
        nondet_interpretation = argmax_transformer(prob_interpretation)

        # Check that regular functions and predicates are preserved
        self.assertEqual(nondet_interpretation.functions, prob_interpretation.functions)
        self.assertEqual(nondet_interpretation.preds, prob_interpretation.preds)

        # Test regular functions still work
        self.assertEqual(nondet_interpretation.functions["identity"](*["alice"]), "alice")
        self.assertEqual(nondet_interpretation.functions["father"](*["alice"]), "bob")

        # Test regular predicates still work
        self.assertTrue(nondet_interpretation.preds["human"](*["alice"]))
        self.assertTrue(nondet_interpretation.preds["parent"](*["bob", "alice"]))
        self.assertFalse(nondet_interpretation.preds["parent"](*["alice", "bob"]))

    def test_argmax_empty_distributions(self):
        """Test argmax behavior with edge cases."""
        # Create interpretation with single-value distributions
        prob_interpretation = Interpretation(
            universe=self.universe,
            functions={},
            mfunctions={},
            preds={},
            mpreds={
                "certain_true": lambda x: Prob({True: 1.0}),
                "certain_false": lambda x: Prob({False: 1.0}),
            },
        )

        # Apply argmax transformation
        nondet_interpretation = prob_interpretation.transform(argmax())

        # Test certain distributions
        result_true = nondet_interpretation.mpreds["certain_true"](*["alice"])
        self.assertIsInstance(result_true, NonEmptyPowerset)
        self.assertEqual(set(result_true.value), {True})

        result_false = nondet_interpretation.mpreds["certain_false"](*["alice"])
        self.assertIsInstance(result_false, NonEmptyPowerset)
        self.assertEqual(set(result_false.value), {False})
        
    def test_traffic_light_example(self):
        """Test argmax with a traffic light example."""
        traffic_interpretation = traffic_light_model

        # Apply argmax transformation
        nondet_interpretation = traffic_interpretation.transform(argmax())

        # Test argmax on traffic light
        f = parse("L := $light() (D := $driveF(L) (eval(D) -> equals(L, green)))")

        # Test the formula
        result = f.eval(self.prob_nesy, nondet_interpretation, {})
        self.assertIsInstance(result, NonEmptyPowerset)
        self.assertSetEqual(set(result.value), {True})
        

if __name__ == "__main__":
    unittest.main()

import unittest
from typing import Mapping

from common import nf as prob_nf, traffic_light_model

from muller.hkt import List
from muller.monad.distribution import Prob
from muller.monad.non_empty_powerset import NonEmptyPowerset
from muller.nesy_framework import nesy
from muller.parser import parse
from muller.transformation import argmax


class TestTransformation(unittest.TestCase):
    """Test suite for the argmax transformation."""

    def setUp(self):
        """Set up common test data."""
        self.universe = ["alice", "bob", "charlie"]
        self.prob_nesy = nesy(Prob, bool, List)
        self.nondet_nesy = nesy(NonEmptyPowerset, bool, List)

    def test_argmax_basic_functionality(self):
        """Test basic argmax transformation functionality."""
        prob_interpretation = self.prob_nesy.create_interpretation(
            sort=List(self.universe),
            mpredicates={
                "uncertain_pred": lambda args: (
                    Prob({True: 0.7, False: 0.3})
                    if args[0] == "alice"
                    else Prob({True: 0.2, False: 0.8})
                ),
                "tie_pred": lambda _args: Prob(
                    {True: 0.5, False: 0.5}
                ),
            },
        )

        # Apply argmax transformation
        nondet_interpretation = argmax()(prob_interpretation)

        # Test that the transformation returns an interpretation
        self.assertEqual(nondet_interpretation.sort, List(self.universe))

        # Test argmax on uncertain_pred for alice (should return {True})
        result_alice = nondet_interpretation.mpredicates["uncertain_pred"](
            ["alice"]
        )
        self.assertIsInstance(result_alice, NonEmptyPowerset)
        self.assertEqual(set(result_alice._inner_value), {True})

        # Test argmax on uncertain_pred for bob (should return {False})
        result_bob = nondet_interpretation.mpredicates["uncertain_pred"](
            ["bob"]
        )
        self.assertIsInstance(result_bob, NonEmptyPowerset)
        self.assertEqual(set(result_bob._inner_value), {False})

        # Test argmax on tie_pred (should return {True, False})
        result_tie = nondet_interpretation.mpredicates["tie_pred"](
            ["alice"]
        )
        self.assertIsInstance(result_tie, NonEmptyPowerset)
        self.assertEqual(set(result_tie._inner_value), {True, False})

    def test_argmax_with_monadic_functions(self):
        """Test argmax transformation with monadic functions."""
        prob_interpretation = self.prob_nesy.create_interpretation(
            sort=List(self.universe),
            mfunctions={
                "random_choice": lambda _args: Prob(
                    {"alice": 0.6, "bob": 0.4}
                ),
                "uniform_choice": lambda _args: Prob(
                    {"alice": 0.33, "bob": 0.33, "charlie": 0.34}
                ),
            },
        )

        # Apply argmax transformation
        nondet_interpretation = argmax()(prob_interpretation)

        # Test argmax on random_choice (should return {"alice"})
        result_random = nondet_interpretation.mfunctions["random_choice"]([])
        self.assertIsInstance(result_random, NonEmptyPowerset)
        self.assertEqual(set(result_random._inner_value), {"alice"})

        # Test argmax on uniform_choice (should return {"charlie"})
        result_uniform = nondet_interpretation.mfunctions["uniform_choice"](
            []
        )
        self.assertIsInstance(result_uniform, NonEmptyPowerset)
        self.assertEqual(set(result_uniform._inner_value), {"charlie"})

    def test_argmax_preserves_regular_predicates_and_functions(self):
        """Test that argmax transformation preserves regular predicates and functions."""
        prob_interpretation = self.prob_nesy.create_interpretation(
            sort=List(self.universe),
            functions={
                "identity": lambda args: args[0],
                "father": lambda args: {"alice": "bob"}.get(
                    args[0], "unknown"
                ),
            },
            mfunctions={
                "m_func": lambda _args: Prob({"result": 1.0}),
            },
            predicates={
                "human": lambda args: args[0] in self.universe,
                "parent": lambda args: args[0] == "bob"
                and args[1] == "alice",
            },
            mpredicates={
                "m_pred": lambda _args: Prob({True: 0.8, False: 0.2}),
            },
        )

        # Apply argmax transformation
        nondet_interpretation = argmax()(prob_interpretation)

        # Check that regular functions and predicates are preserved
        self.assertEqual(
            nondet_interpretation.functions,
            prob_interpretation.functions,
        )
        self.assertEqual(
            nondet_interpretation.predicates,
            prob_interpretation.predicates,
        )

        # Test regular functions still work
        self.assertEqual(
            nondet_interpretation.functions["identity"](["alice"]), "alice"
        )
        self.assertEqual(
            nondet_interpretation.functions["father"](["alice"]), "bob"
        )

        # Test regular predicates still work
        self.assertTrue(
            nondet_interpretation.predicates["human"](["alice"])
        )
        self.assertTrue(
            nondet_interpretation.predicates["parent"](["bob", "alice"])
        )
        self.assertFalse(
            nondet_interpretation.predicates["parent"](["alice", "bob"])
        )

    def test_argmax_empty_distributions(self):
        """Test argmax behavior with edge cases."""
        prob_interpretation = self.prob_nesy.create_interpretation(
            sort=List(self.universe),
            mpredicates={
                "certain_true": lambda _args: Prob({True: 1.0}),
                "certain_false": lambda _args: Prob({False: 1.0}),
            },
        )

        # Apply argmax transformation
        nondet_interpretation = argmax()(prob_interpretation)

        # Test certain distributions
        result_true = nondet_interpretation.mpredicates["certain_true"](
            ["alice"]
        )
        self.assertIsInstance(result_true, NonEmptyPowerset)
        self.assertEqual(set(result_true._inner_value), {True})

        result_false = nondet_interpretation.mpredicates["certain_false"](
            ["alice"]
        )
        self.assertIsInstance(result_false, NonEmptyPowerset)
        self.assertEqual(set(result_false._inner_value), {False})

    def test_traffic_light_example(self):
        """Test argmax with a traffic light example."""
        # Apply argmax transformation
        nondet_interpretation = argmax()(traffic_light_model)

        # Test the formula
        f = parse(
            "L := $light()"
            " (D := $driveF(L) (eval(D) -> equals(L, green)))"
        )

        result = self.nondet_nesy.eval(
            f, nondet_interpretation, {}
        )
        self.assertIsInstance(result, NonEmptyPowerset)
        self.assertSetEqual(set(result._inner_value), {True})


if __name__ == "__main__":
    unittest.main()

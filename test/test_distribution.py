import random
import unittest

from muller.monad.distribution import (
    Prob,
    bernoulli,
    uniform,
    weighted,
)


class TestDistributionMonad(unittest.TestCase):
    """Test cases for the Prob (distribution) monad implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.tolerance_places = 6
        self.tolerance = 1 / 10**self.tolerance_places
        # Set random seed for reproducible tests
        random.seed(42)

    def assertAlmostEqualFloat(self, first: float, second: float, msg=None):
        """Helper method for comparing floats with tolerance."""
        self.assertAlmostEqual(first, second, places=self.tolerance_places, msg=msg)

    def assertDistributionEqual(self, dist1: Prob, dist2: Prob, msg=None):
        """Helper method for comparing distributions."""
        self.assertEqual(
            set(dist1.value.keys()),
            set(dist2.value.keys()),
            msg=f"Distribution keys differ: {msg}",
        )
        for key in dist1.value:
            self.assertAlmostEqualFloat(
                dist1.value[key],
                dist2.value[key],
                msg=f"Probability for {key} differs: {msg}",
            )

    def test_initialization_and_normalization(self):
        """Test that distributions are properly normalized."""
        # Test with unnormalized distribution
        dist = Prob({"A": 2.0, "B": 3.0, "C": 5.0})

        # Should be normalized to sum to 1
        total_prob = sum(dist.value.values())
        self.assertAlmostEqualFloat(total_prob, 1.0)

        # Check specific probabilities
        self.assertAlmostEqualFloat(dist.value["A"], 0.2)  # 2/10
        self.assertAlmostEqualFloat(dist.value["B"], 0.3)  # 3/10
        self.assertAlmostEqualFloat(dist.value["C"], 0.5)  # 5/10

    def test_insert_operation(self):
        """Test the insert (unit/return) operation."""
        # Create a deterministic distribution
        dist = Prob.insert("certain")

        # Should have probability 1 for the single value
        self.assertEqual(len(dist.value), 1)
        self.assertAlmostEqualFloat(dist.value["certain"], 1.0)

    def test_map_operation(self):
        """Test the map (functor) operation."""
        # Create a simple distribution
        dist = Prob({"1": 0.3, "2": 0.7})

        # Map with a function that converts strings to integers and doubles them
        mapped = dist.map(lambda x: int(x) * 2)

        # Check the result
        expected = Prob({2: 0.3, 4: 0.7})
        self.assertDistributionEqual(mapped, expected)

    def test_map_with_duplicates(self):
        """Test map operation when function produces duplicate values."""
        # Create distribution where map will create duplicates
        dist = Prob({"1": 0.3, "2": 0.4, "3": 0.3})

        # Map with a function that makes 1 and 3 both map to the same value
        mapped = dist.map(lambda x: "even" if int(x) % 2 == 0 else "odd")

        # Check that probabilities are properly combined
        expected = Prob({"even": 0.4, "odd": 0.6})  # 0.3 + 0.3 = 0.6 for odd
        self.assertDistributionEqual(mapped, expected)

    def test_bind_operation(self):
        """Test the bind (monadic composition) operation."""
        # Create initial distribution
        dist = Prob({"heads": 0.5, "tails": 0.5})

        # Bind with a function that creates new distributions
        def coin_bias(outcome):
            if outcome == "heads":
                return Prob({"win": 0.8, "lose": 0.2})
            else:
                return Prob({"win": 0.3, "lose": 0.7})

        result = dist.bind(coin_bias)

        # Expected: P(win) = 0.5 * 0.8 + 0.5 * 0.3 = 0.55
        #          P(lose) = 0.5 * 0.2 + 0.5 * 0.7 = 0.45
        expected = Prob({"win": 0.55, "lose": 0.45})
        self.assertDistributionEqual(result, expected)

    def test_monad_laws(self):
        """Test that the monad laws hold."""
        # Left identity: unit(a).bind(f) == f(a)
        a = "test"

        def f(x):
            return Prob({x + "_mapped": 0.6, x + "_other": 0.4})

        left = Prob.insert(a).bind(f)
        right = f(a)
        self.assertDistributionEqual(left, right)

        # Right identity: m.bind(unit) == m
        m = Prob({"A": 0.3, "B": 0.7})
        bound = m.bind(Prob.insert)
        self.assertDistributionEqual(m, bound)

        # Associativity: (m.bind(f)).bind(g) == m.bind(lambda x: f(x).bind(g))
        def g(x):
            return Prob({x + "_g1": 0.2, x + "_g2": 0.8})

        left_assoc = m.bind(f).bind(g)
        right_assoc = m.bind(lambda x: f(x).bind(g))
        self.assertDistributionEqual(left_assoc, right_assoc)

    def test_expected_value(self):
        """Test expected value calculation."""
        # Simple numeric distribution
        dist = Prob({1: 0.2, 2: 0.3, 3: 0.5})

        # Expected value should be 1*0.2 + 2*0.3 + 3*0.5 = 2.3
        expected_val = dist.expected_value(lambda x: x)
        self.assertAlmostEqualFloat(expected_val, 2.3)

        # Test with a transformation function
        expected_squared = dist.expected_value(lambda x: x**2)
        # E[X^2] = 1^2*0.2 + 2^2*0.3 + 3^2*0.5 = 0.2 + 1.2 + 4.5 = 5.9
        self.assertAlmostEqualFloat(expected_squared, 5.9)

    def test_sampling(self):
        """Test sampling from distribution."""
        dist = Prob({"A": 0.8, "B": 0.2})

        # Sample many times and check approximate frequencies
        samples = [dist.sample() for _ in range(1000)]
        freq_A = samples.count("A") / 1000
        freq_B = samples.count("B") / 1000

        # Should be approximately 0.8 and 0.2 (within reasonable tolerance)
        self.assertAlmostEqual(freq_A, 0.8, places=1)
        self.assertAlmostEqual(freq_B, 0.2, places=1)

    def test_filter_operation(self):
        """Test filtering distributions."""
        dist = Prob({1: 0.2, 2: 0.3, 3: 0.5})

        # Filter to keep only even numbers
        filtered = dist.filter(lambda x: x % 2 == 0)

        # Should only contain 2, renormalized
        expected = Prob({2: 1.0})
        self.assertDistributionEqual(filtered, expected)

        # Filter to keep numbers > 1
        filtered2 = dist.filter(lambda x: x > 1)

        # Should contain 2 and 3, renormalized
        # Original: 2->0.3, 3->0.5, total=0.8
        # Normalized: 2->0.3/0.8=0.375, 3->0.5/0.8=0.625
        expected2 = Prob({2: 0.375, 3: 0.625})
        self.assertDistributionEqual(filtered2, expected2)

    def test_max_probability_and_argmax(self):
        """Test finding maximum probability and corresponding values."""
        dist = Prob({"A": 0.2, "B": 0.5, "C": 0.3})

        # Max probability should be 0.5
        self.assertAlmostEqualFloat(dist.max_probability(), 0.5)

        # Argmax should return ["B"]
        self.assertEqual(dist.argmax(), ["B"])

        # Test with tie
        dist_tie = Prob({"A": 0.4, "B": 0.4, "C": 0.2})
        argmax_tie = set(dist_tie.argmax())
        self.assertEqual(argmax_tie, {"A", "B"})

    def test_uniform_distribution(self):
        """Test uniform distribution creation."""
        values = ["red", "green", "blue"]
        dist = uniform(values)

        # Each value should have probability 1/3
        for value in values:
            self.assertAlmostEqualFloat(dist.value[value], 1 / 3)

        # Test empty uniform distribution
        empty_dist = uniform([])
        self.assertEqual(len(empty_dist.value), 0)

    def test_weighted_distribution(self):
        """Test weighted distribution creation."""
        pairs = [("A", 2.0), ("B", 3.0), ("C", 1.0)]
        dist = weighted(pairs)

        # Should be normalized: total weight = 6
        expected = Prob({"A": 2 / 6, "B": 3 / 6, "C": 1 / 6})
        self.assertDistributionEqual(dist, expected)

    def test_bernoulli_distribution(self):
        """Test Bernoulli distribution creation."""
        p = 0.3
        dist = bernoulli(p)

        # Default values should be True/False
        self.assertAlmostEqualFloat(dist.value[True], 0.3)
        self.assertAlmostEqualFloat(dist.value[False], 0.7)

        # Test with custom values
        dist_custom = bernoulli(0.6, true_val="success", false_val="failure")
        self.assertAlmostEqualFloat(dist_custom.value["success"], 0.6)
        self.assertAlmostEqualFloat(dist_custom.value["failure"], 0.4)

    def test_complex_composition(self):
        """Test complex monadic compositions."""
        # Start with a biased coin
        coin = Prob({"heads": 0.6, "tails": 0.4})

        # Chain multiple operations
        result = (
            coin.bind(lambda outcome: Prob({outcome + "_round1": 1.0}))
            .map(lambda x: x.replace("round1", "round2"))
            .bind(lambda x: Prob({x + "_final": 0.8, x + "_alternate": 0.2}))
        )

        # Check that probabilities sum to 1
        total_prob = sum(result.value.values())
        self.assertAlmostEqualFloat(total_prob, 1.0)

        # Check specific probabilities
        expected_heads_final = 0.6 * 0.8  # 0.48
        expected_tails_final = 0.4 * 0.8  # 0.32
        self.assertAlmostEqualFloat(
            result.value["heads_round2_final"], expected_heads_final
        )
        self.assertAlmostEqualFloat(
            result.value["tails_round2_final"], expected_tails_final
        )

    def test_empty_distribution_handling(self):
        """Test handling of edge cases with empty distributions."""
        # Test max_probability on empty distribution
        empty_dist = Prob({})
        self.assertEqual(empty_dist.max_probability(), 0.0)
        self.assertEqual(empty_dist.argmax(), [])

        # Test expected value on empty distribution
        self.assertEqual(empty_dist.expected_value(lambda x: x), 0)

    def test_string_representation(self):
        """Test string representation of distributions."""
        dist = Prob({"A": 0.3, "B": 0.7})
        repr_str = repr(dist)

        # Should be sorted by probability (descending)
        self.assertTrue("Prob(" in repr_str)
        self.assertTrue("B" in repr_str)
        self.assertTrue("A" in repr_str)

        # B should come before A (higher probability)
        b_pos = repr_str.find("'B'")
        a_pos = repr_str.find("'A'")
        self.assertLess(b_pos, a_pos)

    def test_variance_calculation(self):
        """Test variance calculation using expected value."""
        dist = Prob({1: 0.2, 2: 0.3, 3: 0.5})

        # Calculate mean and variance
        mean = dist.expected_value(lambda x: x)
        variance = dist.expected_value(lambda x: (x - mean) ** 2)

        # Manual calculation:
        # mean = 1*0.2 + 2*0.3 + 3*0.5 = 2.3
        # variance = (1-2.3)^2*0.2 + (2-2.3)^2*0.3 + (3-2.3)^2*0.5
        #          = 1.69*0.2 + 0.09*0.3 + 0.49*0.5 = 0.338 + 0.027 + 0.245 = 0.61
        self.assertAlmostEqualFloat(mean, 2.3)
        self.assertAlmostEqualFloat(variance, 0.61)

    def test_bind_with_empty_result(self):
        """Test bind operation that can produce empty distributions."""
        dist = Prob({"A": 0.5, "B": 0.5})

        # Bind with function that sometimes returns empty distribution
        def conditional_dist(x):
            if x == "A":
                return Prob({"result": 1.0})
            else:
                return Prob({})  # Empty distribution

        result = dist.bind(conditional_dist)

        # Should only contain results from "A"
        expected = Prob({"result": 1.0})
        self.assertDistributionEqual(result, expected)


if __name__ == "__main__":
    unittest.main()

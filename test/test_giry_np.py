import math
import unittest

from muller.monad.giry_sampling import (
    GirySampling,
    beta,
    binomial,
    normal,
    uniform,
    categorical,
    geometric
)

def sample(value: GirySampling) -> float:
    """Numerically integrate a function over the support of the distribution."""
    # For simplicity, we'll use a basic rectangle method for integration
    # In a real implementation, you'd want to use a more robust numerical integration method
    return value.mean(100_000)


class TestGirySampling(unittest.TestCase):
    """Test cases for the Giry monad implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.tolerance_places = 1

    def assertAlmostEqualFloat(self, first: float, second: float, msg=None):
        """Helper method for comparing floats with tolerance."""
        self.assertAlmostEqual(first, second, places=self.tolerance_places, msg=msg)

    def test_insert_creation(self):
        """Test that.insert creates a point mass measure."""
        # Create a point mass at value 5
        point_mass = GirySampling.insert(5)

        # Test that integrating identity function gives the value
        result = sample(point_mass)
        self.assertAlmostEqualFloat(result, 5.0)

    def test_map_operation(self):
        """Test the map operation (functor law)."""
        # Create a point mass at 3
        monad = GirySampling.insert(3)

        # Map with a function that doubles the value
        mapped = monad.map(lambda x: x * 2)

        # Integrate identity to get the new value
        result = sample(mapped)
        self.assertAlmostEqualFloat(result, 6.0)

    def test_bind_operation(self):
        """Test the bind operation (monad law)."""
        # Create a point mass at 2
        monad = GirySampling.insert(2)

        # Bind with a function that creates a point mass at double the value
        bound = monad.bind(lambda x: GirySampling.insert(x * 3))

        # Integrate identity to get the result
        result = sample(bound)
        self.assertAlmostEqualFloat(result, 6.0)

    def test_monad_laws(self):
        """Test that the monad laws hold."""
        # Left identity:.insert(a).bind(f) == f(a)
        a = 5

        def f(x):
            return GirySampling.insert(x + 1)

        left = GirySampling.insert(a).bind(f)
        right = f(a)

        left_result = sample(left)
        right_result = sample(right)
        self.assertAlmostEqualFloat(left_result, right_result)

        # Right identity: m.bind.insert) == m
        m = GirySampling.insert(7)
        bound = m.bind(GirySampling.insert)

        left_result = sample(m)
        right_result = sample(bound)
        self.assertAlmostEqualFloat(left_result, right_result)

    def test_binomial_distribution(self):
        """Test the binomial distribution implementation."""
        n = 3
        p = 0.4
        monad = binomial(n, p)

        # Test expected value (np)
        expected_value = sample(monad)
        self.assertAlmostEqualFloat(expected_value, n * p)

    def test_binomial_edge_cases(self):
        """Test binomial distribution edge cases."""
        # Test with p = 0 (always 0 successes)
        monad = binomial(5, 0.0)
        prob_zero = sample(monad)
        self.assertAlmostEqualFloat(prob_zero, 0)

        # Test with p = 1 (always n successes)
        n = 4
        monad = binomial(n, 1.0)
        prob_n = sample(monad)
        self.assertAlmostEqualFloat(prob_n, n)

    def test_beta_distribution(self):
        """Test the beta distribution implementation."""
        a, b = 2.0, 3.0
        monad = beta(a, b)

        # Test expected value (a/(a+b) for beta distribution)
        expected_value = sample(monad)
        theoretical_mean = a / (a + b)
        self.assertAlmostEqual(expected_value, theoretical_mean, places=2)

    def test_uniform_distribution(self):
        """Test the uniform distribution implementation."""
        low, high = 0.0, 1.0
        monad = uniform(low, high)

        # Test expected value ((low + high) / 2 for uniform distribution)
        expected_value = sample(monad)
        theoretical_mean = (low + high) / 2
        self.assertAlmostEqual(expected_value, theoretical_mean, places=2)

    def test_normal_distribution(self):
        """Test the normal distribution implementation."""
        mu, sigma = 0.0, 1.0
        monad = normal(mu, sigma)

        # Test expected value (mu for normal distribution)
        expected_value = sample(monad)
        theoretical_mean = mu
        self.assertAlmostEqual(expected_value, theoretical_mean, places=2)

    def test_complex_composition(self):
        """Test complex monadic compositions."""
        # Create a chain of operations
        initial = GirySampling.insert(1)

        # Chain multiple bind operations
        result = (
            initial.bind(lambda x: GirySampling.insert(x + 1))  # 1 -> 2
            .bind(lambda x: binomial(x, 0.5))  # 2 -> binomial(2, 0.5)
            .map(lambda x: x * 2)
        )  # double each outcome

        # Test that the expected value is reasonable
        expected_value = sample(result)
        # E[2 * Binomial(2, 0.5)] = 2 * 2 * 0.5 = 2
        self.assertAlmostEqualFloat(expected_value, 2.0)


    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very small probabilities
        small_p = 1e-10
        monad = binomial(2, small_p)
        total_prob = sample(monad)
        self.assertAlmostEqual(total_prob, 0.0, places=8)

        # Test with probabilities close to 1
        large_p = 1 - 1e-10
        monad = binomial(2, large_p)
        total_prob = sample(monad)
        self.assertAlmostEqual(total_prob, 2, places=8)


if __name__ == "__main__":
   unittest.main()

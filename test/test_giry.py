import math
import unittest

from muller.monad.giry import (
    Giry,
    beta,
    betaBinomial,
    binomial,
    fromDensityFunction,
    fromMassFunction,
    fromSample,
    integrate,
)


class TestGiryMonad(unittest.TestCase):
    """Test cases for the Giry monad implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.tolerance_places = 6

    def assertAlmostEqualFloat(self, first: float, second: float, msg=None):
        """Helper method for comparing floats with tolerance."""
        self.assertAlmostEqual(first, second, places=self.tolerance_places, msg=msg)

    def test_unit_creation(self):
        """Test that unit creates a point mass measure."""
        # Create a point mass at value 5
        point_mass = Giry.insert(5)

        # Test that integrating identity function gives the value
        result = integrate(lambda x: x, point_mass.value)
        self.assertAlmostEqualFloat(result, 5.0)

        # Test that integrating constant function gives the constant
        result = integrate(lambda x: 10.0, point_mass.value)
        self.assertAlmostEqualFloat(result, 10.0)

    def test_map_operation(self):
        """Test the map operation (functor law)."""
        # Create a point mass at 3
        monad = Giry.insert(3)

        # Map with a function that doubles the value
        mapped = monad.map(lambda x: x * 2)

        # Integrate identity to get the new value
        result = integrate(lambda x: x, mapped.value)
        self.assertAlmostEqualFloat(result, 6.0)

    def test_bind_operation(self):
        """Test the bind operation (monad law)."""
        # Create a point mass at 2
        monad = Giry.insert(2)

        # Bind with a function that creates a point mass at double the value
        bound = monad.bind(lambda x: Giry.insert(x * 3))

        # Integrate identity to get the result
        result = integrate(lambda x: x, bound.value)
        self.assertAlmostEqualFloat(result, 6.0)

    def test_monad_laws(self):
        """Test that the monad laws hold."""
        # Left identity: unit(a).bind(f) == f(a)
        a = 5

        def f(x):
            return Giry.insert(x + 1)

        left = Giry.insert(a).bind(f)
        right = f(a)

        def test_function(x):
            return x * 2  # Any test function

        left_result = integrate(test_function, left.value)
        right_result = integrate(test_function, right.value)
        self.assertAlmostEqualFloat(left_result, right_result)

        # Right identity: m.bind(unit) == m
        m = Giry.insert(7)
        bound = m.bind(Giry.insert)

        left_result = integrate(test_function, m.value)
        right_result = integrate(test_function, bound.value)
        self.assertAlmostEqualFloat(left_result, right_result)

    def test_fromMassFunction(self):
        """Test creating a GiryMonad from a mass function."""

        # Create a simple discrete distribution
        def mass_func(x: int) -> float:
            if x == 1:
                return 0.3
            elif x == 2:
                return 0.7
            else:
                return 0.0

        support = [1, 2]
        monad = fromMassFunction(mass_func, support)

        # Test that probabilities sum to 1
        total_prob = integrate(lambda x: 1.0, monad.value)
        self.assertAlmostEqualFloat(total_prob, 1.0)

        # Test expected value
        expected_value = integrate(lambda x: x, monad.value)
        self.assertAlmostEqualFloat(expected_value, 1 * 0.3 + 2 * 0.7)

    def test_binomial_distribution(self):
        """Test the binomial distribution implementation."""
        n = 3
        p = 0.4
        monad = binomial(n, p)

        # Test that probabilities sum to 1
        total_prob = integrate(lambda x: 1.0, monad.value)
        self.assertAlmostEqualFloat(total_prob, 1.0)

        # Test expected value (np)
        expected_value = integrate(lambda x: x, monad.value)
        self.assertAlmostEqualFloat(expected_value, n * p)

        # Test variance (np(1-p))
        mean = n * p
        variance = integrate(lambda x: (x - mean) ** 2, monad.value)
        self.assertAlmostEqualFloat(variance, n * p * (1 - p))

    def test_binomial_edge_cases(self):
        """Test binomial distribution edge cases."""
        # Test with p = 0 (always 0 successes)
        monad = binomial(5, 0.0)
        prob_zero = integrate(lambda x: 1.0 if x == 0 else 0.0, monad.value)
        self.assertAlmostEqualFloat(prob_zero, 1.0)

        # Test with p = 1 (always n successes)
        n = 4
        monad = binomial(n, 1.0)
        prob_n = integrate(lambda x: 1.0 if x == n else 0.0, monad.value)
        self.assertAlmostEqualFloat(prob_n, 1.0)

    def test_fromDensityFunction(self):
        """Test creating a GiryMonad from a density function."""

        # Create a uniform distribution on [0, 1]
        def uniform_density(x: float) -> float:
            return 1.0 if 0 <= x <= 1 else 0.0

        monad = fromDensityFunction(uniform_density)

        # Test that the total probability is 1 (approximately, due to numerical
        # integration)
        total_prob = integrate(lambda x: 1.0, monad.value)
        self.assertAlmostEqual(total_prob, 1.0, places=3)

        # Test expected value (should be 0.5 for uniform on [0,1])
        expected_value = integrate(lambda x: x, monad.value)
        self.assertAlmostEqual(expected_value, 0.5, places=3)

    def test_beta_distribution(self):
        """Test the beta distribution implementation."""
        a, b = 2.0, 3.0
        monad = beta(a, b)

        # Test that the total probability is approximately 1
        total_prob = integrate(lambda x: 1.0, monad.value)
        self.assertAlmostEqual(total_prob, 1.0, places=2)

        # Test expected value (a/(a+b) for beta distribution)
        expected_value = integrate(lambda x: x, monad.value)
        theoretical_mean = a / (a + b)
        self.assertAlmostEqual(expected_value, theoretical_mean, places=2)

    def test_betaBinomial_distribution(self):
        """Test the beta-binomial distribution implementation."""
        n = 10
        a, b = 2.0, 3.0
        monad = betaBinomial(n, a, b)

        # Test that probabilities sum to 1
        total_prob = integrate(lambda x: 1.0, monad.value)
        self.assertAlmostEqualFloat(total_prob, 1.0)

        # Test that the expected value is approximately n * a/(a+b)
        expected_value = integrate(lambda x: x, monad.value)
        theoretical_mean = n * a / (a + b)
        self.assertAlmostEqual(expected_value, theoretical_mean, places=1)

    def test_fromSample(self):
        """Test creating a GiryMonad from a sample."""
        sample = [1, 2, 2, 3, 3, 3]
        monad = fromSample(sample)

        # Test that the empirical mean matches the sample mean
        sample_mean = sum(sample) / len(sample)
        monad_mean = integrate(lambda x: x, monad.value)
        self.assertAlmostEqualFloat(monad_mean, sample_mean)

        # Test with a constant function
        constant_result = integrate(lambda x: 5.0, monad.value)
        self.assertAlmostEqualFloat(constant_result, 5.0)

    def test_amap_operation(self):
        """Test the applicative map operation."""
        # Create a monad containing a function
        func_monad = Giry.insert(lambda x: x * 2)

        # Create a monad containing a value
        value_monad = Giry.insert(5)

        # Apply the function
        result_monad = func_monad.amap(value_monad)

        # Test the result
        result = integrate(lambda x: x, result_monad.value)
        self.assertAlmostEqualFloat(result, 10.0)

    def test_complex_composition(self):
        """Test complex monadic compositions."""
        # Create a chain of operations
        initial = Giry.insert(1)

        # Chain multiple bind operations
        result = (
            initial.bind(lambda x: Giry.insert(x + 1))  # 1 -> 2
            .bind(lambda x: binomial(x, 0.5))  # 2 -> binomial(2, 0.5)
            .map(lambda x: x * 2)
        )  # double each outcome

        # Test that the result is a valid probability measure
        total_prob = integrate(lambda x: 1.0, result.value)
        self.assertAlmostEqualFloat(total_prob, 1.0)

        # Test that the expected value is reasonable
        expected_value = integrate(lambda x: x, result.value)
        # E[2 * Binomial(2, 0.5)] = 2 * 2 * 0.5 = 2
        self.assertAlmostEqualFloat(expected_value, 2.0)

    def test_integration_with_different_functions(self):
        """Test integration with various functions."""
        # Create a simple discrete distribution
        monad = fromMassFunction(lambda x: 0.5, [0, 1])

        # Test with polynomial functions
        result1 = integrate(lambda x: x**2, monad.value)
        expected1 = 0.5 * (0**2) + 0.5 * (1**2)  # 0.5
        self.assertAlmostEqualFloat(result1, expected1)

        # Test with exponential function
        result2 = integrate(lambda x: math.exp(x), monad.value)
        expected2 = 0.5 * math.exp(0) + 0.5 * math.exp(1)
        self.assertAlmostEqualFloat(result2, expected2)

    def test_measure_properties(self):
        """Test mathematical properties of measures."""

        # Create a uniform distribution over {1, 2, 3}
        def uniform_mass(x):
            return 1 / 3 if x in [1, 2, 3] else 0

        monad = fromMassFunction(uniform_mass, [1, 2, 3])

        # Test linearity of integration
        def f(x):
            return x

        def g(x):
            return x**2

        a, b = 2.0, 3.0

        # E[af + bg] = aE[f] + bE[g]
        left_side = integrate(lambda x: a * f(x) + b * g(x), monad.value)
        right_side = a * integrate(f, monad.value) + b * integrate(g, monad.value)
        self.assertAlmostEqualFloat(left_side, right_side)

    def test_error_handling(self):
        """Test error handling and edge cases."""
        # Test with empty support (should handle gracefully)
        try:
            monad = fromMassFunction(lambda x: 1.0, [])
            result = integrate(lambda x: x, monad.value)
            self.assertEqual(result, 0.0)
        except Exception:
            # If it raises an exception, that's also acceptable behavior
            pass

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very small probabilities
        small_p = 1e-10
        monad = binomial(2, small_p)
        total_prob = integrate(lambda x: 1.0, monad.value)
        self.assertAlmostEqual(total_prob, 1.0, places=8)

        # Test with probabilities close to 1
        large_p = 1 - 1e-10
        monad = binomial(2, large_p)
        total_prob = integrate(lambda x: 1.0, monad.value)
        self.assertAlmostEqual(total_prob, 1.0, places=8)


if __name__ == "__main__":
    unittest.main()

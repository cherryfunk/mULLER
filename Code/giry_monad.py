# pip install pymonad numpy scipy
from pymonad.monad import Monad
from typing import TypeVar, Callable, Union, List, Tuple, Optional
import numpy as np
from scipy import integrate
from scipy.stats import norm, uniform as scipy_uniform, expon, gamma
import random
import math

T = TypeVar('T')
U = TypeVar('U')

class GiryMeasure(Monad):
    """
    Giry Monad for Probability Measures on Measurable Spaces
    
    Represents probability measures, supporting both discrete and continuous distributions.
    Based on the mathematical definition:
    - ηX(x) := δx (Dirac delta measure)
    - δx(A) = { 1, x ∈ A; 0, x ∉ A }
    - f*(ρ)(A) := ∫ f(x)(A) dρ(x) for f : X → GY, ρ ∈ GX, A ⊆ Y measurable
    """
    
    def __init__(self, distribution_type: str = "discrete", 
                 values: Optional[List] = None, 
                 probabilities: Optional[List] = None,
                 density_func: Optional[Callable] = None,
                 sampler: Optional[Callable] = None,
                 support: Optional[Tuple[float, float]] = None,
                 name: str = "unnamed"):
        """
        Initialize a Giry measure.
        
        Args:
            distribution_type: "discrete", "continuous", or "dirac"
            values: For discrete distributions, list of values
            probabilities: For discrete distributions, corresponding probabilities
            density_func: For continuous distributions, probability density function
            sampler: Function to sample from the distribution
            support: (min, max) support of the distribution
            name: Name for debugging/display
        """
        self.distribution_type = distribution_type
        self.values = values or []
        self.probabilities = probabilities or []
        self.density_func = density_func
        self.sampler = sampler
        self.support = support
        self.name = name
        
        # Normalize discrete probabilities
        if distribution_type == "discrete" and self.probabilities:
            total = sum(self.probabilities)
            if total > 0:
                self.probabilities = [p/total for p in self.probabilities]
    
    @classmethod
    def unit(cls, value: T) -> 'GiryMeasure':
        """
        Create a Dirac delta measure δx.
        Implements ηX(x) := δx
        
        Args:
            value: The point mass location
            
        Returns:
            Dirac delta measure at the given point
        """
        return cls(
            distribution_type="dirac",
            values=[value],
            probabilities=[1.0],
            sampler=lambda: value,
            name=f"δ_{value}"
        )
    
    def bind(self, f: Callable[[T], 'GiryMeasure']) -> 'GiryMeasure':
        """
        Monadic bind operation (Kleisli extension).
        Implements f*(ρ)(A) := ∫ f(x)(A) dρ(x)
        
        For discrete measures, this becomes a weighted sum.
        For continuous measures, we use Monte Carlo integration.
        
        Args:
            f: Function from value to Giry measure
            
        Returns:
            New Giry measure
        """
        if self.distribution_type == "discrete" or self.distribution_type == "dirac":
            # Discrete case: weighted sum
            new_values = []
            new_probabilities = []
            
            for value, prob in zip(self.values, self.probabilities):
                new_measure = f(value)
                if new_measure.distribution_type == "discrete" or new_measure.distribution_type == "dirac":
                    for new_val, new_prob in zip(new_measure.values, new_measure.probabilities):
                        new_values.append(new_val)
                        new_probabilities.append(prob * new_prob)
                else:
                    # If f returns continuous distribution, we need to sample
                    # This is a simplification - in practice would need more sophisticated handling
                    samples = [new_measure.sample() for _ in range(100)]
                    for sample in samples:
                        new_values.append(sample)
                        new_probabilities.append(prob / 100)
            
            return GiryMeasure(
                distribution_type="discrete",
                values=new_values,
                probabilities=new_probabilities,
                name=f"bind({self.name})"
            )
        
        else:
            # Continuous case: Monte Carlo integration
            n_samples = 1000
            samples = [self.sample() for _ in range(n_samples)]
            
            # Apply f to each sample and collect results
            result_samples = []
            for sample in samples:
                new_measure = f(sample)
                result_samples.append(new_measure.sample())
            
            # Create empirical distribution from samples
            return GiryMeasure(
                distribution_type="discrete",
                values=result_samples,
                probabilities=[1.0/len(result_samples)] * len(result_samples),
                name=f"bind({self.name})"
            )
    
    def map(self, f: Callable[[T], U]) -> 'GiryMeasure':
        """
        Apply a function to the measure (pushforward measure).
        
        Args:
            f: Function to apply
            
        Returns:
            New Giry measure
        """
        if self.distribution_type == "discrete" or self.distribution_type == "dirac":
            # Apply function to discrete values
            new_values = [f(val) for val in self.values]
            return GiryMeasure(
                distribution_type="discrete",
                values=new_values,
                probabilities=self.probabilities.copy(),
                name=f"map({self.name})"
            )
        else:
            # For continuous distributions, create new sampler
            new_sampler = lambda: f(self.sample())
            return GiryMeasure(
                distribution_type="continuous",
                sampler=new_sampler,
                name=f"map({self.name})"
            )
    
    def sample(self) -> T:
        """Sample from the measure."""
        if self.sampler:
            return self.sampler()
        elif self.distribution_type == "discrete":
            return random.choices(self.values, weights=self.probabilities)[0]
        else:
            raise ValueError("No sampler available for this measure")
    
    def measure(self, measurable_set: Callable[[T], bool]) -> float:
        """
        Compute the measure of a measurable set.
        Implements δx(A) = { 1, x ∈ A; 0, x ∉ A } for Dirac measures
        
        Args:
            measurable_set: Characteristic function of the set
            
        Returns:
            Measure of the set
        """
        if self.distribution_type == "discrete" or self.distribution_type == "dirac":
            # Sum probabilities of values in the set
            total = 0.0
            for value, prob in zip(self.values, self.probabilities):
                if measurable_set(value):
                    total += prob
            return total
        else:
            # Monte Carlo estimation for continuous distributions
            n_samples = 10000
            samples = [self.sample() for _ in range(n_samples)]
            count = sum(1 for s in samples if measurable_set(s))
            return count / n_samples
    
    def expected_value(self, f: Callable[[T], float] = lambda x: float(x)) -> float:
        """Compute expected value E[f(X)]."""
        if self.distribution_type == "discrete" or self.distribution_type == "dirac":
            return sum(f(val) * prob for val, prob in zip(self.values, self.probabilities))
        else:
            # Monte Carlo estimation
            n_samples = 1000
            samples = [self.sample() for _ in range(n_samples)]
            return sum(f(s) for s in samples) / n_samples
    
    def __repr__(self):
        if self.distribution_type == "dirac":
            return f"δ_{self.values[0]}"
        elif self.distribution_type == "discrete":
            return f"GiryMeasure({self.name}, discrete)"
        else:
            return f"GiryMeasure({self.name}, continuous)"


# Convenience functions for creating common measures

def dirac(value: T) -> GiryMeasure:
    """Create a Dirac delta measure δx."""
    return GiryMeasure.unit(value)

def discrete_uniform(values: List[T]) -> GiryMeasure:
    """Create discrete uniform distribution."""
    n = len(values)
    probs = [1.0/n] * n
    return GiryMeasure(
        distribution_type="discrete",
        values=values,
        probabilities=probs,
        name="discrete_uniform"
    )

def bernoulli(p: float) -> GiryMeasure:
    """Create Bernoulli distribution."""
    return GiryMeasure(
        distribution_type="discrete",
        values=[True, False],
        probabilities=[p, 1-p],
        name=f"Bernoulli({p})"
    )

def normal(mu: float = 0.0, sigma: float = 1.0) -> GiryMeasure:
    """Create normal distribution."""
    dist = norm(loc=mu, scale=sigma)
    return GiryMeasure(
        distribution_type="continuous",
        density_func=dist.pdf,
        sampler=dist.rvs,
        support=(-np.inf, np.inf),
        name=f"N({mu},{sigma})"
    )

def uniform(a: float = 0.0, b: float = 1.0) -> GiryMeasure:
    """Create uniform distribution."""
    dist = scipy_uniform(loc=a, scale=b-a)
    return GiryMeasure(
        distribution_type="continuous",
        density_func=dist.pdf,
        sampler=dist.rvs,
        support=(a, b),
        name=f"U({a},{b})"
    )

def exponential(rate: float = 1.0) -> GiryMeasure:
    """Create exponential distribution."""
    dist = expon(scale=1/rate)
    return GiryMeasure(
        distribution_type="continuous",
        density_func=dist.pdf,
        sampler=dist.rvs,
        support=(0, np.inf),
        name=f"Exp({rate})"
    )


# Example usage demonstrating probability measures

if __name__ == "__main__":
    # Example 1: Dirac delta measure
    delta_5 = dirac(5)
    print("Dirac δ_5:", delta_5)
    print("δ_5({5}):", delta_5.measure(lambda x: x == 5))
    print("δ_5({1,2,3}):", delta_5.measure(lambda x: x in [1,2,3]))
    
    # Example 2: Discrete uniform distribution
    dice = discrete_uniform([1, 2, 3, 4, 5, 6])
    print("\nDice:", dice)
    print("P(X ≤ 3):", dice.measure(lambda x: x <= 3))
    print("E[X]:", dice.expected_value())
    
    # Example 3: Bernoulli distribution
    coin = bernoulli(0.6)
    print("\nBiased coin:", coin)
    print("P(True):", coin.measure(lambda x: x == True))
    
    # Example 4: Normal distribution
    gaussian = normal(0, 1)
    print("\nStandard normal:", gaussian)
    print("E[X]:", gaussian.expected_value())
    print("E[X²]:", gaussian.expected_value(lambda x: x**2))
    
    # Example 5: Monadic bind with Dirac measures
    # Start with δ_2, then map x → δ_{x+1}
    result = delta_5.bind(lambda x: dirac(x + 1))
    print("\nδ_5 >>= (x → δ_{x+1}):", result)
    print("Sample:", result.sample())
    
    # Example 6: Bind with discrete distribution
    # Roll dice, then flip coins based on the result
    dice_then_coins = dice.bind(lambda x: discrete_uniform([0, 1] * x))
    print("\nDice then coins:", dice_then_coins)
    print("Sample:", dice_then_coins.sample())
    
    # Example 7: Continuous distribution bind
    # Sample from normal, then create exponential with that rate
    normal_to_exp = normal(2, 0.5).bind(lambda x: exponential(abs(x)) if x > 0 else dirac(0))
    print("\nNormal to exponential:", normal_to_exp)
    print("Sample:", normal_to_exp.sample())
    
    # Example 8: Pushforward measure (map)
    squared_dice = dice.map(lambda x: x**2)
    print("\nSquared dice:", squared_dice)
    print("E[X²]:", squared_dice.expected_value())
    
    # Example 9: Measure of intervals for continuous distributions
    unit_uniform = uniform(0, 1)
    print("\nUnit uniform:", unit_uniform)
    print("P(X ∈ [0.3, 0.7]):", unit_uniform.measure(lambda x: 0.3 <= x <= 0.7))
    
    # Example 10: Composition of measures
    # Sample from uniform, then normal with that mean
    composition = unit_uniform.bind(lambda mu: normal(mu, 0.1))
    print("\nComposition:", composition)
    print("Sample:", composition.sample()) 
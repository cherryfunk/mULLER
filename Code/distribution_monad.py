# pip install pymonad
from pymonad.monad import Monad
from collections import defaultdict
from typing import TypeVar, Callable, Dict, Tuple, Any
import random

T = TypeVar('T')
U = TypeVar('U')

class Prob(Monad):
    """
    Probability Distribution Monad
    
    Represents a discrete probability distribution as a dictionary
    mapping values to their probabilities.
    """
    
    def __init__(self, dist: Dict[T, float]):
        """
        Initialize a probability distribution.
        
        Args:
            dist: Dictionary mapping values to probabilities
        """
        # Normalize probabilities
        total = sum(dist.values())
        if total > 0:
            self.dist = {k: v/total for k, v in dist.items()}
        else:
            self.dist = {}
    
    @classmethod
    def unit(cls, value: T) -> 'Prob[T]':
        """
        Create a distribution with certainty for a single value.
        Also known as 'return' or 'pure' in Haskell.
        
        Args:
            value: The value with probability 1.0
            
        Returns:
            Prob distribution with single value
        """
        return cls({value: 1.0})
    
    def bind(self, f: Callable[[T], 'Prob[U]']) -> 'Prob[U]':
        """
        Monadic bind operation (>>=).
        
        Apply a function that returns a probability distribution
        to each value in this distribution, weighted by probability.
        
        Args:
            f: Function from value to probability distribution
            
        Returns:
            New probability distribution
        """
        result = defaultdict(float)
        
        for value, prob in self.dist.items():
            new_dist = f(value)
            for new_val, new_prob in new_dist.dist.items():
                result[new_val] += prob * new_prob
        
        return Prob(dict(result))
    
    def map(self, f: Callable[[T], U]) -> 'Prob[U]':
        """
        Apply a function to all values in the distribution.
        
        Args:
            f: Function to apply to values
            
        Returns:
            New probability distribution
        """
        result = defaultdict(float)
        for value, prob in self.dist.items():
            result[f(value)] += prob
        return Prob(dict(result))
    
    def __repr__(self):
        items = sorted(self.dist.items(), key=lambda x: -x[1])
        return f"Prob({dict(items)})"
    
    # Utility methods
    
    def expected_value(self, f: Callable[[T], float] = lambda x: float(x)) -> float:
        """Calculate expected value of the distribution."""
        return sum(f(val) * prob for val, prob in self.dist.items())
    
    def sample(self) -> T:
        """Sample a value from the distribution."""
        values = list(self.dist.keys())
        probs = list(self.dist.values())
        return random.choices(values, weights=probs)[0]
    
    def filter(self, predicate: Callable[[T], bool]) -> 'Prob[T]':
        """Filter distribution keeping only values satisfying predicate."""
        filtered = {v: p for v, p in self.dist.items() if predicate(v)}
        return Prob(filtered)


# Convenience functions for creating common distributions

def uniform(values: list) -> Prob:
    """Create uniform distribution over given values."""
    if not values:
        return Prob({})
    prob = 1.0 / len(values)
    return Prob({v: prob for v in values})

def weighted(pairs: list[Tuple[Any, float]]) -> Prob:
    """Create distribution from (value, weight) pairs."""
    return Prob(dict(pairs))

def bernoulli(p: float, true_val=True, false_val=False) -> Prob:
    """Create Bernoulli distribution."""
    return Prob({true_val: p, false_val: 1-p})


# Example usage demonstrating monadic operations

if __name__ == "__main__":
    # Example 1: Fair coin
    coin = uniform(['H', 'T'])
    print("Fair coin:", coin)
    
    # Example 2: Die
    die = uniform(list(range(1,7)))
    print("Die:", die)
    
    # Example 3: Two coin flips using monadic bind
    two_flips = coin.bind(lambda x1: 
                coin.bind(lambda x2: 
                Prob.unit((x1, x2))))
    print("Two coin flips:", two_flips)
    
    # Example 4: Sum of two dice using bind
    two_dice_sum = die.bind(lambda d1:
                   die.bind(lambda d2:
                   Prob.unit(d1 + d2)))
    print("Sum of two dice:", two_dice_sum)
    
    # Example 5: Using map
    squared_die = die.map(lambda x: x ** 2)
    print("Squared die values:", squared_die)
    
    # Example 6: Conditional probability - at least one head in two flips
    at_least_one_head = two_flips.filter(lambda pair: 'H' in pair)
    print("At least one head in two flips:", at_least_one_head)
    
    # Example 7: Expected value
    print("Expected value of die:", die.expected_value())
    
    # Example 8: Sampling
    print("5 samples from die:", [die.sample() for _ in range(5)])
    

# pip install pymonad
from pymonad.monad import Monad
from typing import TypeVar, Callable, Set, FrozenSet, Any, Union
import random

T = TypeVar('T')
U = TypeVar('U')

class Powerset(Monad):
    """
    Powerset Monad for Non-deterministic Computation
    
    Represents non-deterministic computations as sets of possible values.
    Based on the mathematical definition:
    - TX = PX (powerset of X)
    - ηX(x) = {x} (unit)
    - f*(A) = ∪a∈A f(a) for f : X → PY, A ⊆ X (Kleisli extension)
    """
    
    def __init__(self, values: Union[Set[T], FrozenSet[T], list]):
        """
        Initialize a powerset computation.
        
        Args:
            values: Set, frozenset, or list of values representing possible outcomes
        """
        if isinstance(values, (set, frozenset)):
            self.values = frozenset(values)
        else:
            self.values = frozenset(values)
    
    @classmethod
    def unit(cls, value: T) -> 'Powerset':
        """
        Create a powerset with a single value (deterministic computation).
        Implements ηX(x) = {x}
        
        Args:
            value: The single value to wrap
            
        Returns:
            Powerset containing only the given value
        """
        return cls({value})
    
    def bind(self, f: Callable[[T], 'Powerset']) -> 'Powerset':
        """
        Monadic bind operation (Kleisli extension).
        Implements f*(A) = ∪a∈A f(a)
        
        Apply a function that returns a powerset to each value in this powerset,
        then take the union of all results.
        
        Args:
            f: Function from value to powerset
            
        Returns:
            New powerset containing union of all results
        """
        result = set()
        
        for value in self.values:
            new_powerset = f(value)
            result.update(new_powerset.values)
        
        return Powerset(result)
    
    def map(self, f: Callable[[T], U]) -> 'Powerset':
        """
        Apply a function to all values in the powerset.
        
        Args:
            f: Function to apply to values
            
        Returns:
            New powerset with function applied to all values
        """
        return Powerset({f(value) for value in self.values})
    
    def __repr__(self):
        sorted_values = sorted(self.values, key=str)
        return f"Powerset({set(sorted_values)})"
    
    def __eq__(self, other):
        if not isinstance(other, Powerset):
            return False
        return self.values == other.values
    
    def __hash__(self):
        return hash(self.values)
    
    # Utility methods
    
    def size(self) -> int:
        """Return the number of possible values."""
        return len(self.values)
    
    def is_empty(self) -> bool:
        """Check if the powerset is empty (failed computation)."""
        return len(self.values) == 0
    
    def is_deterministic(self) -> bool:
        """Check if this represents a deterministic computation (single value)."""
        return len(self.values) == 1
    
    def sample(self) -> T:
        """Sample a random value from the powerset."""
        if self.is_empty():
            raise ValueError("Cannot sample from empty powerset")
        return random.choice(list(self.values))
    
    def filter(self, predicate: Callable[[T], bool]) -> 'Powerset':
        """Filter powerset keeping only values satisfying predicate."""
        filtered = {v for v in self.values if predicate(v)}
        return Powerset(filtered)
    
    def union(self, other: 'Powerset') -> 'Powerset':
        """Union with another powerset."""
        return Powerset(self.values | other.values)
    
    def intersection(self, other: 'Powerset') -> 'Powerset':
        """Intersection with another powerset."""
        return Powerset(self.values & other.values)
    
    def difference(self, other: 'Powerset') -> 'Powerset':
        """Difference with another powerset."""
        return Powerset(self.values - other.values)


# Convenience functions for creating common powersets

def singleton(value: T) -> Powerset:
    """Create a powerset with a single value."""
    return Powerset.unit(value)

def empty() -> Powerset:
    """Create an empty powerset (failed computation)."""
    return Powerset(set())

def from_list(values: list) -> Powerset:
    """Create powerset from a list of values."""
    return Powerset(values)

def choice(*options) -> Powerset:
    """Create powerset representing a choice between options."""
    return Powerset(options)

def maybe(value: T) -> Powerset:
    """Create powerset representing optional value (success or failure)."""
    return Powerset({value, None})


# Example usage demonstrating non-deterministic computations

if __name__ == "__main__":
    # Example 1: Simple non-deterministic choice
    coin_flip = choice('H', 'T')
    print("Coin flip:", coin_flip)
    
    # Example 2: Deterministic computation
    certain = singleton(42)
    print("Certain value:", certain)
    
    # Example 3: Non-deterministic computation using bind
    # Choose a number, then add 1 or 2 to it
    numbers = choice(1, 2, 3)
    incremented = numbers.bind(lambda x: choice(x + 1, x + 2))
    print("Numbers incremented by 1 or 2:", incremented)
    
    # Example 4: Chaining non-deterministic computations
    # Start with a choice, square it, then add 1 or subtract 1
    computation = (choice(2, 3)
                  .bind(lambda x: singleton(x * x))
                  .bind(lambda x: choice(x + 1, x - 1)))
    print("Complex computation:", computation)
    
    # Example 5: Using map for deterministic transformation
    doubled = choice(1, 2, 3).map(lambda x: x * 2)
    print("Doubled values:", doubled)
    
    # Example 6: Filtering non-deterministic results
    large_numbers = choice(1, 5, 10, 15).filter(lambda x: x > 7)
    print("Large numbers only:", large_numbers)
    
    # Example 7: Combining powersets
    set1 = choice('a', 'b')
    set2 = choice('c', 'd')
    combined = set1.bind(lambda x: set2.bind(lambda y: singleton((x, y))))
    print("Cartesian product:", combined)
    
    # Example 8: Maybe computation (success or failure)
    maybe_value = maybe(100)
    safe_division = maybe_value.bind(lambda x: singleton(x / 2) if x is not None else empty())
    print("Maybe computation:", safe_division)
    
    # Example 9: Sampling from non-deterministic computation
    random_samples = [choice(1, 2, 3, 4, 5).sample() for _ in range(10)]
    print("Random samples:", random_samples)
    
    # Example 10: Set operations
    set_a = choice(1, 2, 3)
    set_b = choice(2, 3, 4)
    print("Union:", set_a.union(set_b))
    print("Intersection:", set_a.intersection(set_b))
    print("Difference:", set_a.difference(set_b)) 
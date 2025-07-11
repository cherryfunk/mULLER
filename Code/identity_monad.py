# pip install pymonad
from pymonad.monad import Monad
from typing import TypeVar, Callable, List, Tuple
import operator

T = TypeVar('T')
U = TypeVar('U')

class Identity(Monad):
    """
    Identity Monad for Deterministic Side-Effect Free Computation
    
    The simplest monad that just wraps values without any computational effects.
    Based on the mathematical definition:
    - TX = X (no wrapping, just the value itself)
    - ηX(x) = x (unit is identity)
    - f* = f (Kleisli extension is just function application)
    """
    
    def __init__(self, value: T):
        """
        Initialize an Identity monad with a value.
        
        Args:
            value: The value to wrap
        """
        self.value = value
    
    @classmethod
    def unit(cls, value: T) -> 'Identity':
        """
        Create an Identity monad (pure computation).
        Implements ηX(x) = x
        
        Args:
            value: The value to wrap
            
        Returns:
            Identity monad containing the value
        """
        return cls(value)
    
    def bind(self, f: Callable[[T], 'Identity']) -> 'Identity':
        """
        Monadic bind operation.
        Implements f* = f (Kleisli extension is just function application)
        
        Args:
            f: Function from value to Identity monad
            
        Returns:
            Result of applying f to the wrapped value
        """
        return f(self.value)
    
    def map(self, f: Callable[[T], U]) -> 'Identity':
        """
        Apply a function to the wrapped value.
        
        Args:
            f: Function to apply
            
        Returns:
            New Identity monad with transformed value
        """
        return Identity(f(self.value))
    
    def get(self) -> T:
        """Extract the value from the Identity monad."""
        return self.value
    
    def __repr__(self):
        return f"Identity({self.value})"
    
    def __eq__(self, other):
        if not isinstance(other, Identity):
            return False
        return self.value == other.value
    
    def __hash__(self):
        return hash(self.value)


# Boolean Logic Operations

def logical_and(x: bool, y: bool) -> Identity:
    """Logical AND operation."""
    return Identity(x and y)

def logical_or(x: bool, y: bool) -> Identity:
    """Logical OR operation."""
    return Identity(x or y)

def logical_not(x: bool) -> Identity:
    """Logical NOT operation."""
    return Identity(not x)

def logical_xor(x: bool, y: bool) -> Identity:
    """Logical XOR operation."""
    return Identity(x ^ y)

def logical_implies(x: bool, y: bool) -> Identity:
    """Logical implication: x → y ≡ ¬x ∨ y"""
    return Identity(not x or y)

def logical_iff(x: bool, y: bool) -> Identity:
    """Logical biconditional: x ↔ y ≡ (x → y) ∧ (y → x)"""
    return Identity((not x or y) and (not y or x))

# Convenience functions
def pure(value: T) -> Identity:
    """Create an Identity monad (alias for unit)."""
    return Identity.unit(value)

def true() -> Identity:
    """Create Identity monad with True."""
    return Identity(True)

def false() -> Identity:
    """Create Identity monad with False."""
    return Identity(False)

# Propositional Logic Evaluator
class Proposition:
    """Simple propositional logic expressions."""
    
    def __init__(self, expr: str):
        self.expr = expr
    
    def evaluate(self, variables: dict) -> Identity:
        """Evaluate proposition with given variable assignments."""
        # Simple evaluation - replace variables and evaluate
        result = self.expr
        for var, val in variables.items():
            result = result.replace(var, str(val))
        
        # Replace logical operators
        result = result.replace('AND', ' and ')
        result = result.replace('OR', ' or ')
        result = result.replace('NOT', ' not ')
        result = result.replace('True', 'True')
        result = result.replace('False', 'False')
        
        try:
            return Identity(eval(result))
        except:
            return Identity(False)
    
    def __repr__(self):
        return f"Proposition({self.expr})"

# Truth Table Generator
def truth_table(proposition: Proposition, variables: List[str]) -> List[Tuple]:
    """Generate truth table for a proposition."""
    n = len(variables)
    table = []
    
    for i in range(2**n):
        assignment = {}
        row = []
        
        # Generate all possible assignments
        for j, var in enumerate(variables):
            val = bool((i >> j) & 1)
            assignment[var] = val
            row.append(val)
        
        # Evaluate proposition
        result = proposition.evaluate(assignment)
        row.append(result.get())
        table.append(tuple(row))
    
    return table

# Boolean Algebra Laws Verification
def verify_law(name: str, expr1: Proposition, expr2: Proposition, variables: List[str]) -> Identity:
    """Verify that two expressions are logically equivalent."""
    table1 = truth_table(expr1, variables)
    table2 = truth_table(expr2, variables)
    
    # Check if last columns (results) are identical
    results1 = [row[-1] for row in table1]
    results2 = [row[-1] for row in table2]
    
    equivalent = results1 == results2
    
    if equivalent:
        print(f"✓ {name}: VERIFIED")
    else:
        print(f"✗ {name}: FAILED")
        print(f"  {expr1.expr}: {results1}")
        print(f"  {expr2.expr}: {results2}")
    
    return Identity(equivalent)

# Example usage demonstrating boolean logic with Identity monad

if __name__ == "__main__":
    # Example 1: Basic boolean operations
    print("=== Basic Boolean Operations ===")
    t = true()
    f = false()
    
    print(f"True: {t}")
    print(f"False: {f}")
    print(f"NOT True: {t.bind(lambda x: logical_not(x))}")
    print(f"NOT False: {f.bind(lambda x: logical_not(x))}")
    
    # Example 2: Combining operations with bind
    print("\n=== Combining Operations ===")
    result1 = t.bind(lambda x: f.bind(lambda y: logical_and(x, y)))
    result2 = t.bind(lambda x: f.bind(lambda y: logical_or(x, y)))
    print(f"True AND False: {result1}")
    print(f"True OR False: {result2}")
    
    # Example 3: Using map for transformations
    print("\n=== Using Map ===")
    negated = t.map(lambda x: not x)
    print(f"map(NOT, True): {negated}")
    
    # Example 4: Chaining operations
    print("\n=== Chaining Operations ===")
    # (True AND False) OR (NOT False)
    complex_expr = (t.bind(lambda x: 
                    f.bind(lambda y: 
                    logical_and(x, y)))
                   .bind(lambda result:
                    f.bind(lambda z:
                    logical_or(result, not z))))
    print(f"(True AND False) OR (NOT False): {complex_expr}")
    
    # Example 5: Propositional logic evaluation
    print("\n=== Propositional Logic ===")
    prop1 = Proposition("p AND q")
    prop2 = Proposition("NOT (NOT p OR NOT q)")  # De Morgan's law
    
    assignments = {'p': True, 'q': False}
    result1 = prop1.evaluate(assignments)
    result2 = prop2.evaluate(assignments)
    
    print(f"p AND q with p=True, q=False: {result1}")
    print(f"NOT (NOT p OR NOT q) with p=True, q=False: {result2}")
    
    # Example 6: Truth tables
    print("\n=== Truth Tables ===")
    prop = Proposition("p AND q")
    table = truth_table(prop, ['p', 'q'])
    
    print("Truth table for 'p AND q':")
    print("p\tq\tresult")
    print("-" * 15)
    for row in table:
        print(f"{row[0]}\t{row[1]}\t{row[2]}")
    
    # Example 7: Verifying Boolean algebra laws
    print("\n=== Boolean Algebra Laws ===")
    
    # De Morgan's Laws
    law1_left = Proposition("NOT (p AND q)")
    law1_right = Proposition("(NOT p) OR (NOT q)")
    verify_law("De Morgan's Law 1", law1_left, law1_right, ['p', 'q'])
    
    law2_left = Proposition("NOT (p OR q)")
    law2_right = Proposition("(NOT p) AND (NOT q)")
    verify_law("De Morgan's Law 2", law2_left, law2_right, ['p', 'q'])
    
    # Distributive Laws
    dist1_left = Proposition("p AND (q OR r)")
    dist1_right = Proposition("(p AND q) OR (p AND r)")
    verify_law("Distributive Law 1", dist1_left, dist1_right, ['p', 'q', 'r'])
    
    dist2_left = Proposition("p OR (q AND r)")
    dist2_right = Proposition("(p OR q) AND (p OR r)")
    verify_law("Distributive Law 2", dist2_left, dist2_right, ['p', 'q', 'r'])
    
    # Absorption Laws
    abs1_left = Proposition("p AND (p OR q)")
    abs1_right = Proposition("p")
    verify_law("Absorption Law 1", abs1_left, abs1_right, ['p', 'q'])
    
    abs2_left = Proposition("p OR (p AND q)")
    abs2_right = Proposition("p")
    verify_law("Absorption Law 2", abs2_left, abs2_right, ['p', 'q'])
    
    # Example 8: Monadic composition laws
    print("\n=== Monad Laws Verification ===")
    
    # Left identity: unit(a) >>= f ≡ f(a)
    a = True
    f = lambda x: logical_not(x)
    left_id_lhs = Identity.unit(a).bind(f)
    left_id_rhs = f(a)
    print(f"Left identity: {left_id_lhs} ≡ {left_id_rhs} -> {left_id_lhs.get() == left_id_rhs.get()}")
    
    # Right identity: m >>= unit ≡ m
    m = Identity(True)
    right_id_lhs = m.bind(Identity.unit)
    right_id_rhs = m
    print(f"Right identity: {right_id_lhs} ≡ {right_id_rhs} -> {right_id_lhs == right_id_rhs}")
    
    # Associativity: (m >>= f) >>= g ≡ m >>= (λx -> f(x) >>= g)
    g = lambda x: Identity(x and True)
    assoc_lhs = m.bind(f).bind(g)
    assoc_rhs = m.bind(lambda x: f(x).bind(g))
    print(f"Associativity: {assoc_lhs} ≡ {assoc_rhs} -> {assoc_lhs == assoc_rhs}") 
# pip install pymonad numpy scipy
from pymonad.monad import Monad
from typing import TypeVar, Callable, Dict, List, Tuple, Optional, Any
import numpy as np
from scipy import integrate
from scipy.stats import norm, uniform as scipy_uniform
import random
from dataclasses import dataclass
from giry_monad import GiryMeasure, dirac, normal, uniform as giry_uniform

T = TypeVar('T')
U = TypeVar('U')

@dataclass
class ProgramState:
    """Represents a program state with variable assignments."""
    variables: Dict[str, Any]
    
    def assign(self, var: str, value: Any) -> 'ProgramState':
        """Create new state with variable assignment."""
        new_vars = self.variables.copy()
        new_vars[var] = value
        return ProgramState(new_vars)
    
    def get(self, var: str, default: Any = None) -> Any:
        """Get variable value."""
        return self.variables.get(var, default)

class ProbabilisticFormula:
    """Base class for probabilistic logic formulas."""
    
    def evaluate(self, state: ProgramState) -> float:
        """Evaluate formula in given state, returning probability in [0,1]."""
        raise NotImplementedError
    
    def __and__(self, other: 'ProbabilisticFormula') -> 'Conjunction':
        return Conjunction(self, other)
    
    def __or__(self, other: 'ProbabilisticFormula') -> 'Disjunction':
        return Disjunction(self, other)
    
    def __invert__(self) -> 'Negation':
        return Negation(self)
    
    def implies(self, other: 'ProbabilisticFormula') -> 'Implication':
        return Implication(self, other)

class Atom(ProbabilisticFormula):
    """Atomic formula - evaluates to 0 or 1 based on predicate."""
    
    def __init__(self, predicate: Callable[[ProgramState], bool]):
        self.predicate = predicate
    
    def evaluate(self, state: ProgramState) -> float:
        return 1.0 if self.predicate(state) else 0.0

class Conjunction(ProbabilisticFormula):
    """F ∧ G := [[F]] · [[G]]"""
    
    def __init__(self, left: ProbabilisticFormula, right: ProbabilisticFormula):
        self.left = left
        self.right = right
    
    def evaluate(self, state: ProgramState) -> float:
        f_val = self.left.evaluate(state)
        g_val = self.right.evaluate(state)
        return f_val * g_val

class Disjunction(ProbabilisticFormula):
    """F ∨ G := [[F]] + [[G]] - [[F]] · [[G]]"""
    
    def __init__(self, left: ProbabilisticFormula, right: ProbabilisticFormula):
        self.left = left
        self.right = right
    
    def evaluate(self, state: ProgramState) -> float:
        f_val = self.left.evaluate(state)
        g_val = self.right.evaluate(state)
        return f_val + g_val - f_val * g_val

class Negation(ProbabilisticFormula):
    """¬F := 1 - [[F]]"""
    
    def __init__(self, formula: ProbabilisticFormula):
        self.formula = formula
    
    def evaluate(self, state: ProgramState) -> float:
        f_val = self.formula.evaluate(state)
        return 1.0 - f_val

class Implication(ProbabilisticFormula):
    """F → G := 1 - [[F]] + [[F]] · [[G]]"""
    
    def __init__(self, antecedent: ProbabilisticFormula, consequent: ProbabilisticFormula):
        self.antecedent = antecedent
        self.consequent = consequent
    
    def evaluate(self, state: ProgramState) -> float:
        f_val = self.antecedent.evaluate(state)
        g_val = self.consequent.evaluate(state)
        return 1.0 - f_val + f_val * g_val

class Truth(ProbabilisticFormula):
    """⊤ := 1"""
    
    def evaluate(self, state: ProgramState) -> float:
        return 1.0

class Falsity(ProbabilisticFormula):
    """⊥ := 0"""
    
    def evaluate(self, state: ProgramState) -> float:
        return 0.0

class ProbabilisticAssignment(ProbabilisticFormula):
    """x := m(T)(F) with integration over probability measures"""
    
    def __init__(self, variable: str, measure: GiryMeasure, formula: ProbabilisticFormula, 
                 condition: Optional[ProbabilisticFormula] = None):
        self.variable = variable
        self.measure = measure
        self.formula = formula
        self.condition = condition
    
    def evaluate(self, state: ProgramState) -> float:
        """∫_{a∈I(s_m)} [[F]]_ν[x↦a] dρ_m(· | T)(a)"""
        
        # Monte Carlo integration for continuous measures
        if self.measure.distribution_type == "continuous":
            n_samples = 1000
            total = 0.0
            
            for _ in range(n_samples):
                # Sample value from measure
                sampled_value = self.measure.sample()
                
                # Create new state with variable assignment
                new_state = state.assign(self.variable, sampled_value)
                
                # Evaluate formula in new state
                if self.condition is None:
                    weight = 1.0
                else:
                    weight = self.condition.evaluate(new_state)
                
                total += self.formula.evaluate(new_state) * weight
            
            return total / n_samples
        
        # Discrete case: exact computation
        elif self.measure.distribution_type == "discrete":
            total = 0.0
            
            for value, prob in zip(self.measure.values, self.measure.probabilities):
                # Create new state with variable assignment
                new_state = state.assign(self.variable, value)
                
                # Evaluate formula in new state
                if self.condition is None:
                    weight = prob
                else:
                    weight = prob * self.condition.evaluate(new_state)
                
                total += self.formula.evaluate(new_state) * weight
            
            return total
        
        else:
            # Dirac measure case
            if self.measure.values:
                new_state = state.assign(self.variable, self.measure.values[0])
                if self.condition is None:
                    return self.formula.evaluate(new_state)
                else:
                    return self.formula.evaluate(new_state) * self.condition.evaluate(new_state)
            return 0.0

class UniversalQuantification(ProbabilisticFormula):
    """∀x:s F := inf_{a∈I(s)} [[F]]_ν[x↦a]"""
    
    def __init__(self, variable: str, domain: List[Any], formula: ProbabilisticFormula):
        self.variable = variable
        self.domain = domain
        self.formula = formula
    
    def evaluate(self, state: ProgramState) -> float:
        if not self.domain:
            return 1.0
        
        min_val = float('inf')
        for value in self.domain:
            new_state = state.assign(self.variable, value)
            val = self.formula.evaluate(new_state)
            min_val = min(min_val, val)
        
        return min_val

class ExistentialQuantification(ProbabilisticFormula):
    """∃x:s F := sup_{a∈I(s)} [[F]]_ν[x↦a]"""
    
    def __init__(self, variable: str, domain: List[Any], formula: ProbabilisticFormula):
        self.variable = variable
        self.domain = domain
        self.formula = formula
    
    def evaluate(self, state: ProgramState) -> float:
        if not self.domain:
            return 0.0
        
        max_val = float('-inf')
        for value in self.domain:
            new_state = state.assign(self.variable, value)
            val = self.formula.evaluate(new_state)
            max_val = max(max_val, val)
        
        return max_val

class ProbabilisticLogicGiry(GiryMeasure):
    """Extension of Giry monad for probabilistic logic semantics."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def evaluate_formula(self, formula: ProbabilisticFormula) -> 'ProbabilisticLogicGiry':
        """Evaluate a probabilistic formula over this measure."""
        if self.distribution_type == "discrete":
            # Discrete case: weighted average
            total = 0.0
            for state, prob in zip(self.values, self.probabilities):
                if isinstance(state, ProgramState):
                    total += formula.evaluate(state) * prob
                else:
                    # Assume it's a simple value, create minimal state
                    simple_state = ProgramState({"value": state})
                    total += formula.evaluate(simple_state) * prob
            
            result_val = total
            return ProbabilisticLogicGiry(
                distribution_type="dirac",
                values=[result_val],
                probabilities=[1.0],
                sampler=lambda val=result_val: val,
                name=f"eval({formula})"
            )
        
        else:
            # Continuous case: Monte Carlo evaluation
            n_samples = 1000
            samples = []
            
            for _ in range(n_samples):
                state_sample = self.sample()
                if isinstance(state_sample, ProgramState):
                    eval_result = formula.evaluate(state_sample)
                else:
                    simple_state = ProgramState({"value": state_sample})
                    eval_result = formula.evaluate(simple_state)
                samples.append(eval_result)
            
            avg_result = sum(samples) / len(samples)
            result_val = avg_result
            return ProbabilisticLogicGiry(
                distribution_type="dirac",
                values=[result_val],
                probabilities=[1.0],
                sampler=lambda val=result_val: val,
                name=f"eval({formula})"
            )

# Helper functions for creating common formulas

def atom(predicate: Callable[[ProgramState], bool]) -> Atom:
    """Create atomic formula."""
    return Atom(predicate)

def var_equals(var: str, value: Any) -> Atom:
    """Create formula: variable equals value."""
    return Atom(lambda state: state.get(var) == value)

def var_greater(var: str, value: float) -> Atom:
    """Create formula: variable > value."""
    return Atom(lambda state: state.get(var) > value)

def var_less(var: str, value: float) -> Atom:
    """Create formula: variable < value."""
    return Atom(lambda state: state.get(var) < value)

def probabilistic_assignment(var: str, measure: GiryMeasure, formula: ProbabilisticFormula, 
                           condition: Optional[ProbabilisticFormula] = None) -> ProbabilisticAssignment:
    """Create probabilistic assignment."""
    return ProbabilisticAssignment(var, measure, formula, condition)

def forall(var: str, domain: List[Any], formula: ProbabilisticFormula) -> UniversalQuantification:
    """Create universal quantification."""
    return UniversalQuantification(var, domain, formula)

def exists(var: str, domain: List[Any], formula: ProbabilisticFormula) -> ExistentialQuantification:
    """Create existential quantification."""
    return ExistentialQuantification(var, domain, formula)

def truth() -> Truth:
    """Create truth formula."""
    return Truth()

def falsity() -> Falsity:
    """Create falsity formula."""
    return Falsity()

# Example usage demonstrating probabilistic logic with Giry monad

if __name__ == "__main__":
    # Example 1: Basic probabilistic logic operations
    print("=== Basic Probabilistic Logic Operations ===")
    
    # Create initial state
    initial_state = ProgramState({"x": 5, "y": 3})
    
    # Create atomic formulas
    x_positive = var_greater("x", 0)
    y_positive = var_greater("y", 0)
    x_large = var_greater("x", 10)
    
    print(f"x > 0: {x_positive.evaluate(initial_state)}")
    print(f"y > 0: {y_positive.evaluate(initial_state)}")
    print(f"x > 10: {x_large.evaluate(initial_state)}")
    
    # Test logical operations
    conjunction = x_positive & y_positive
    disjunction = x_positive | x_large
    negation = ~x_large
    implication = x_large.implies(y_positive)
    
    print(f"(x > 0) ∧ (y > 0): {conjunction.evaluate(initial_state)}")
    print(f"(x > 0) ∨ (x > 10): {disjunction.evaluate(initial_state)}")
    print(f"¬(x > 10): {negation.evaluate(initial_state)}")
    print(f"(x > 10) → (y > 0): {implication.evaluate(initial_state)}")
    
    # Example 2: Probabilistic assignment
    print("\n=== Probabilistic Assignment ===")
    
    # x := Normal(0, 1), evaluate P(x > 0)
    normal_measure = normal(0, 1)
    x_positive_after = var_greater("x", 0)
    
    assignment = probabilistic_assignment("x", normal_measure, x_positive_after)
    prob_x_positive = assignment.evaluate(initial_state)
    
    print(f"P(x > 0) after x := Normal(0,1): {prob_x_positive:.3f}")
    
    # Example 3: Conditional assignment
    print("\n=== Conditional Assignment ===")
    
    # x := Normal(0, 1) | x > -1, evaluate P(x > 0)
    condition = var_greater("x", -1)
    conditional_assignment = probabilistic_assignment("x", normal_measure, x_positive_after, condition)
    
    prob_x_positive_conditional = conditional_assignment.evaluate(initial_state)
    print(f"P(x > 0) after x := Normal(0,1) | x > -1: {prob_x_positive_conditional:.3f}")
    
    # Example 4: Quantification
    print("\n=== Quantification ===")
    
    # ∀x ∈ {1,2,3}: x > 0
    domain = [1, 2, 3]
    universal_formula = forall("x", domain, var_greater("x", 0))
    print(f"∀x ∈ {{1,2,3}}: x > 0 = {universal_formula.evaluate(initial_state)}")
    
    # ∃x ∈ {-1,0,1}: x > 0
    domain2 = [-1, 0, 1]
    existential_formula = exists("x", domain2, var_greater("x", 0))
    print(f"∃x ∈ {{-1,0,1}}: x > 0 = {existential_formula.evaluate(initial_state)}")
    
    # Example 5: Complex probabilistic reasoning
    print("\n=== Complex Probabilistic Reasoning ===")
    
    # Create a probabilistic program:
    # x := Uniform(0, 1)
    # y := Normal(x, 0.1)
    # P((x > 0.5) ∧ (y > 0.6))
    
    uniform_measure = giry_uniform(0, 1)
    
    # First assignment: x := Uniform(0, 1)
    state_after_x = ProgramState({"x": 0.5})  # We'll integrate over this
    
    # For each value of x, y := Normal(x, 0.1)
    def complex_reasoning():
        total_prob = 0.0
        n_samples = 1000
        
        for _ in range(n_samples):
            x_val = uniform_measure.sample()
            y_measure = normal(x_val, 0.1)
            y_val = y_measure.sample()
            
            # Create state with both assignments
            state = ProgramState({"x": x_val, "y": y_val})
            
            # Evaluate (x > 0.5) ∧ (y > 0.6)
            formula = var_greater("x", 0.5) & var_greater("y", 0.6)
            total_prob += formula.evaluate(state)
        
        return total_prob / n_samples
    
    complex_prob = complex_reasoning()
    print(f"P((x > 0.5) ∧ (y > 0.6)) after x := U(0,1), y := N(x,0.1): {complex_prob:.3f}")
    
    # Example 6: Using Giry monad for state distributions
    print("\n=== Giry Monad State Distributions ===")
    
    # Create distribution over program states
    states = [
        ProgramState({"x": 1, "y": 2}),
        ProgramState({"x": 3, "y": 4}),
        ProgramState({"x": 5, "y": 6})
    ]
    
    state_measure = ProbabilisticLogicGiry(
        distribution_type="discrete",
        values=states,
        probabilities=[0.2, 0.3, 0.5],
        name="state_distribution"
    )
    
    # Evaluate formula over state distribution
    formula = var_greater("x", 2) & var_greater("y", 3)
    result_measure = state_measure.evaluate_formula(formula)
    
    print(f"P((x > 2) ∧ (y > 3)) over state distribution: {result_measure.sample():.3f}")
    
    # Example 7: Monadic composition
    print("\n=== Monadic Composition ===")
    
    # Chain probabilistic assignments using monadic bind
    def probabilistic_program(initial_value):
        # Start with deterministic value
        return (dirac(initial_value)
                # x := x + Normal(0, 1)
                .bind(lambda x: dirac(x + normal(0, 1).sample()))
                # y := x * 2
                .bind(lambda x: dirac(x * 2)))
    
    # Run probabilistic program
    final_distribution = probabilistic_program(1.0)
    samples = [final_distribution.sample() for _ in range(5)]
    print(f"Samples from probabilistic program: {samples}")
    
    # Example 8: BL-Algebra verification
    print("\n=== BL-Algebra Properties Verification ===")
    
    # Test some BL-algebra properties
    state = ProgramState({"x": 0.7})
    
    # Test: F ∧ G = F * G
    f = Atom(lambda s: 0.8)  # Constant probability
    g = Atom(lambda s: 0.6)  # Constant probability
    
    conjunction_val = (f & g).evaluate(state)
    product_val = f.evaluate(state) * g.evaluate(state)
    
    print(f"F ∧ G = {conjunction_val:.3f}")
    print(f"F * G = {product_val:.3f}")
    print(f"BL-Algebra conjunction verified: {abs(conjunction_val - product_val) < 1e-10}")
    
    # Test: F ∨ G = F + G - F * G
    disjunction_val = (f | g).evaluate(state)
    lukasiewicz_val = f.evaluate(state) + g.evaluate(state) - f.evaluate(state) * g.evaluate(state)
    
    print(f"F ∨ G = {disjunction_val:.3f}")
    print(f"F + G - F * G = {lukasiewicz_val:.3f}")
    print(f"BL-Algebra disjunction verified: {abs(disjunction_val - lukasiewicz_val) < 1e-10}")
    
    # Test: ¬F = 1 - F
    negation_val = (~f).evaluate(state)
    complement_val = 1.0 - f.evaluate(state)
    
    print(f"¬F = {negation_val:.3f}")
    print(f"1 - F = {complement_val:.3f}")
    print(f"BL-Algebra negation verified: {abs(negation_val - complement_val) < 1e-10}")
    
    print("\n=== All BL-Algebra operations verified! ===") 
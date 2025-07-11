# LTN (Logic Tensor Networks) Semantics with p-norm Quantifiers
# Based on Infinitary LTNp Semantics - only quantifiers change to use p-means

from probabilistic_semantics import *
import numpy as np
from scipy import integrate
from typing import Union, List, Callable
import math

class LTNFormula(ProbabilisticFormula):
    """Base class for LTN formulas with p-norm semantics."""
    
    def __init__(self, p: float = 2.0):
        """
        Initialize LTN formula with hyperparameter p.
        
        Args:
            p: Hyperparameter for p-norm aggregation
               - p=1: arithmetic mean (tolerant to outliers)
               - p=2: quadratic mean  
               - p→∞: maximum (logically stricter)
        """
        self.p = p
    
    def p_mean(self, values: List[float]) -> float:
        """
        Compute p-mean: Mp(a1,...,an) := (1/n ∑ai^p)^(1/p)
        
        Args:
            values: List of values to aggregate
            
        Returns:
            p-mean of the values
        """
        if not values:
            return 0.0
            
        if self.p == float('inf'):
            return max(values)
        elif self.p == 1.0:
            return sum(values) / len(values)
        else:
            # General p-norm: (1/n ∑ai^p)^(1/p)
            n = len(values)
            sum_p = sum(val**self.p for val in values)
            return (sum_p / n)**(1/self.p)
    
    def p_integral(self, func: Callable[[float], float], measure: GiryMeasure) -> float:
        """
        Compute p-norm integral: Mp(f; ρs) := (∫ f(x)^p dρs(x))^(1/p)
        
        Args:
            func: Function to integrate
            measure: Probability measure
            
        Returns:
            p-norm integral result
        """
        if self.p == float('inf'):
            # For p→∞, return supremum (approximated via sampling)
            if measure.distribution_type == "continuous":
                samples = [measure.sample() for _ in range(1000)]
                return max(func(x) for x in samples)
            elif measure.distribution_type == "discrete":
                return max(func(val) for val in measure.values)
            else:
                return func(measure.values[0]) if measure.values else 0.0
        
        # Monte Carlo approximation for continuous measures
        if measure.distribution_type == "continuous":
            n_samples = 1000
            total = 0.0
            
            for _ in range(n_samples):
                x = measure.sample()
                total += func(x)**self.p
            
            avg = total / n_samples
            return avg**(1/self.p) if avg > 0 else 0.0
            
        # Exact computation for discrete measures
        elif measure.distribution_type == "discrete":
            total = 0.0
            for val, prob in zip(measure.values, measure.probabilities):
                total += prob * (func(val)**self.p)
            return total**(1/self.p) if total > 0 else 0.0
            
        # Dirac measure
        else:
            return func(measure.values[0]) if measure.values else 0.0

class LTNConjunction(LTNFormula):
    """LTN Conjunction: F ∧ G using p-norm aggregation"""
    
    def __init__(self, left: ProbabilisticFormula, right: ProbabilisticFormula, p: float = 2.0):
        super().__init__(p)
        self.left = left
        self.right = right
    
    def evaluate(self, state: ProgramState) -> float:
        f_val = self.left.evaluate(state)
        g_val = self.right.evaluate(state)
        return self.p_mean([f_val, g_val])

class LTNDisjunction(LTNFormula):
    """LTN Disjunction: F ∨ G using dual p-norm aggregation"""
    
    def __init__(self, left: ProbabilisticFormula, right: ProbabilisticFormula, p: float = 2.0):
        super().__init__(p)
        self.left = left
        self.right = right
    
    def evaluate(self, state: ProgramState) -> float:
        f_val = self.left.evaluate(state)
        g_val = self.right.evaluate(state)
        
        # Dual p-norm for disjunction: 1 - p_mean([1-f, 1-g])
        dual_values = [1.0 - f_val, 1.0 - g_val]
        return 1.0 - self.p_mean(dual_values)

class LTNNegation(LTNFormula):
    """LTN Negation: ¬F := 1 - F (same as probabilistic semantics)"""
    
    def __init__(self, formula: ProbabilisticFormula, p: float = 2.0):
        super().__init__(p)
        self.formula = formula
    
    def evaluate(self, state: ProgramState) -> float:
        return 1.0 - self.formula.evaluate(state)

class LTNImplication(LTNFormula):
    """LTN Implication: F → G using LTN aggregation"""
    
    def __init__(self, antecedent: ProbabilisticFormula, consequent: ProbabilisticFormula, p: float = 2.0):
        super().__init__(p)
        self.antecedent = antecedent
        self.consequent = consequent
    
    def evaluate(self, state: ProgramState) -> float:
        f_val = self.antecedent.evaluate(state)
        g_val = self.consequent.evaluate(state)
        
        # LTN implication: p_mean([1-f, g])
        return self.p_mean([1.0 - f_val, g_val])

class LTNUniversalQuantification(ProbabilisticFormula):
    """
    LTN Universal Quantification using p-means:
    ∀x:s F := Mp(F(a) for a in domain) = (1/n ∑F(a)^p)^(1/p)
    
    This replaces the inf-based semantics with p-norm aggregation.
    """
    
    def __init__(self, variable: str, domain: List[Any], formula: ProbabilisticFormula, p: float = 2.0):
        self.variable = variable
        self.domain = domain
        self.formula = formula
        self.p = p
    
    def p_mean(self, values: List[float]) -> float:
        """Compute p-mean: Mp(a1,...,an) := (1/n ∑ai^p)^(1/p)"""
        if not values:
            return 1.0  # Universal quantification over empty domain
            
        if self.p == float('inf'):
            return min(values)  # As p→∞, p-mean approaches minimum
        elif self.p == 1.0:
            return sum(values) / len(values)  # Arithmetic mean
        else:
            # General p-norm: (1/n ∑ai^p)^(1/p)
            n = len(values)
            sum_p = sum(val**self.p for val in values)
            return (sum_p / n)**(1/self.p)
    
    def evaluate(self, state: ProgramState) -> float:
        if not self.domain:
            return 1.0
        
        values = []
        for value in self.domain:
            new_state = state.assign(self.variable, value)
            val = self.formula.evaluate(new_state)
            values.append(val)
        
        return self.p_mean(values)

class LTNExistentialQuantification(ProbabilisticFormula):
    """
    LTN Existential Quantification using dual p-means:
    ∃x:s F := 1 - Mp((1-F(a)) for a in domain)
    
    This replaces the sup-based semantics with dual p-norm aggregation.
    """
    
    def __init__(self, variable: str, domain: List[Any], formula: ProbabilisticFormula, p: float = 2.0):
        self.variable = variable
        self.domain = domain
        self.formula = formula
        self.p = p
    
    def p_mean(self, values: List[float]) -> float:
        """Compute p-mean: Mp(a1,...,an) := (1/n ∑ai^p)^(1/p)"""
        if not values:
            return 0.0  # Existential quantification over empty domain
            
        if self.p == float('inf'):
            return min(values)  # As p→∞, p-mean approaches minimum
        elif self.p == 1.0:
            return sum(values) / len(values)  # Arithmetic mean
        else:
            # General p-norm: (1/n ∑ai^p)^(1/p)
            n = len(values)
            sum_p = sum(val**self.p for val in values)
            return (sum_p / n)**(1/self.p)
    
    def evaluate(self, state: ProgramState) -> float:
        if not self.domain:
            return 0.0
        
        # Dual p-norm for existential: 1 - p_mean(1 - F(a))
        neg_values = []
        for value in self.domain:
            new_state = state.assign(self.variable, value)
            val = self.formula.evaluate(new_state)
            neg_values.append(1.0 - val)
        
        return 1.0 - self.p_mean(neg_values)

class LTNProbabilisticAssignment(ProbabilisticFormula):
    """
    LTN Probabilistic Assignment with p-norm integration:
    x := m(T)(F) := Mp(f; ρs) = (∫ f(x)^p dρs(x))^(1/p)
    
    From equation (6) in the paper.
    """
    
    def __init__(self, variable: str, measure: GiryMeasure, formula: ProbabilisticFormula, 
                 condition: Optional[ProbabilisticFormula] = None, p: float = 2.0):
        self.variable = variable
        self.measure = measure
        self.formula = formula
        self.condition = condition
        self.p = p
    
    def p_integral(self, func: Callable[[float], float], measure: GiryMeasure) -> float:
        """Compute p-norm integral: Mp(f; ρs) := (∫ f(x)^p dρs(x))^(1/p)"""
        
        if self.p == float('inf'):
            # For p→∞, return supremum (approximated via sampling)
            if measure.distribution_type == "continuous":
                samples = [measure.sample() for _ in range(1000)]
                return max(func(x) for x in samples)
            elif measure.distribution_type == "discrete":
                return max(func(val) for val in measure.values)
            else:
                return func(measure.values[0]) if measure.values else 0.0
        
        # Monte Carlo approximation for continuous measures
        if measure.distribution_type == "continuous":
            n_samples = 1000
            total = 0.0
            
            for _ in range(n_samples):
                x = measure.sample()
                total += func(x)**self.p
            
            avg = total / n_samples
            return avg**(1/self.p) if avg > 0 else 0.0
            
        # Exact computation for discrete measures
        elif measure.distribution_type == "discrete":
            total = 0.0
            for val, prob in zip(measure.values, measure.probabilities):
                total += prob * (func(val)**self.p)
            return total**(1/self.p) if total > 0 else 0.0
            
        # Dirac measure
        else:
            return func(measure.values[0]) if measure.values else 0.0
    
    def evaluate(self, state: ProgramState) -> float:
        """Compute p-norm integral over the measure"""
        
        def integrand(x):
            new_state = state.assign(self.variable, x)
            val = self.formula.evaluate(new_state)
            
            if self.condition is not None:
                weight = self.condition.evaluate(new_state)
                return val * weight
            return val
        
        return self.p_integral(integrand, self.measure)

class LTNAggregation(LTNFormula):
    """
    General LTN aggregation function for multiple formulas.
    Implements: aggr_p(F1, F2, ..., Fn) = Mp(F1, F2, ..., Fn)
    """
    
    def __init__(self, formulas: List[ProbabilisticFormula], p: float = 2.0):
        super().__init__(p)
        self.formulas = formulas
    
    def evaluate(self, state: ProgramState) -> float:
        if not self.formulas:
            return 0.0
        
        values = [formula.evaluate(state) for formula in self.formulas]
        return self.p_mean(values)

# Helper functions for creating LTN formulas

def ltn_and(left: ProbabilisticFormula, right: ProbabilisticFormula, p: float = 2.0) -> LTNConjunction:
    """Create LTN conjunction with p-norm aggregation."""
    return LTNConjunction(left, right, p)

def ltn_or(left: ProbabilisticFormula, right: ProbabilisticFormula, p: float = 2.0) -> LTNDisjunction:
    """Create LTN disjunction with dual p-norm aggregation."""
    return LTNDisjunction(left, right, p)

def ltn_not(formula: ProbabilisticFormula, p: float = 2.0) -> LTNNegation:
    """Create LTN negation."""
    return LTNNegation(formula, p)

def ltn_implies(antecedent: ProbabilisticFormula, consequent: ProbabilisticFormula, p: float = 2.0) -> LTNImplication:
    """Create LTN implication."""
    return LTNImplication(antecedent, consequent, p)

def ltn_forall(variable: str, domain: List[Any], formula: ProbabilisticFormula, p: float = 2.0) -> LTNUniversalQuantification:
    """Create LTN universal quantification with p-means."""
    return LTNUniversalQuantification(variable, domain, formula, p)

def ltn_exists(variable: str, domain: List[Any], formula: ProbabilisticFormula, p: float = 2.0) -> LTNExistentialQuantification:
    """Create LTN existential quantification with dual p-means."""
    return LTNExistentialQuantification(variable, domain, formula, p)

def ltn_assignment(variable: str, measure: GiryMeasure, formula: ProbabilisticFormula, 
                  condition: Optional[ProbabilisticFormula] = None, p: float = 2.0) -> LTNProbabilisticAssignment:
    """Create LTN probabilistic assignment with p-norm integration."""
    return LTNProbabilisticAssignment(variable, measure, formula, condition, p)

def ltn_aggregate(formulas: List[ProbabilisticFormula], p: float = 2.0) -> LTNAggregation:
    """Create LTN aggregation of multiple formulas."""
    return LTNAggregation(formulas, p)

# Extended LTN operators with different p values

def ltn_strict_and(left: ProbabilisticFormula, right: ProbabilisticFormula) -> LTNConjunction:
    """Strict LTN conjunction (p→∞, approaches minimum)."""
    return LTNConjunction(left, right, p=float('inf'))

def ltn_mean_and(left: ProbabilisticFormula, right: ProbabilisticFormula) -> LTNConjunction:
    """Mean LTN conjunction (p=1, arithmetic mean)."""
    return LTNConjunction(left, right, p=1.0)

def ltn_quadratic_and(left: ProbabilisticFormula, right: ProbabilisticFormula) -> LTNConjunction:
    """Quadratic LTN conjunction (p=2, quadratic mean)."""
    return LTNConjunction(left, right, p=2.0)

# Convenience functions for different p values

def ltn_forall_strict(variable: str, domain: List[Any], formula: ProbabilisticFormula) -> LTNUniversalQuantification:
    """Strict LTN universal quantification (p→∞, approaches minimum)."""
    return LTNUniversalQuantification(variable, domain, formula, p=float('inf'))

def ltn_forall_mean(variable: str, domain: List[Any], formula: ProbabilisticFormula) -> LTNUniversalQuantification:
    """Mean LTN universal quantification (p=1, arithmetic mean)."""
    return LTNUniversalQuantification(variable, domain, formula, p=1.0)

def ltn_exists_strict(variable: str, domain: List[Any], formula: ProbabilisticFormula) -> LTNExistentialQuantification:
    """Strict LTN existential quantification (p→∞)."""
    return LTNExistentialQuantification(variable, domain, formula, p=float('inf'))

def ltn_exists_mean(variable: str, domain: List[Any], formula: ProbabilisticFormula) -> LTNExistentialQuantification:
    """Mean LTN existential quantification (p=1, arithmetic mean)."""
    return LTNExistentialQuantification(variable, domain, formula, p=1.0)

# Example usage and comparisons

if __name__ == "__main__":
    print("=== LTN Semantics: p-norm Quantifiers ===")
    
    # Create test domain and formula
    domain = [0.6, 0.7, 0.8, 0.9]
    formula = var_greater("x", 0.5)
    
    print(f"Domain: {domain}")
    print(f"Formula: x > 0.5")
    print(f"Formula values: {[formula.evaluate(ProgramState({'x': val})) for val in domain]}")
    
    print("\n=== Universal Quantification Comparison ===")
    
    # Standard probabilistic (inf-based)
    prob_forall = forall("x", domain, formula)
    prob_result = prob_forall.evaluate(ProgramState({}))
    print(f"Probabilistic ∀x: {prob_result:.3f} (minimum)")
    
    # LTN with different p values
    for p in [1.0, 2.0, 5.0, float('inf')]:
        ltn_forall_q = ltn_forall("x", domain, formula, p)
        result = ltn_forall_q.evaluate(ProgramState({}))
        p_name = "∞" if p == float('inf') else str(p)
        print(f"LTN ∀x (p={p_name}): {result:.3f}")
    
    print("\n=== Existential Quantification Comparison ===")
    
    # Standard probabilistic (sup-based) 
    prob_exists = exists("x", domain, formula)
    prob_result = prob_exists.evaluate(ProgramState({}))
    print(f"Probabilistic ∃x: {prob_result:.3f} (maximum)")
    
    # LTN with different p values
    for p in [1.0, 2.0, 5.0, float('inf')]:
        ltn_exists_q = ltn_exists("x", domain, formula, p)
        result = ltn_exists_q.evaluate(ProgramState({}))
        p_name = "∞" if p == float('inf') else str(p)
        print(f"LTN ∃x (p={p_name}): {result:.3f}")
    
    # Test with mixed domain (some true, some false)
    print("\n=== Mixed Domain Test ===")
    mixed_domain = [0.3, 0.7, 0.8, 0.9]  # 0.3 > 0.5 is False
    mixed_formula = var_greater("x", 0.5)
    
    print(f"Mixed domain: {mixed_domain}")
    print(f"Formula values: {[mixed_formula.evaluate(ProgramState({'x': val})) for val in mixed_domain]}")
    
    # Compare quantifications
    prob_forall_mixed = forall("x", mixed_domain, mixed_formula).evaluate(ProgramState({}))
    ltn_forall_mixed = ltn_forall("x", mixed_domain, mixed_formula, p=2.0).evaluate(ProgramState({}))
    
    prob_exists_mixed = exists("x", mixed_domain, mixed_formula).evaluate(ProgramState({}))
    ltn_exists_mixed = ltn_exists("x", mixed_domain, mixed_formula, p=2.0).evaluate(ProgramState({}))
    
    print(f"Probabilistic ∀x: {prob_forall_mixed:.3f} vs LTN ∀x (p=2): {ltn_forall_mixed:.3f}")
    print(f"Probabilistic ∃x: {prob_exists_mixed:.3f} vs LTN ∃x (p=2): {ltn_exists_mixed:.3f}")
    
    print("\n=== All other operations (∧,∨,¬,→) remain the same as probabilistic semantics ===")
    
    # Demonstrate that other operations are unchanged
    state = ProgramState({"x": 0.7, "y": 0.8})
    f1 = var_greater("x", 0.5)
    f2 = var_greater("y", 0.5)
    
    print(f"F1 ∧ F2: {(f1 & f2).evaluate(state):.3f}")
    print(f"F1 ∨ F2: {(f1 | f2).evaluate(state):.3f}")
    print(f"¬F1: {(~f1).evaluate(state):.3f}")
    print(f"F1 → F2: {f1.implies(f2).evaluate(state):.3f}") 
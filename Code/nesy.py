from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Any, Union, Optional, Tuple
from enum import Enum
from collections import defaultdict
import random
from functools import reduce

# Using pymonad library for proper monadic structures
from pymonad.maybe import Maybe, Just, Nothing
from pymonad.list import ListMonad
from pymonad.tools import curry
from pymonad.monad import Monad

# ---------------------- Probability Distribution Monad ------------------------------

class Probability:
    """Probability distribution monad based on pymonad structure"""
    
    def __init__(self, probability_dict: Dict[Any, float]):
        """Initialize with a dictionary of {outcome: probability}"""
        # Normalize probabilities
        total = sum(probability_dict.values()) if probability_dict else 0
        if total > 0:
            self.probability = {k: v/total for k, v in probability_dict.items()}
        else:
            self.probability = probability_dict.copy()
    
    def __eq__(self, other):
        if not isinstance(other, Probability):
            return False
        return self.probability == other.probability
    
    def __repr__(self):
        return f"Probability({self.probability})"
    
    @classmethod
    def pure(cls, value):
        """Monadic return/pure - create a deterministic distribution"""
        return cls({value: 1.0})
    
    def map(self, func):
        """Functor map - apply function to all outcomes"""
        new_outcomes = {}
        for outcome, prob in self.probability.items():
            new_outcome = func(outcome)
            if new_outcome in new_outcomes:
                new_outcomes[new_outcome] += prob
            else:
                new_outcomes[new_outcome] = prob
        return Probability(new_outcomes)
    
    def apply(self, func_dist):
        """Applicative apply"""
        new_outcomes = {}
        for func, func_prob in func_dist.probability.items():
            for value, value_prob in self.probability.items():
                result = func(value)
                combined_prob = func_prob * value_prob
                if result in new_outcomes:
                    new_outcomes[result] += combined_prob
                else:
                    new_outcomes[result] = combined_prob
        return Probability(new_outcomes)
    
    def bind(self, func):
        """Monadic bind - probabilistic composition"""
        new_outcomes = {}
        for outcome, prob in self.probability.items():
            result_dist = func(outcome)
            if isinstance(result_dist, Probability):
                for new_outcome, new_prob in result_dist.probability.items():
                    combined_prob = prob * new_prob
                    if new_outcome in new_outcomes:
                        new_outcomes[new_outcome] += combined_prob
                    else:
                        new_outcomes[new_outcome] = combined_prob
            else:
                # If func doesn't return a Probability, treat as deterministic
                if outcome in new_outcomes:
                    new_outcomes[outcome] += prob
                else:
                    new_outcomes[outcome] = prob
        return Probability(new_outcomes)
    
    @classmethod
    def uniform(cls, values: List[Any]):
        """Create uniform distribution over values"""
        if not values:
            return cls({})
        prob = 1.0 / len(values)
        return cls({v: prob for v in values})
    
    @classmethod
    def from_freqs(cls, freq_pairs: List[Tuple[Any, float]]):
        """Create distribution from frequency pairs"""
        return cls(dict(freq_pairs))
    
    def sample(self):
        """Sample from the distribution"""
        if not self.probability:
            return None
        values, probs = zip(*self.probability.items())
        return random.choices(values, weights=probs)[0]
    
    def max_probability(self):
        """Get the maximum probability value"""
        return max(self.probability.values()) if self.probability else 0.0
    
    def argmax(self):
        """Get outcomes with maximum probability"""
        if not self.probability:
            return []
        max_prob = self.max_probability()
        return [k for k, v in self.probability.items() if v == max_prob]

# ---------------------- NeSy Frameworks ------------------------------

class Aggr2SGrpBLat(ABC):
    """Algebra of truth values - Abstract base class"""
    
    @abstractmethod
    def top(self):
        """Top element (true)"""
        pass
    
    @abstractmethod
    def bot(self):
        """Bottom element (false)"""
        pass
    
    @abstractmethod
    def neg(self, a):
        """Negation"""
        pass
    
    @abstractmethod
    def conj(self, a, b):
        """Conjunction (and)"""
        pass
    
    @abstractmethod
    def disj(self, a, b):
        """Disjunction (or)"""
        pass
    
    def implies(self, a, b):
        """Implication - default implementation"""
        return self.disj(self.neg(a), b)
    
    def aggr_e(self, lst):
        """Existential aggregation - default implementation"""
        if not lst:
            return self.bot()
        return reduce(self.disj, lst)
    
    def aggr_a(self, lst):
        """Universal aggregation - default implementation"""
        if not lst:
            return self.top()
        return reduce(self.conj, lst)

class NeSyFramework(Aggr2SGrpBLat):
    """Base class for NeSy frameworks"""
    pass

# Classical Boolean Logic (Identity monad)
class BooleanLogic(NeSyFramework):
    """Classical boolean logic framework using identity monad"""
    
    def top(self):
        return Just(True)
    
    def bot(self):
        return Just(False)
    
    def neg(self, a):
        return a.map(lambda x: not x)
    
    def conj(self, a, b):
        return a.bind(lambda x: b.map(lambda y: x and y))
    
    def disj(self, a, b):
        return a.bind(lambda x: b.map(lambda y: x or y))

# Probability Distribution using custom Probability monad
class ProbabilisticLogic(NeSyFramework):
    """Probabilistic logic framework using custom Probability monad"""
    
    def top(self):
        return Probability({True: 1.0})
    
    def bot(self):
        return Probability({False: 1.0})
    
    def neg(self, a):
        return a.map(lambda x: not x)
    
    def conj(self, a, b):
        return a.bind(lambda x: b.map(lambda y: x and y))
    
    def disj(self, a, b):
        return a.bind(lambda x: b.map(lambda y: x or y))

# Non-deterministic Set using ListMonad with custom bind
class NonDetSet:
    """Non-deterministic set wrapper around ListMonad for proper monadic behavior"""
    
    def __init__(self, values):
        if isinstance(values, list):
            # Remove duplicates while preserving order
            seen = set()
            unique_values = []
            for v in values:
                if v not in seen:
                    seen.add(v)
                    unique_values.append(v)
            self.items = unique_values
        else:
            self.items = [values]
    
    def map(self, func):
        """Apply function to all values"""
        return NonDetSet([func(v) for v in self.items])
    
    def bind(self, func):
        """Monadic bind operation"""
        result = []
        for v in self.items:
            bound_result = func(v)
            if isinstance(bound_result, NonDetSet):
                result.extend(bound_result.items)
            elif hasattr(bound_result, 'items'):
                result.extend(bound_result.items)
            else:
                result.append(bound_result)
        return NonDetSet(result)
    
    def __repr__(self):
        return f"NonDetSet({self.items})"

class NonDeterministicLogic(NeSyFramework):
    """Non-deterministic logic framework using custom NonDetSet"""
    
    def top(self):
        return NonDetSet([True])
    
    def bot(self):
        return NonDetSet([False])
    
    def neg(self, a):
        return a.map(lambda x: not x)
    
    def conj(self, a, b):
        return a.bind(lambda x: b.map(lambda y: x and y))
    
    def disj(self, a, b):
        return a.bind(lambda x: b.map(lambda y: x or y))

# ---------------------- Syntax ------------------------------

class Term:
    """Base class for terms"""
    pass

class Var(Term):
    """Variable term"""
    def __init__(self, name: str):
        self.name = name
    
    def __repr__(self):
        return f"Var({self.name})"

class Appl(Term):
    """Function application term"""
    def __init__(self, func_name: str, args: List[Term]):
        self.func_name = func_name
        self.args = args
    
    def __repr__(self):
        return f"Appl({self.func_name}, {self.args})"

class Formula:
    """Base class for formulas"""
    pass

class T(Formula):
    """Truth constant"""
    def __repr__(self):
        return "T"

class F(Formula):
    """Falsehood constant"""
    def __repr__(self):
        return "F"

class Pred(Formula):
    """Predicate"""
    def __init__(self, pred_name: str, args: List[Term]):
        self.pred_name = pred_name
        self.args = args
    
    def __repr__(self):
        return f"Pred({self.pred_name}, {self.args})"

class MPred(Formula):
    """Monadic predicate"""
    def __init__(self, pred_name: str, args: List[Term]):
        self.pred_name = pred_name
        self.args = args
    
    def __repr__(self):
        return f"MPred({self.pred_name}, {self.args})"

class Not(Formula):
    """Negation"""
    def __init__(self, formula: Formula):
        self.formula = formula
    
    def __repr__(self):
        return f"Not({self.formula})"

class And(Formula):
    """Conjunction"""
    def __init__(self, left: Formula, right: Formula):
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"And({self.left}, {self.right})"

class Or(Formula):
    """Disjunction"""
    def __init__(self, left: Formula, right: Formula):
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"Or({self.left}, {self.right})"

class Implies(Formula):
    """Implication"""
    def __init__(self, left: Formula, right: Formula):
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"Implies({self.left}, {self.right})"

class Forall(Formula):
    """Universal quantification"""
    def __init__(self, var: str, formula: Formula):
        self.var = var
        self.formula = formula
    
    def __repr__(self):
        return f"Forall({self.var}, {self.formula})"

class Exists(Formula):
    """Existential quantification"""
    def __init__(self, var: str, formula: Formula):
        self.var = var
        self.formula = formula
    
    def __repr__(self):
        return f"Exists({self.var}, {self.formula})"

class Comp(Formula):
    """Computation formula: var := func(args)(formula)"""
    def __init__(self, var: str, func_name: str, args: List[Term], formula: Formula):
        self.var = var
        self.func_name = func_name
        self.args = args
        self.formula = formula
    
    def __repr__(self):
        return f"Comp({self.var}, {self.func_name}, {self.args}, {self.formula})"

# ---------------------- Semantics ------------------------------

class Interpretation:
    """Interpretation structure"""
    
    def __init__(self, universe: List[Any], 
                 funcs: Dict[str, Callable] = None,
                 mfuncs: Dict[str, Callable] = None,
                 preds: Dict[str, Callable] = None,
                 mpreds: Dict[str, Callable] = None):
        self.universe = universe
        self.funcs = funcs or {}
        self.mfuncs = mfuncs or {}
        self.preds = preds or {}
        self.mpreds = mpreds or {}

def lookup_id(identifier: str, mapping: Dict[str, Any]) -> Any:
    """Look up identifier in mapping, raise error if not found"""
    if identifier not in mapping:
        raise KeyError(f"{identifier} has not been declared")
    return mapping[identifier]

def eval_term(interp: Interpretation, valuation: Dict[str, Any], term: Term) -> Any:
    """Evaluate a term"""
    if isinstance(term, Var):
        val = lookup_id(term.name, valuation)
        # If we get a ListMonad from valuation, we need to extract the value
        # This shouldn't happen in well-formed evaluations, but let's be safe
        if hasattr(val, 'items') and isinstance(val.items, list):
            return val.items[0] if val.items else None
        return val
    elif isinstance(term, Appl):
        func = lookup_id(term.func_name, interp.funcs)
        args = [eval_term(interp, valuation, arg) for arg in term.args]
        return func(args)
    else:
        raise ValueError(f"Unknown term type: {type(term)}")

def eval_formula(interp: Interpretation, valuation: Dict[str, Any], 
                formula: Formula, framework: NeSyFramework) -> Any:
    """Evaluate a formula in the given framework"""
    
    if isinstance(formula, T):
        return framework.top()
    elif isinstance(formula, F):
        return framework.bot()
    elif isinstance(formula, Pred):
        pred = lookup_id(formula.pred_name, interp.preds)
        args = [eval_term(interp, valuation, arg) for arg in formula.args]
        result = pred(args)
        
        # Wrap in appropriate monad based on framework
        if isinstance(framework, ProbabilisticLogic):
            return Probability({result: 1.0})
        elif isinstance(framework, NonDeterministicLogic):
            return NonDetSet([result])
        else:  # BooleanLogic
            return Just(result)
            
    elif isinstance(formula, MPred):
        pred = lookup_id(formula.pred_name, interp.mpreds)
        args = [eval_term(interp, valuation, arg) for arg in formula.args]
        return pred(args)
    elif isinstance(formula, Not):
        sub_result = eval_formula(interp, valuation, formula.formula, framework)
        return framework.neg(sub_result)
    elif isinstance(formula, And):
        left = eval_formula(interp, valuation, formula.left, framework)
        right = eval_formula(interp, valuation, formula.right, framework)
        return framework.conj(left, right)
    elif isinstance(formula, Or):
        left = eval_formula(interp, valuation, formula.left, framework)
        right = eval_formula(interp, valuation, formula.right, framework)
        return framework.disj(left, right)
    elif isinstance(formula, Implies):
        left = eval_formula(interp, valuation, formula.left, framework)
        right = eval_formula(interp, valuation, formula.right, framework)
        return framework.implies(left, right)
    elif isinstance(formula, Forall):
        results = []
        for elem in interp.universe:
            new_val = valuation.copy()
            new_val[formula.var] = elem
            results.append(eval_formula(interp, new_val, formula.formula, framework))
        return framework.aggr_a(results)
    elif isinstance(formula, Exists):
        results = []
        for elem in interp.universe:
            new_val = valuation.copy()
            new_val[formula.var] = elem
            results.append(eval_formula(interp, new_val, formula.formula, framework))
        return framework.aggr_e(results)
    elif isinstance(formula, Comp):
        mfunc = lookup_id(formula.func_name, interp.mfuncs)
        args = [eval_term(interp, valuation, arg) for arg in formula.args]
        
        # Get the monadic computation
        m_result = mfunc(args)
        
        # Bind with the formula evaluation
        def eval_with_binding(a):
            new_val = valuation.copy()
            new_val[formula.var] = a
            return eval_formula(interp, new_val, formula.formula, framework)
        
        return m_result.bind(eval_with_binding)
    else:
        raise ValueError(f"Unknown formula type: {type(formula)}")

# Argmax transformation
def maximal_values(prob_dist: Probability) -> NonDetSet:
    """Extract maximal probability values from distribution"""
    max_vals = prob_dist.argmax()
    return NonDetSet(max_vals)

def argmax_transform(interp: Interpretation) -> Interpretation:
    """Transform probabilistic interpretation to non-deterministic"""
    new_mfuncs = {}
    for name, func in interp.mfuncs.items():
        def wrapped_func(args, original_func=func):
            prob_result = original_func(args)
            return maximal_values(prob_result)
        new_mfuncs[name] = wrapped_func
    
    new_mpreds = {}
    for name, pred in interp.mpreds.items():
        def wrapped_pred(args, original_pred=pred):
            prob_result = original_pred(args)
            return maximal_values(prob_result)
        new_mpreds[name] = wrapped_pred
    
    # Keep original predicates and functions unchanged
    # The issue was that we need to handle the universe properly
    return Interpretation(
        universe=interp.universe,
        funcs=interp.funcs,
        mfuncs=new_mfuncs,
        preds=interp.preds,  # Keep original predicates
        mpreds=new_mpreds
    )

# ---------------------- Examples ------------------------------

# Dice example
def create_dice_model():
    """Create dice model interpretation"""
    universe = list(range(1, 7))
    
    funcs = {str(i): lambda args, val=i: val for i in range(1, 7)}
    
    def die_func(args):
        # Uniform distribution over dice outcomes
        return Probability.uniform(list(range(1, 7)))
    
    mfuncs = {
        "die": die_func
    }
    
    preds = {
        "==": lambda args: args[0] == args[1],
        "even": lambda args: args[0] % 2 == 0
    }
    
    return Interpretation(universe, funcs, mfuncs, preds)

# Traffic light example
class Universe(Enum):
    RED = "Red"
    YELLOW = "Yellow" 
    GREEN = "Green"
    B_FALSE = "B_False"
    B_TRUE = "B_True"

def create_traffic_model():
    """Create traffic light model interpretation"""
    universe = list(Universe)
    
    funcs = {
        "green": lambda args: Universe.GREEN
    }
    
    def light_func(args):
        return Probability({
            Universe.RED: 0.3,
            Universe.GREEN: 0.6,
            Universe.YELLOW: 0.1
        })
    
    def drive_f_func(args):
        light = args[0]
        if light == Universe.RED:
            return Probability({
                Universe.B_TRUE: 0.1,
                Universe.B_FALSE: 0.9
            })
        elif light == Universe.YELLOW:
            return Probability({
                Universe.B_TRUE: 0.2,
                Universe.B_FALSE: 0.8
            })
        elif light == Universe.GREEN:
            return Probability({
                Universe.B_TRUE: 0.9,
                Universe.B_FALSE: 0.1
            })
        else:
            return Probability({Universe.B_FALSE: 1.0})
    
    mfuncs = {
        "light": light_func,
        "driveF": drive_f_func
    }
    
    def eval_pred(args):
        b = args[0]
        return b == Universe.B_TRUE
    
    preds = {
        "==": lambda args: args[0] == args[1],
        "eval": eval_pred
    }
    
    def drive_p_pred(args):
        light = args[0]
        if light == Universe.RED:
            return Probability({True: 0.1, False: 0.9})
        elif light == Universe.YELLOW:
            return Probability({True: 0.2, False: 0.8})
        elif light == Universe.GREEN:
            return Probability({True: 0.9, False: 0.1})
        else:
            return Probability({False: 1.0})
    
    mpreds = {
        "driveP": drive_p_pred
    }
    
    return Interpretation(universe, funcs, mfuncs, preds, mpreds)

# Example usage and testing
def main():
    """Main function demonstrating the framework"""
    
    # Create frameworks
    bool_framework = BooleanLogic()
    prob_framework = ProbabilisticLogic()
    nondet_framework = NonDeterministicLogic()
    
    # Create dice model
    dice_model = create_dice_model()
    
    # Dice sentence 1: x:=die() (x==6 ∧ even(x))
    die_sen1 = Comp("x", "die", [], 
                    And(Pred("==", [Var("x"), Appl("6", [])]),
                        Pred("even", [Var("x")])))
    
    # Dice sentence 2: (x:=die() (x==6)) ∧ (x:=die() even(x))
    die_sen2 = And(Comp("x", "die", [], Pred("==", [Var("x"), Appl("6", [])])),
                   Comp("x", "die", [], Pred("even", [Var("x")])))
    
    # Evaluate with probabilistic framework
    print("=== Dice Examples (Probabilistic) ===")
    d1 = eval_formula(dice_model, {}, die_sen1, prob_framework)
    print(f"d1 (x:=die() (x==6 ∧ even(x))): {d1.probability}")
    
    d2 = eval_formula(dice_model, {}, die_sen2, prob_framework)
    print(f"d2 ((x:=die() (x==6)) ∧ (x:=die() even(x))): {d2.probability}")
    
    # Transform and evaluate with non-deterministic framework
    print("\n=== Dice Examples (Non-deterministic after argmax) ===")
    dice_model_c = argmax_transform(dice_model)
    d1c = eval_formula(dice_model_c, {}, die_sen1, nondet_framework)
    print(f"d1C: {d1c.items}")
    
    d2c = eval_formula(dice_model_c, {}, die_sen2, nondet_framework)
    print(f"d2C: {d2c.items}")
    
    # Create traffic model
    traffic_model = create_traffic_model()
    
    # Traffic sentence 1: l:=light(), d:=driveF(l) (eval(d) -> l==green)
    traffic_sen1 = Comp("l", "light", [],
                       Comp("d", "driveF", [Var("l")],
                           Implies(Pred("eval", [Var("d")]),
                                  Pred("==", [Var("l"), Appl("green", [])]))))
    
    # Traffic sentence 2: l:=light() (driveP(l) -> l==green)
    traffic_sen2 = Comp("l", "light", [],
                       Implies(MPred("driveP", [Var("l")]),
                              Pred("==", [Var("l"), Appl("green", [])])))
    
    # Evaluate with probabilistic framework
    print("\n=== Traffic Examples (Probabilistic) ===")
    t1 = eval_formula(traffic_model, {}, traffic_sen1, prob_framework)
    print(f"t1 (l:=light(), d:=driveF(l) (eval(d) -> l==green)): {t1.probability}")
    
    t2 = eval_formula(traffic_model, {}, traffic_sen2, prob_framework)
    print(f"t2 (l:=light() (driveP(l) -> l==green)): {t2.probability}")
    
    # Transform and evaluate with non-deterministic framework
    print("\n=== Traffic Examples (Non-deterministic after argmax) ===")
    traffic_model_c = argmax_transform(traffic_model)
    t1c = eval_formula(traffic_model_c, {}, traffic_sen1, nondet_framework)
    print(f"t1C: {t1c.items}")
    
    t2c = eval_formula(traffic_model_c, {}, traffic_sen2, nondet_framework)
    print(f"t2C: {t2c.items}")
    
    # Sample from results
    print("\n=== Sampling Results ===")
    sample_sets = [
        ("d1C", d1c),
        ("d2C", d2c), 
        ("t1C", t1c),
        ("t2C", t2c)
    ]
    
    for name, result in sample_sets:
        if result.items:
            sample = random.choice(result.items)
            print(f"{name} sample: {sample}")
        else:
            print(f"{name} sample: No values to sample")

if __name__ == "__main__":
    # Note: You'll need to install pymonad first:
    # pip install PyMonad
    main()

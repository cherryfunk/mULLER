# mULLER: A Modular Categorical Semantics of the Neurosymbolic ULLER Framework via Monads

mULLER is a Python implementation of the neurosymbolic framework described in "mULLER: A Modular Categorical Semantics of the Neurosymbolic ULLER Framework via Monads" by Daniel Romero Schellhorn and Till Mossakowski. This package provides a unified approach to neurosymbolic AI by using monads to model computational effects in first-order logic formulas, enabling modular integration of neural components with symbolic reasoning.

## Overview

mULLER extends the ULLER (Unified Language for LEarning and Reasoning) framework with a categorical semantics based on monads. This allows for:

- **Modular semantics**: Different logics (classical, probabilistic, fuzzy, non-deterministic) as instances of one framework
- **Neural integration**: Support for computational function and predicate symbols that can be realized by neural networks
- **Flexible truth spaces**: Work with different monadic truth value spaces (boolean, probabilistic, fuzzy)
- **Systematic transformations**: Convert between different semantic frameworks (e.g., probabilistic to classical via argmax)

The framework supports formulas of the form `x := m(T₁, ..., Tₙ)(F)` where `m` is a computational (neural) model, enabling seamless integration of machine learning components into logical reasoning.

## Installation

### Prerequisites

- Python 3.13 or higher
- uv package manager

### Install from source

```bash
git clone https://github.com/your-username/mULLER.git
cd mULLER
uv pip install -e .
```

### Install from GitHub
```bash
uv pip install git+https://github.com/your-username/mULLER.git
```

### Dependencies

The package automatically installs the following dependencies:
- `lark>=1.2.2` - for parsing first-order logic formulas
- `numpy>=2.3.2` - for numerical computations
- `pymonad>=2.4.0` - for monadic operations
- `scipy>=1.16.0` - for statistical distributions

## Quick Start

### Basic Probabilistic Example

```python
from muller import nesy, parse, Prob, uniform, weighted
from muller import NeSyFramework, Interpretation

# Create a probabilistic NeSy framework
prob_framework = nesy(Prob, bool)

# Define an interpretation with a universe and functions
interpretation = Interpretation(
    universe=[1, 2, 3, 4, 5, 6],
    functions={"1": lambda: 1, "6": lambda: 6},
    mfunctions={"die": lambda: uniform([1, 2, 3, 4, 5, 6])},
    preds={"equals": lambda x, y: x == y, "even": lambda x: x % 2 == 0},
    mpreds={}
)

# Parse and evaluate a dice formula: "roll a die and check if it's 6 and even"
formula = parse("x := $die() (equals(x, 6) and even(x))")
result = formula.eval(prob_framework, interpretation, {})

print(f"Probability of rolling an even 6: {result.value[True]}")  # 1/6 ≈ 0.167
```

### Non-Deterministic Example

```python
from muller import NonEmptyPowerset, from_list, singleton

# Create a non-deterministic NeSy framework
nondet_framework = nesy(NonEmptyPowerset, bool)

interpretation = Interpretation(
    universe=["alice", "bob"],
    functions={},
    mfunctions={"choose_person": lambda: from_list(["alice", "bob"])},
    preds={},
    mpreds={
        "might_be_tall": lambda x: (
            from_list([True, False]) if x == "alice" else singleton(False)
        )
    }
)

# Parse a non-deterministic formula
formula = parse("x := $choose_person() ($might_be_tall(x))")
result = formula.eval(nondet_framework, interpretation, {})

print(f"Possible truth values: {result.value}")  # frozenset({True, False})
```

### Traffic Light Example

```python
# Probabilistic traffic light decision making
interpretation = Interpretation(
    universe=["red", "green", "yellow", True, False],
    functions={"green": lambda: "green", "red": lambda: "red"},
    mfunctions={
        "light": lambda: weighted([("red", 0.3), ("green", 0.6), ("yellow", 0.1)]),
        "driveF": lambda l: (
            weighted([(True, 0.1), (False, 0.9)]) if l == "red" else
            weighted([(True, 0.2), (False, 0.8)]) if l == "yellow" else
            weighted([(True, 0.9), (False, 0.1)])  # green
        )
    },
    preds={"equals": lambda a, b: a == b, "eval": lambda x: x == True},
    mpreds={}
)

# "If we drive, the light should be green"
formula = parse("L := $light()(D := $driveF(L) (eval(D) -> equals(L, green)))")
result = formula.eval(prob_framework, interpretation, {})

print(f"Safety probability: {result.value[True]}")  # ~0.95
```

## Formula Syntax

The parser supports standard first-order logic with computational extensions:

**Logical connectives:**
- `and`, `or`, `not`
- `->` (implication)
- `forall X F`, `exists X F` (quantifiers)

**Computational formulas:**
- `x := $m(T1, ..., Tn) (F)` - computational function application
- `$p(T1, ..., Tn)` - computational predicate application

**Terms:**
- Variables: `X`, `Y`, `Z` (uppercase)
- Functions: `f(X)`, `father(alice)` (lowercase)
- Constants: `alice`, `bob`, `1`, `2`

## API Reference

### Core Functions

#### `nesy(logic)`
Creates a NeSy framework instance from a logic.

**Parameters:**
- `logic`: An instance of `Aggr2SGrpBLat[Monad]` that defines the logical operations

**Returns:** `NeSyFramework` instance

#### `nesy(monad_type, omega=bool)`
Creates a NeSy framework instance from a monad type.

**Parameters:**
- `monad_type`: A monad type (`Prob`, `NonEmptyPowerset`, `Identity`)
- `omega`: Type for truth values (default: `bool`)

**Returns:** `NeSyFramework` instance

#### `parse(formula_string)`
Parses a first-order logic formula string into an AST.

**Parameters:**
- `formula_string`: String representation of the formula

**Returns:** Formula AST that can be evaluated

### Monads

#### `Prob`
Probability distribution monad for probabilistic reasoning.

**Helper functions:**
- `uniform(values)`: Create uniform distribution over values
- `weighted(pairs)`: Create weighted distribution from (value, weight) pairs  
- `bernoulli(p)`: Create Bernoulli distribution with probability p

#### `NonEmptyPowerset`
Non-empty powerset monad for non-deterministic reasoning.

**Helper functions:**
- `from_list(values)`: Create non-deterministic choice from list
- `singleton(value)`: Create singleton set containing one value

#### `Identity`
Identity monad for classical deterministic reasoning.

## Advanced Usage

### Custom Monads

You can define custom monads by inheriting from the base `Monad` class and creating a corresponding logic:

```python
from pymonad.monad import Monad
from muller.logics import Aggr2SGrpBLat

class MyCustomMonad[T](Monad[T]):
    def __init__(self, value: T):
        self.value = value
    
    @classmethod
    def insert(cls, value: T) -> 'MyCustomMonad[T]':
        return cls(value)
    
    def bind(self, function) -> 'MyCustomMonad':
        # Define your monadic bind operation
        return function(self.value)

class MyCustomLogic(Aggr2SGrpBLat[MyCustomMonad[bool]]):
    """Custom logic for your monad"""
    
    def top(self) -> MyCustomMonad[bool]:
        return MyCustomMonad.insert(True)
    
    def bottom(self) -> MyCustomMonad[bool]:
        return MyCustomMonad.insert(False)
    
    # Implement other required methods...

# Use with mULLER
logic = MyCustomLogic()
custom_framework = nesy(logic)
```

## Examples and Tests

See the `test/` directory for comprehensive examples including:
- `test_nesy_framework.py`: Core framework functionality
- `test_parser.py`: Formula parsing and evaluation

Run tests with:
```bash
python -m pytest test/
```

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use mULLER in your research, please cite:

```bibtex
@inproceedings{schellhorn2025muller,
  title={mULLER: A Modular Categorical Semantics of the Neurosymbolic ULLER Framework via Monads},
  author={Schellhorn, Daniel Romero and Mossakowski, Till},
  booktitle={Proceedings of Machine Learning Research},
  volume={284},
  pages={1--28},
  year={2025},
  organization={19th Conference on Neurosymbolic Learning and Reasoning}
}
```

## References

- Original ULLER framework: [Van Krieken et al., 2024](https://doi.org/10.1007/978-3-031-71167-1_12)
- Computational monads: [Moggi, 1991](https://doi.org/10.1016/0890-5401(91)90052-4)
- Logic Tensor Networks: [Badreddine et al., 2022](https://doi.org/10.1016/j.artint.2021.103649)
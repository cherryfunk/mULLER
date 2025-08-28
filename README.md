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

- Python 3.12 or higher
- uv package manager

### Install from source

```bash
git clone https://github.com/cherryfunk/mULLER.git
cd mULLER
uv pip install -e .
```

### Install from GitHub
```bash
uv pip install git+https://github.com/cherryfunk/mULLER.git
```

### Dependencies

The package automatically installs the following dependencies:
- `lark>=1.2.2` - for parsing ULLER formulas
- `numpy>=2.3.2` - for numerical computations
- `pymonad>=2.4.0` - for monadic operations
- `scipy>=1.16.0` - for statistical distributions

## Usage

1. Choose a nesy framework based on your logic. The `nesy` function can create frameworks for different monads like `Prob`, `NonEmptyPowerset`, or `Identity`. Either provide it with a Monad type (T) and a truth value type (Ω, defaulting to `bool`), or instantiate a logic (an implementation of `Aggr2GrpBLat[Monad[O]]`) yourself and pass it to `nesy`. 
2. Define an interpretation with a universe, functions, and predicates. Type checkers should infer the universe types, when you start with the `universe` parameter. Alternatively, specify the type of the universe explicitly e.g.:
    ```python
   Universe = Literal["alice", "bob"]
   O = bool
   interpretation = Interpretation[Universe, O](
       universe=["alice", "bob"],
       ...
   )
    ```
3. Parse the ULLER formula using the `parse` function, which returns an abstract syntax tree (AST) representation of the formula.
4. Evaluate the formula against the interpretation using the `eval` method, which returns a result in the specified Monad.

### Examples

More examples can be found in the `examples/` directory.

#### Basic Probabilistic Example

```python
from muller import Prob, uniform, weighted
from muller import nesy, parse, NeSyFramework, Interpretation
from muller.transformation import argmax  # For transformations between frameworks

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
formula = parse("X := $die() (equals(X, 6) and even(X))")
result = formula.eval(prob_framework, interpretation, {})

print(f"Probability of rolling an even 6: {result.value[True]}")  # 1/6 ≈ 0.167
```

#### Non-Deterministic Example

```python
from muller import NonEmptyPowerset, from_list, singleton
from muller import nesy, parse, NeSyFramework, Interpretation
from muller.logics import Priest

# Create a non-deterministic NeSy framework
nondet_framework = nesy(NonEmptyPowerset, Priest)

interpretation = Interpretation(
    universe=["alice", "bob"],
    functions={},
    mfunctions={"choose_person": lambda: from_list(["alice", "bob"])},
    preds={},
    mpreds={
        "might_be_tall": lambda x: {
            "alice": singleton("Both"),
            "bob": singleton(False) 
        }.get(x, singleton(False))
    }
)

# Parse a non-deterministic formula
formula = parse("X := $choose_person() ($might_be_tall(X))")
result = formula.eval(nondet_framework, interpretation, {})

print(f"Possible truth values: {result.value}")  # frozenset({"Both", False})
```

#### Traffic Light Example

```python
from muller import Prob, weighted
from muller import nesy, parse, Interpretation

prob_framework = nesy(Prob, bool)

interpretation = Interpretation(
    universe=["red", "green", "yellow", True, False],
    functions={"green": lambda: "green", "red": lambda: "red"},
    mfunctions={
        "light": lambda: weighted([("red", 0.6), ("green", 0.3), ("yellow", 0.1)]),
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

print(f"Safety probability: {result.value[True]}")  # ~0.92
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
- `logic`: An instance of `Aggr2SGrpBLat[Monad[O]]` that defines the logical operations

**Returns:** `NeSyFramework` instance

#### `nesy(monad_type, omega=bool)`
Creates a NeSy framework instance from a monad type.

The function will search all loaded modules for a subclass of `Aggr2SGrpBLat` that matches the provided monad and truth value type and returns a corresponding `NeSyFramework`. To extend the built-in logics, you can create a new logic class that inherits from `Aggr2SGrpBLat` and implements the required methods. The search stops at the first matching logic class found and starts with the built-in logics. To overwrite a builtin implementation with a custom implementation, it has to be instantiated (second overload of `nesy` function).

If no matching logic is found, it raises a `ValueError`.

Example:
```python
from muller import nesy, Prob
from pymonad import Prob
from muller.logics import Aggr2SGrpBLat

class MyLogicOverwrite(Aggr2SGrpBLat[Prob[bool]]):
    ...
    
class MyCustomLogic(Aggr2SGrpBLat[Prob[str]]):
    ...
        
nesy_framework = nesy(Prob, bool) # Uses `muller.logics.ProbabilisticBooleanLogic`
nesy_framework = nesy(MyLogicOverwrite()) # Uses `MyLogicOverwrite`
nesy_framework = nesy(Prob, str) # Uses `MyCustomLogic`
```

**Parameters:**
- `monad_type`: A monad type (`Prob`, `NonEmptyPowerset`, `Identity`)
- `omega`: Type for truth values (default: `bool`)

**Returns:** `NeSyFramework` instance

#### `parse(formula_string)`
Parses a ULLER formula string into an AST.

**Parameters:**
- `formula_string`: String representation of the formula

**Returns:** Formula AST that can be evaluated

### Transformations

#### `argmax()`
Transformation that converts probabilistic interpretations to non-deterministic ones by selecting values with maximum probability.

**Usage:**
```python
from muller.transformation import argmax
nondet_interpretation = prob_interpretation.transform(argmax())
```

**Features:**
- Converts `Prob[T]` to `NonEmptyPowerset[T]`
- Handles probability ties by including all maximal values
- Preserves regular functions and predicates

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

### Transformations

mULLER supports systematic transformations between semantic frameworks, converting interpretations from one monad to another.

#### Using Built-in Transformations

```python
from muller.transformation import argmax

# Apply argmax to convert probabilistic to non-deterministic
prob_interpretation = Interpretation(...)  # Your probabilistic interpretation
nondet_interpretation = prob_interpretation.transform(argmax())
```

The `argmax` transformation selects values with maximum probability, handling ties by including all maximal values.

#### Custom Transformations

Create custom transformations by inheriting from `NeSyTransformer`:

```python
from muller.nesy_framework import NeSyTransformer, Interpretation

class MyTransformer[A, B, C](NeSyTransformer[A, B, C]):
    def __call__(self, interpretation: Interpretation[A, B]) -> Interpretation[A, C]:
        # Transform monadic functions/predicates as needed
        transformed_mfunctions = {}
        for name, func in interpretation.mfunctions.items():
            def transform_function(original_func):
                def wrapper(*args):
                    result = original_func(*args)
                    # Apply your transformation logic here
                    return your_transformation(result)
                return wrapper
            transformed_mfunctions[name] = transform_function(func)
        
        # Similar for mpreds...
        return Interpretation(
            universe=interpretation.universe,
            functions=interpretation.functions,  # Usually preserved
            mfunctions=transformed_mfunctions,
            preds=interpretation.preds,  # Usually preserved
            mpreds=transformed_mpreds,
        )

# Use your custom transformer
my_transformer = MyTransformer()
transformed_interpretation = interpretation.transform(my_transformer)
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
"""
GiryPyMC Monad Implementation for Probabilistic Programming

This module implements a simplified version of the Giry monad using basic sampling
instead of PyMC's full probabilistic programming capabilities. The GiryPyMC monad
provides a clean monadic interface for composing probabilistic computations.

Key Features:
- Monadic interface with `insert` (pure/return) and `bind` operations
- Support for basic distributions: uniform, categorical
- Composable probabilistic computations through bind
- Heterogeneous sample types (integers, floats, tuples, dictionaries)
- Practical examples: weather prediction, mixed universe sampling
- Comprehensive type annotations using modern Python 3.12+ syntax

The implementation demonstrates how monadic composition can be used to build
complex probabilistic models from simple building blocks.

Type System:
- Generic monad: GiryPyMC[T] using modern class syntax: `class GiryPyMC[T]:`
- Generic methods: `def bind[S](self, ...) -> GiryPyMC[S]:`
- Type-safe distribution functions with proper return types
- Support for Union types and custom type aliases
"""

from typing import Union, List, Dict
import numpy as np
import pytensor.tensor as pt


def humid_detector(d: int) -> float:
    """Humidity detection function for day d."""
    return 0.7 if d % 2 == 0 else 0.2


def temperature_predictor(d: int) -> tuple[float, float]:
    """Temperature prediction function returning (mean, std) for day d."""
    return (10 + d, (11 + d) / 5.0)

Universe = Union[int, float, tuple[float, float]]

WeatherPrediction = Dict[str, Union[float, int]]


def uniformUniverse() -> GiryPyMC[Universe]:
    """
    Create a mixed universe distribution over integers, floats, and tuples.
    This is a direct function that returns a GiryPyMC monad.
    """
    return categorical([1,1,1]).bind(lambda choice: {
        0: categorical([1] * 21).bind(lambda i: GiryPyMC.insert(i - 10)),
        1: uniform(0, 1),
        2: uniform(-5, 20).bind(lambda x: uniform(0, 5).bind(lambda y: GiryPyMC.insert((x, y))))
    }.get(choice, uniform(0, 1)))  # Default case to avoid KeyError


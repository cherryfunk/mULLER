import sys
from typing import Union, cast

from muller import nesy
from muller.monad import (
    GirySampling,
    giry_bernoulli,
    giry_normal,
)
from muller.monad import (
    giry_categorical as categorical,
)
from muller.monad import (
    giry_uniform as uniform,
)
from muller.nesy_framework import NeSyFramework
from muller.parser import parse

GIRY = GirySampling

# No multi sort implementation yet. Thus, union of all sorts
Universe = Union[int, float, tuple[float, float]]

# Get a framework for bools over the Giry monad and an infinite universe over the Giry monad
nf: NeSyFramework[GIRY[bool], bool, GIRY[Universe], Universe] = nesy(GIRY, bool, GIRY)


def uniformUniverse() -> GIRY[Universe]:
    """
    Create a mixed universe distribution over integers, floats, and tuples.
    """
    return categorical([1, 1, 1]).bind(
        lambda choice: categorical([1] * 21).bind(lambda i: GIRY.from_value(i - 10))
        if choice == 0
        else uniform(0, 1)
        if choice == 1
        else uniform(-5, 20).bind(
            lambda x: uniform(0, 5).bind(lambda y: GIRY.from_value((x, y)))
        )
    )


model = nf.create_interpretation(
    sort=uniformUniverse(),
    functions={
        "0": lambda _: 0.0,
        "1": lambda _: 1.0,
        "15": lambda _: 15.0,
    },
    predicates={
        "==": lambda args: args[0] == args[1],
        "<": lambda args: args[0] < args[1],
        ">": lambda args: args[0] > args[1],
    },
)


@model.fn()
def humid_detector(d: Universe) -> float:
    """Humidity detection function for day d."""
    if isinstance(d, int):
        return 0.7 if d % 2 == 0 else 0.2

    return 0.0


@model.fn()
def temperature_predictor(d: Universe) -> tuple[float, float]:
    """Temperature prediction function returning (mean, std) for day d."""
    if isinstance(d, int):
        return (10 + d, (11 + d) / 5.0)

    return (0.0, 0.0)


@model.fn()
def data1() -> int:
    return 1


@model.comp_fn()
def bernoulli(p: Universe) -> GIRY[Universe]:
    if isinstance(p, (int, float)):
        return cast(GIRY[Universe], giry_bernoulli(p))

    return GIRY.from_value(0)


@model.comp_fn()
def normal(d: Universe) -> GIRY[Universe]:
    match d:
        case (mean, std) if isinstance(mean, (int, float)) and isinstance(
            std, (int, float)
        ):
            return cast(GIRY[Universe], giry_normal(mean, std))
        case _:
            return GIRY.from_value(0)


# weather_model = Interpretation[Universe, bool, GIRY[Universe]](
#     universe=uniformUniverse(),
#     functions={
#         "humid_detector": lambda d: humid_detector(d),
#         "temperature_predictor": lambda d: temperature_predictor(d),
#         "data1": lambda: 1,
#         "0": lambda: 0,
#         "1": lambda: 1,
#         "0": lambda: 0.0,
#         "15": lambda: 15.0,
#     },
#     mfunctions={
#         "bernoulli": lambda d: universeBernoulli(d),
#         "normal": lambda d: universeNormal(d),
#     },
#     preds={
#         "==": lambda x, y: x == y,
#         "<": lambda x, y: x < y,
#         ">": lambda x, y: x > y,
#     },
#     mpreds={},
# )

sys.setrecursionlimit(1500)


formula = parse(
    """
H := $bernoulli(humid_detector(data1)),
T := $normal(temperature_predictor(data1))
((H == 1 ∧ T < 0) ∨ (H == 0 ∧ T > 15))
"""
)
result = nf.eval_formula(formula, model, {})
samples = result.sample(1000)
print(
    f"Probability of a humid and cold or non humid and warm day: {sum(1 for s in samples if s) / len(samples)}"
)


formula = parse(
    """
![D]: (
    H := $bernoulli(humid_detector(D)),
    T := $normal(temperature_predictor(D))
    ((H == 1 ∧ T < 0) ∨ (H == 0 ∧ T > 15))
)
"""
)
result = nf.eval_formula(formula, model, {})
samples = result.sample(100)
print(
    f"Probability that all days are humid and cold or non humid and warm: {sum(1 for s in samples if s) / len(samples)}"
)


formula = parse(
    """
?[D]: (
    H := $bernoulli(humid_detector(D)),
    T := $normal(temperature_predictor(D))
    ((H == 1 ∧ T < 0) ∨ (H == 0 ∧ T > 15))
)
"""
)
result = nf.eval_formula(formula, model, {})
samples = result.sample(1000)
print(
    f"Probability at least one day is humid and cold or non humid and warm: {sum(1 for s in samples if s) / len(samples)}"
)

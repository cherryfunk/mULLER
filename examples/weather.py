from typing import Union, Dict, cast

from muller import Interpretation, nesy
from muller.parser import parse

from muller.monad import (
    GirySampling,
    giry_bernoulli as bernoulli,
    giry_categorical as categorical,
    giry_normal as normal,
    giry_uniform as uniform
)

GIRY = GirySampling

# Get a framework for bools over the Giry monad and an infinite universe over the Giry monad
framework = nesy(GIRY, bool, GIRY)

# No multi sort implementation yet. Thus, union of all sorts
Universe = Union[int, float, tuple[float, float]]

def humid_detector(d: Universe) -> float:
    """Humidity detection function for day d."""
    if isinstance(d, int):
        return 0.7 if d % 2 == 0 else 0.2

    return 0.0


def temperature_predictor(d: Universe) -> tuple[float, float]:
    """Temperature prediction function returning (mean, std) for day d."""
    if isinstance(d, int):
        return (10 + d, (11 + d) / 5.0)

    return (0.0, 0.0)


def uniformUniverse() -> GIRY[Universe]:
    """
    Create a mixed universe distribution over integers, floats, and tuples.
    """
    return categorical([1, 1, 1]).bind(
        lambda choice: {
            0: categorical([1] * 21).bind(lambda i: GIRY.insert(i - 10)),
            1: uniform(0, 1),
            2: uniform(-5, 20).bind(
                lambda x: uniform(0, 5).bind(lambda y: GIRY.insert((x, y)))
            ),
        }[choice]
    )


def universeBernoulli(p: Universe) -> GIRY[Universe]:
    if isinstance(p, (int, float)):
        return cast(GIRY[Universe], bernoulli(p))

    return GIRY.insert(0)


def universeNormal(d: Universe) -> GIRY[Universe]:
    match d:
        case (mean, std) if isinstance(mean, (int, float)) and isinstance(
            std, (int, float)
        ):
            return cast(GIRY[Universe], normal(mean, std))
        case _:
            return GIRY.insert(0)


weather_model = Interpretation[Universe, bool, GIRY[Universe]](
    universe=uniformUniverse(),
    functions={
        "humid_detector": lambda d: humid_detector(d),
        "temperature_predictor": lambda d: temperature_predictor(d),
        "data1": lambda: 1,
        "0": lambda: 0,
        "1": lambda: 1,
        "0": lambda: 0.0,
        "15": lambda: 15.0,
    },
    mfunctions={
        "bernoulli": lambda d: universeBernoulli(d),
        "normal": lambda d: universeNormal(d),
    },
    preds={
        "==": lambda x, y: x == y,
        "<": lambda x, y: x < y,
        ">": lambda x, y: x > y,
    },
    mpreds={},
)

formula = parse(
    """
H := $bernoulli(humid_detector(data1)),
T := $normal(temperature_predictor(data1))
((H == 1 ∧ T < 0) ∨ (H == 0 ∧ T > 15))
"""
)
result = formula.eval(framework, weather_model, {})
samples = result.sample(1000)
print(f"Probability of a humid and cold or non humid and warm day: {sum(1 for s in samples if s) / len(samples)}")


formula = parse(
    """
![D]: (
    H := $bernoulli(humid_detector(D)),
    T := $normal(temperature_predictor(D))
    ((H == 1 ∧ T < 0) ∨ (H == 0 ∧ T > 15))
)
"""
)
result = formula.eval(framework, weather_model, {})
samples = result.sample(100)
print(f"Probability that all days are humid and cold or non humid and warm: {sum(1 for s in samples if s) / len(samples)}")


formula = parse(
    """
?[D]: (
    H := $bernoulli(humid_detector(D)),
    T := $normal(temperature_predictor(D))
    ((H == 1 ∧ T < 0) ∨ (H == 0 ∧ T > 15))
)
"""
)
result = formula.eval(framework, weather_model, {})
samples = result.sample(1000)
print(f"Probability at least one day is humid and cold or non humid and warm: {sum(1 for s in samples if s) / len(samples)}")

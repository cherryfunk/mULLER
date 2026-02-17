import random
from typing import Any, Literal, get_args

from muller import Prob, nesy, parse, weighted
from muller.monad.giry_sampling import GirySampling
from muller.nesy_framework import NeSyFramework

Universe = Literal["red", "green", "yellow", False, True]
universe: list[Universe] = list(get_args(Universe))

nf: NeSyFramework[Prob[bool], bool, GirySampling[Universe], Universe] = nesy(
    Prob, bool, GirySampling
)

model = nf.create_interpretation(sort=GirySampling(lambda: random.choice(universe)))


def _drive(l: Universe) -> Prob[bool]:
    """Drive function based on traffic light color."""
    match l:
        case "red":
            return weighted([(True, 0.1), (False, 0.9)])
        case "yellow":
            return weighted([(True, 0.2), (False, 0.8)])
        case "green":
            return weighted([(True, 0.9), (False, 0.1)])
        case _:
            return Prob({})


@model.comp_fn()
def driveF(l: Universe) -> Prob[bool]:
    return _drive(l)


@model.comp_fn()
def light() -> Prob[Universe]:
    return weighted([("red", 0.6), ("green", 0.3), ("yellow", 0.1)])


@model.fn()
def green() -> Universe:
    return "green"


@model.fn()
def red() -> Universe:
    return "red"


@model.fn()
def yellow() -> Universe:
    return "yellow"


@model.pred()
def equals(a: Universe, b: Universe) -> bool:
    return a == b


@model.pred()
def eval(x: Universe) -> bool:
    if isinstance(x, bool):
        return x

    raise ValueError(f"Expected a boolean value, got {x}")


@model.comp_pred()
def driveP(l: Universe) -> Prob[bool]:
    return _drive(l)


# model = Interpretation[Universe, bool, list[Universe]](
#     universe=universe,
#     functions={
#         "green": lambda: "green",
#         "red": lambda: "red",
#         "yellow": lambda: "yellow",
#     },
#     mfunctions={
#         "light": lambda: weighted([("red", 0.6), ("green", 0.3), ("yellow", 0.1)]),
#         "driveF": _drive,
#     },
#     preds={
#         "equals": lambda a, b: a == b,
#         "eval": lambda x: x,
#     },
#     mpreds={"driveP": _drive},
# )


light_formula1 = parse("L := $light()(D := $driveF(L) (eval(D) -> equals(L, green)))")
# With syntactic sugar, this is equivalent to:
light_formula2 = parse("L := $light(), D := $driveF(L) (eval(D) -> equals(L, green))")
assert light_formula1 == light_formula2

result = nf.eval(light_formula1, model)
print("Probability of driving when light is green:", result)

# Alternative way using mpred `driveP`
light_formula = parse("L := $light() ($driveP(L) -> equals(L, green))")

result = nf.eval(light_formula, model)
print("Probability of driving when light is green (using mpred):", result)

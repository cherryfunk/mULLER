from typing import Any, Literal

from muller import Dist, nesy, parse, weighted
from muller.hkt import List
from muller.framework.nesy import BaseNeSyFramework


Signature = Literal["red", "green", "yellow"] | bool
sort: List[Signature] = List(["red", "green", "yellow", False, True])

nf = nesy(Dist, bool, List)

model = nf.create_interpretation(sort=sort)


def _drive(l: Signature) -> Dist[Signature]:
    """Drive function based on traffic light color."""
    match l:
        case "red":
            return weighted([(True, 0.1), (False, 0.9)])
        case "yellow":
            return weighted([(True, 0.2), (False, 0.8)])
        case "green":
            return weighted([(True, 0.9), (False, 0.1)])
        case _:
            return Dist({})


@model.comp_fn()
def driveF(l: Signature) -> Dist[Signature]:
    return _drive(l)


@model.comp_fn()
def light() -> Dist[Signature]:
    return weighted([("red", 0.6), ("green", 0.3), ("yellow", 0.1)])


@model.fn()
def green() -> Signature:
    return "green"


@model.fn()
def red() -> Signature:
    return "red"


@model.fn()
def yellow() -> Signature:
    return "yellow"


@model.pred()
def equals(a: Signature, b: Signature) -> bool:
    return a == b


@model.pred()
def eval(x: Signature) -> bool:
    if isinstance(x, bool):
        return x

    raise ValueError(f"Expected a boolean value, got {x}")


@model.comp_pred()
def driveP(l: Signature) -> Dist[bool]:
    return _drive(l).bind(
        lambda x: Dist.from_value(x) if isinstance(x, bool) else Dist({})
    )


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

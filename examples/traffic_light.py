from typing import Literal, get_args

from muller import nesy, Prob, parse, Interpretation, weighted

prob_framework = nesy(Prob, bool)

Universe = Literal["red", "green", "yellow", False, True]
universe = list(get_args(Universe))


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


traffic_light_model = Interpretation[Universe, bool, list[Universe]](
    universe=universe,
    functions={
        "green": lambda: "green",
        "red": lambda: "red",
        "yellow": lambda: "yellow",
    },
    mfunctions={
        "light": lambda: weighted([("red", 0.6), ("green", 0.3), ("yellow", 0.1)]),
        "driveF": _drive,
    },
    preds={
        "equals": lambda a, b: a == b,
        "eval": lambda x: x,
    },
    mpreds={
        "driveP": _drive
    },
)

valuation = {}

light_formula1 = parse(
    "L := $light()(D := $driveF(L) (eval(D) -> equals(L, green)))"
)
# With syntactic sugar, this is equivalent to:
light_formula2 = parse(
    "L := $light(), D := $driveF(L) (eval(D) -> equals(L, green))"
)
assert (light_formula1 == light_formula2)

result = light_formula1.eval(prob_framework, traffic_light_model, valuation)
print("Probability of driving when light is green:", result)

# Alternative way using mpred `driveP`
light_formula = parse(
    "L := $light() ($driveP(L) -> equals(L, green))"
)
result = light_formula.eval(prob_framework, traffic_light_model, valuation)
print("Probability of driving when light is green (using mpred):", result)





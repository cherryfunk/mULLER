from muller.monad.distribution import Prob, weighted
from muller.nesy_framework import Interpretation

def _drive(l: str | bool) -> Prob[bool | str]:
    """Drive function based on traffic light color."""
    if l == "red":
        return weighted([(True, 0.1), (False, 0.9)])
    elif l == "yellow":
        return weighted([(True, 0.2), (False, 0.8)])
    elif l == "green":
        return weighted([(True, 0.9), (False, 0.1)])
    else:
        return Prob({})

traffic_light_model = Interpretation(
    universe=["red", "green", "yellow", False, True],
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

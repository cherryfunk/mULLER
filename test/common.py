from typing import Callable, Literal, cast, get_args

from muller.monad.base import ParametrizedMonad
from muller.monad.distribution import Prob, weighted
from muller.nesy_framework import Interpretation

TrafficLightUniverse = Literal["red", "green", "yellow", False, True]
traffic_light_universe = list(get_args(TrafficLightUniverse))

def _drive(light: TrafficLightUniverse) -> ParametrizedMonad[TrafficLightUniverse]:
    """Drive function based on traffic light color."""
    if light == "red":
        return weighted([(True, 0.1), (False, 0.9)])
    elif light == "yellow":
        return weighted([(True, 0.2), (False, 0.8)])
    elif light == "green":
        return weighted([(True, 0.9), (False, 0.1)])
    else:
        return Prob({})



traffic_light_model: Interpretation[TrafficLightUniverse, bool] = Interpretation(
    universe=traffic_light_universe,
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
        "eval": lambda x: isinstance(x, bool) and x,
    },
    mpreds={
        "driveP": cast(Callable[[TrafficLightUniverse], ParametrizedMonad[bool]], _drive)
    },
)

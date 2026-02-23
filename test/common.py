from typing import Literal

from muller.framework.interpretation import Interpretation
from muller.framework.nesy import BaseNeSyFramework
from muller.hkt import List
from muller.monad.distribution import Dist, weighted
from muller.nesy_framework import nesy

Universe = Literal["red", "green", "yellow", False, True]
universe: list[Universe] = ["red", "green", "yellow", False, True]

nf: BaseNeSyFramework[Dist[bool], bool, List[Universe], Universe] = nesy(Dist, bool, List, Universe)


def _drive(light: Universe) -> Dist[bool]:
    """Drive function based on traffic light color."""
    if light == "red":
        return weighted([(True, 0.1), (False, 0.9)])
    elif light == "yellow":
        return weighted([(True, 0.2), (False, 0.8)])
    elif light == "green":
        return weighted([(True, 0.9), (False, 0.1)])
    else:
        return Dist({})


traffic_light_model: Interpretation[
    Dist[bool], bool, List[Universe], Universe
] = nf.create_interpretation(
    sort=List(universe),
    functions={
        "green": lambda _args: "green",
        "red": lambda _args: "red",
        "yellow": lambda _args: "yellow",
    },
    mfunctions={
        "light": lambda _args: weighted(
            [("red", 0.6), ("green", 0.3), ("yellow", 0.1)]
        ),
        "driveF": lambda args: _drive(args[0]),
    },
    predicates={
        "equals": lambda args: args[0] == args[1],
        "eval": lambda args: isinstance(args[0], bool) and args[0],
    },
    mpredicates={
        "driveP": lambda args: _drive(args[0]),
    },
)

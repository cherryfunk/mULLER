from muller.monad.distribution import Prob, weighted
from muller.nesy_framework import Interpretation


traffic_light_model = Interpretation(
        universe=["red", "green", "yellow", False, True],
        functions={
            "green": lambda: "green",
            "red": lambda: "red",
            "yellow": lambda: "yellow",
        },
        mfunctions={
            "light": lambda: weighted([("red", 0.3), ("green", 0.6), ("yellow", 0.1)]),
            "driveF": lambda l: (
                weighted([(True, 0.1), (False, 0.9)])
                if l == "red"
                else (
                    weighted([(True, 0.2), (False, 0.8)])
                    if l == "yellow"
                    else (
                        weighted([(True, 0.9), (False, 0.1)])
                        if l == "green"
                        else Prob({})
                    )
                )
            ),
        },
        preds={
            "equals": lambda a, b: a == b,
            "eval": lambda x: x == True,
        },
        mpreds={
            "driveP": lambda l: (
                weighted([(True, 0.1), (False, 0.9)])
                if l == "red"
                else (
                    weighted([(True, 0.2), (False, 0.8)])
                    if l == "yellow"
                    else (
                        weighted([(True, 0.9), (False, 0.1)])
                        if l == "green"
                        else Prob({})
                    )
                )
            )
        },
    )
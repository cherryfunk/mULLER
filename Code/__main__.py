from .double_semi_group_bounded_lattice import DistributionBoolDoubleSemiGroupBoundedLattice
from .distribution_monad import Prob, uniform
from .non_empty_powerset_monad import NonEmptyPowerset
from .monad import ParametrizedMonad
from .nesy_framework import Application, Computation, Conjunction, Interpretation, NeSyFramework, Predicate, Variable

if __name__ == "__main__":
    # dieModel :: Interpretation (Dist.T Double) DBool Integer
    die_model: Interpretation[int, bool] = Interpretation(
        universe=list(range(1, 7)),
        functions={str(i): lambda: i for i in range(1, 7)},
        mfunctions={"die": lambda: uniform(list(range(1, 7)))},
        preds= {
            "is_even": lambda x: x % 2 == 0,
            "is_odd": lambda x: x % 2 != 0,
            "==": lambda x, y: x == y,
            },
        mpreds={}
    )

    die_sen_1 = Computation("x", "die", [], (Conjunction(
        Predicate("==", [Variable("x"), Application("6", [])]),
        Predicate("is_even", [Variable("x")]),
    )))

    nesy = NeSyFramework(Prob, bool, DistributionBoolDoubleSemiGroupBoundedLattice())
    print(die_sen_1.eval(nesy, die_model, {}))
    
    
    die_sen_2 = Conjunction(
        Computation("x", "die", [], Predicate("==", [Variable("x"), Application("6", [])])),
        Computation("x", "die", [], Predicate("is_even", [Variable("x")])),
    )
    print(die_sen_2.eval(nesy, die_model, {}))
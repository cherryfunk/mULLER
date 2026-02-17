from typing import Any, Callable

from muller import List, Prob, nesy, parse, uniform


def const[T](value: T) -> Callable[[list[Any]], T]:
    """Helper to create constant functions for interpretation."""
    return lambda _: value

nf = nesy(Prob, bool, List, int)

i = nf.create_interpretation(
    sort=List(range(1, 7))
)
i.functions = {str(n): const(n) for n in range(1, 7)}

@i.comp_fn()
def die() -> Prob[int]:
    return uniform(list(range(1, 7)))

@i.pred()
def equals(x: int, y: int) -> bool:
    return x == y

@i.pred()
def even(x: int) -> bool:
    return x % 2 == 0

dice_formula = parse("X := $die() (equals(X, 6) and even(X))")
result = nf.eval(dice_formula, i)
# result = dice_formula.eval(nf, interpretation, valuation)
print("Probability of rolling a 6 that is even:", result)

dice_formula = parse("X := $die() (equals(X, 6)) and X := $die() (even(X))")
# result = dice_formula.eval(prob_framework, interpretation, valuation)
result = nf.eval(dice_formula, i)
print("Probability of rolling a 6 and of rolling an even number:", result)

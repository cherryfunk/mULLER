from muller import Interpretation, parse, uniform, nesy, Prob

prob_framework = nesy(Prob, bool)

interpretation = Interpretation(
    universe=list(range(7)),
    functions={str(i): lambda i=i: i for i in range(1, 7)},
    mfunctions={"die": lambda: uniform(list(range(1, 7)))},
    preds={"equals": lambda x, y: x == y, "even": lambda x: x % 2 == 0},
    mpreds={},
)

valuation = {}

dice_formula = parse("X := $die() (equals(X, 6) and even(X))")
result = dice_formula.eval(prob_framework, interpretation, valuation)
print("Probability of rolling a 6 that is even:", result)

dice_formula = parse("X := $die() (equals(X, 6)) and X := $die() (even(X))")
result = dice_formula.eval(prob_framework, interpretation, valuation)
print("Probability of rolling a 6 and of rolling an even number:", result)

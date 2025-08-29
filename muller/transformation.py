from typing import cast

from muller.monad.distribution import Prob
from muller.monad.non_empty_powerset import NonEmptyPowerset
from muller.nesy_framework import Interpretation, NeSyTransformer


# Argmax transformation
def maximal_values[T](prob_dist: Prob[T]) -> NonEmptyPowerset[T]:
    """Extract maximal probability values from distribution"""
    max_vals = prob_dist.argmax()
    return NonEmptyPowerset(max_vals)


class argmax[A](NeSyTransformer[A, bool, list[A], bool, list[A]]):
    def __call__(
        self, interpretation: Interpretation[A, bool, list[A]]
    ) -> Interpretation[A, bool, list[A]]:
        # Transform monadic functions
        transformed_mfunctions = {}
        for name, func in interpretation.mfunctions.items():

            def make_argmax_function(original_func):
                def argmax_wrapper(*args):
                    prob_result = original_func(*args)
                    return maximal_values(cast(Prob[A], prob_result))

                return argmax_wrapper

            transformed_mfunctions[name] = make_argmax_function(func)

        # Transform monadic predicates
        transformed_mpreds = {}
        for name, pred in interpretation.mpreds.items():

            def make_argmax_predicate(original_pred):
                def argmax_wrapper(*args):
                    prob_result = original_pred(*args)
                    return maximal_values(cast(Prob[bool], prob_result))

                return argmax_wrapper

            transformed_mpreds[name] = make_argmax_predicate(pred)

        return Interpretation(
            universe=interpretation.universe,
            functions=interpretation.functions,
            mfunctions=transformed_mfunctions,
            preds=interpretation.preds,
            mpreds=transformed_mpreds,
        )


__all__ = ["argmax"]

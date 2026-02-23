from typing import Any, Callable

from muller.framework.interpretation import _Interpretation, Interpretation
from muller.monad.distribution import Dist
from muller.monad.non_empty_powerset import NonEmptyPowerset
from muller.nesy_framework import SingleArgumentTypeFunction


# Argmax transformation
def maximal_values[T](prob_dist: Dist[T]) -> NonEmptyPowerset[T]:
    """Extract maximal probability values from distribution"""
    max_vals = prob_dist.argmax()
    return NonEmptyPowerset(max_vals)


class argmax[A]:
    def __call__(
        self, interpretation: Interpretation[Any, bool, Any, A]
    ) -> Interpretation[Any, bool, Any, A]:
        # Transform monadic functions
        transformed_mfunctions: dict[
            str, SingleArgumentTypeFunction[A, Any]
        ] = {}
        for name, func in interpretation.mfunctions.items():

            def make_argmax_function(
                original_func: Callable[[list[A]], Dist[A]],
            ) -> SingleArgumentTypeFunction[A, NonEmptyPowerset[A]]:
                def argmax_wrapper(args: list[A]) -> NonEmptyPowerset[A]:
                    prob_result: Dist[A] = original_func(args)
                    return maximal_values(prob_result)

                return argmax_wrapper

            transformed_mfunctions[name] = make_argmax_function(func)

        # Transform monadic predicates
        transformed_mpredicates: dict[
            str, SingleArgumentTypeFunction[A, Any]
        ] = {}
        for name, pred in interpretation.mpredicates.items():

            def make_argmax_predicate(
                original_pred: Callable[[list[A]], Dist[bool]],
            ) -> SingleArgumentTypeFunction[A, NonEmptyPowerset[bool]]:
                def argmax_wrapper(
                    args: list[A],
                ) -> NonEmptyPowerset[bool]:
                    prob_result: Dist[bool] = original_pred(args)
                    return maximal_values(prob_result)

                return argmax_wrapper

            transformed_mpredicates[name] = make_argmax_predicate(pred)

        return _Interpretation(
            sort=interpretation.sort,
            functions=interpretation.functions,
            mfunctions=transformed_mfunctions,
            predicates=interpretation.predicates,
            mpredicates=transformed_mpredicates,
        )


__all__ = ["argmax"]

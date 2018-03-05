"""
Module with Surprise-compatible algorithm for combining multiple Suprise-
algorithms.
"""
from itertools import chain

import operator
from collections import namedtuple
from functools import reduce
from surprise import AlgoBase, PredictionImpossible


"""
Tuple representing an algorithm and its weight.

Attributes:
    algorithm: Instance of an algorithm which inherits from AlgoBase.
    weight: How much to weight this algorithm, compared to all other algorithms.
        Set this to float('inf') to make it behave like a filter (multiplying
        instead of adding).
"""
AlgorithmTuple = namedtuple('AlgorithmTuple', ['algorithm', 'weight'])


AlgorithmResult = namedtuple('AlgorithmResult',
                             ['algorithm', 'weight', 'prediction', 'extra'])


class Hybrid(AlgoBase):
    """
    Algorithm combining multiple algorithms into a hybrid system.
    """

    def __init__(self, algorithms, **kwargs):
        """
        Set up which algorithms make up this

        Args:
            algorithms: List of AlgorithmTuple. Each tuple consists of an
                algorithm instance and a weight.
            **kwargs: Extra keyword arguments for the AlgoBase constructor.
        """
        super().__init__(**kwargs)
        self.algorithms = algorithms

        weights = map(lambda a: a.weight, algorithms)
        weights_without_inf = filter(lambda w: w != float('inf'), weights)
        self.sum_weights = sum(weights_without_inf)
        self.trainset = None

    def fit(self, trainset):
        self.trainset = trainset
        for algorithm, _ in self.algorithms:
            algorithm.fit(trainset)
        return self

    def estimate(self, u, i):
        # Let each algorithm make its prediction, and register the result
        results = []
        total_weights = self.sum_weights
        rejected_results = []
        for algorithm, weight in self.algorithms:
            try:
                this_result = algorithm.estimate(u, i)
                extras = None
                if isinstance(this_result, tuple):
                    extras = this_result[1]
                    this_result = this_result[0]
                results.append(AlgorithmResult(
                    algorithm,
                    weight,
                    this_result,
                    extras
                ))
            except PredictionImpossible as e:
                rejected_results.append(AlgorithmResult(
                    algorithm,
                    weight,
                    None,
                    e
                ))
                # Don't use this algorithm's weight when weighting
                if weight != float('inf'):
                    total_weights -= weight

        if not results:
            raise PredictionImpossible('No algorithm could give a result')

        normal_results = filter(lambda r: r.weight != float('inf'), results)
        filter_results = filter(lambda r: r.weight == float('inf'), results)

        def weight_prediction(algo_res):
            algorithm, weight, prediction, extra = algo_res
            # This is how we weight: Imagine a cake diagram, with each algorithm
            # having a portion of the cake corresponding with their weight.
            # One with weight 2 has twice as much as one with weight 1.
            # Algorithms which did not produce a result, are not considered.
            # The algorithms contribute to the final prediction according to
            # their weight.
            normalized_prediction = (prediction * weight) / total_weights
            return AlgorithmResult(algorithm, weight, normalized_prediction,
                                   extra)

        def weight_filtering_prediction(algo_res):
            algorithm, weight, prediction, extra = algo_res
            # Normalize
            lower, upper = self.trainset.rating_scale
            normalized_prediction = (prediction - lower) / (upper - lower)
            return AlgorithmResult(algorithm, weight, normalized_prediction,
                                   extra)

        weighted_results = map(weight_prediction, normal_results)
        weighted_predictions = map(lambda r: r.prediction, weighted_results)
        summarized_predictions = sum(weighted_predictions)

        # Normalize/weight the infinity weighted algorithms
        weighted_filtering_results = map(
            weight_filtering_prediction,
            filter_results
        )
        # We're only interested in the prediction
        filter_predictions = map(
            lambda r: r.prediction,
            weighted_filtering_results
        )

        # Multiply all infinity weighted algorithms and the sum of normal pred.
        prediction = reduce(
            operator.mul,
            chain([summarized_predictions], filter_predictions),
            1.0,
        )

        # Create extras, so you can inspect the individual results
        # (this might take too much memory?)
        all_results = chain(
            weighted_results,
            weighted_filtering_results,
            rejected_results
        )

        def use_class_name(result):
            algorithm, weight, prediction, extra = result
            name = type(algorithm).__name__
            return AlgorithmResult(name, weight, prediction, extra)

        all_results_with_algoname = map(use_class_name, all_results)
        extras = {r.algorithm: r for r in all_results_with_algoname}

        return prediction, extras

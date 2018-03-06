"""
Module with Surprise-compatible algorithm for combining multiple Surprise-
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
        Set up which algorithms make up this hybrid algorithm.

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
        # Propagate the fit call to all algorithms that make up this algorithm
        self.trainset = trainset
        for algorithm, _ in self.algorithms:
            algorithm.fit(trainset)
        return self

    def estimate(self, u, i):
        # Let each algorithm make its prediction, and register the result
        results, rejected_results, total_weights = self.run_child_algos(u, i)

        # We have two types of results, each of which is used differently.
        # Normal results are weighted, filter_results are made into numbers
        # in range [0,1] and are then multiplied with the prediction.
        # For example, an algorithm could give lower value to older articles.
        normal_results = filter(lambda r: r.weight != float('inf'), results)
        filter_results = filter(lambda r: r.weight == float('inf'), results)

        # First, we concentrate on the "normal" results
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

        # Weight the predictions according to the weight and total_weights
        weighted_results = tuple(map(weight_prediction, normal_results))
        # Throw away everything but the weighted prediction
        weighted_predictions = map(lambda r: r.prediction, weighted_results)
        # Our prediction (so far) is the sum of all weighted predictions
        summarized_predictions = sum(weighted_predictions)

        # Next, we turn to the results that act as filters
        def weight_filtering_prediction(algo_res):
            algorithm, weight, prediction, extra = algo_res
            # Normalize so we get a number between 0 and 1.
            # We assume the algorithm uses the entire rating_scale from the
            # training set.
            lower, upper = self.trainset.rating_scale
            normalized_prediction = (prediction - lower) / (upper - lower)
            return AlgorithmResult(algorithm, weight, normalized_prediction,
                                   extra)

        # Normalize/weight the infinity weighted algorithms
        weighted_filtering_results = tuple(map(
            weight_filtering_prediction,
            filter_results
        ))
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
        extras = self._create_extras(
            weighted_results,
            weighted_filtering_results,
            rejected_results
        )

        return prediction, extras

    def run_child_algos(self, u, i):
        """
        Run all algorithms that make up this hybrid algorithm.

        Args:
            u: Inner user ID.
            i: Inner item ID.

        Returns:
            tuple of results, rejected_results and total_weights.
            results is a list of PredictionResult for algorithms that made a
            prediction. rejected_results is a list of PredictionResult for
            algorithms that did not make a prediction. total_weights is the
            total number of weights for the algorithms that did succeed at
            giving a prediction.
        """
        # All results which the algorithm was able to produce
        results = []
        # Total weight of all algorithms that produced a result
        total_weights = self.sum_weights
        # Algorithms that failed to produce a result
        rejected_results = []

        for algorithm, weight in self.algorithms:
            # First, let's try to calculate using this algorithm
            try:
                this_result = algorithm.estimate(u, i)
                # Algorithms may either return the prediction alone, or a tuple
                # of (prediction, extras). Assume the first case is true.
                extras = None
                if isinstance(this_result, tuple):
                    # Turns out it's the second case, fix this_result and extras
                    extras = this_result[1]
                    this_result = this_result[0]

                if this_result == self.trainset.global_mean:
                    # Though the algorithm did not admit it, it failed to
                    # produce a result different than the global mean (a symptom
                    # that a prediction was impossible)
                    raise PredictionImpossible(
                        'Algorithm prediction equals global mean'
                    )

                # If we are here, the algorithm managed to produce a result!
                results.append(AlgorithmResult(
                    algorithm,
                    weight,
                    this_result,
                    extras
                ))

            except PredictionImpossible as e:
                # The algorithm failed! Register it as such
                rejected_results.append(AlgorithmResult(
                    algorithm,
                    weight,
                    None,
                    e
                ))
                # Don't use this algorithm's weight when weighting
                if weight != float('inf'):
                    total_weights -= weight

        # Did any algorithm succeed at predicting?
        if not results or total_weights == 0:
            raise PredictionImpossible('No algorithm could give a result')

        return results, rejected_results, total_weights

    @staticmethod
    def _create_extras(*results):
        """
        Create the extras dictionary for this prediction.

        The extras dictionary can be used to inspect and better understand how
        we made a prediction.

        Args:
            *results: Iterables of AlgorithmResult

        Returns:
            Dictionary where key is the algorithm's class name, and the value
            is AlgorithmResult except the algorithm value is the class name and
            not the instance of the algorithm.
        """
        all_results = chain(*results)

        def use_class_name(result):
            algorithm, weight, prediction, extra = result
            name = type(algorithm).__name__
            return AlgorithmResult(name, weight, prediction, extra)

        all_results_with_algoname = map(use_class_name, all_results)
        extras = {r.algorithm: r for r in all_results_with_algoname}
        return extras


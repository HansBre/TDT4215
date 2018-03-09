"""
Module with Surprise-compatible algorithm for combining multiple Surprise-
algorithms.
"""
from itertools import chain

import gc
from os import path, getpid, remove
import operator
from glob import glob
from collections import namedtuple
from functools import reduce
from surprise import AlgoBase, PredictionImpossible, dump


"""
Tuple representing an algorithm and its weight.

Attributes:
    algorithm: Instance of an algorithm which inherits from AlgoBase.
    weight: How much to weight this algorithm, compared to all other algorithms.
        Set this to float('inf') to make it behave like a filter (multiplying
        instead of adding).
"""
AlgorithmTuple = namedtuple('AlgorithmTuple', ['algorithm', 'weight'])


AlgorithmResult = namedtuple(
    'AlgorithmResult',
    ['algorithm_name', 'weight', 'prediction', 'extra']
)


class Hybrid(AlgoBase):
    """
    Algorithm combining multiple algorithms into a hybrid system.
    """

    def __init__(self, algorithms, spool_dir=None, **kwargs):
        """
        Set up which algorithms make up this hybrid algorithm.

        To use the memory saving techniques, you must ensure the following:
        1. Do not hold any of the given algorithms in a variable. If you need to
           to so, make sure you delete that variable (using the del statement)
           before using the Hybrid instance. Without this, the algorithm
           instance will not be garbage collected and will stay in memory.
        2. Specify a spool_dir located on disk. This should be an existing
           directory which the process can write files to. The process may share
           this directory with other processes running the Hybrid algorithm.
           A directory under /tmp may work, but on some configurations /tmp is
           located in memory, making the whole memory saving thing pointless.
           You should choose another directory on such systems.

        Args:
            algorithms: List of AlgorithmTuple. Each tuple consists of an
                algorithm instance and a weight.
            spool_dir: Path to directory where algorithms can be stored on disk.
                The default disables memory conserving techniques.
            **kwargs: Extra keyword arguments for the AlgoBase constructor.
        """
        super().__init__(**kwargs)
        self._algorithms = list(algorithms)
        self.spool_dir = spool_dir

        weights = map(lambda a: a.weight, algorithms)
        weights_without_inf = filter(lambda w: w != float('inf'), weights)
        self.sum_weights = sum(weights_without_inf)
        self.trainset = None

    def _write(self, index: int) -> None:
        """
        Write the algorithm at the given index to disk.

        The algorithm stays in memory after the write.

        If self.spool_dir is None, this method does nothing.

        Args:
            index: Number identifying the algorithm instance. This is the third
                argument yielded by all_algorithms().

        Returns:
            Nothing.
        """
        if self.spool_dir:
            dump.dump(
                self._file_for(index),
                algo=self._algorithms[index].algorithm
            )

    def _close(self, index: int) -> None:
        """
        Remove the algorithm from memory.

        The algorithm is NOT written to disk automatically. You must call
        self._write(index) yourself.

        Make sure you have no references to the algorithm before calling this,
        as it will try to garbage collect the algorithm instance.

        If self.spool_dir is None, this method does nothing.

        Internally, the self._algorithms list contains the AlgorithmTuple when
        the algorithm is loaded, and a function which loads and returns the
        AlgorithmTuple when called if the algorithm is not stored in memory.
        This function replaces the current entry in self._algorithms with such
        a function, and runs the garbage collector afterwards to (hopefully)
        collect the algorithm instance.

        Args:
            index: Number identifying the algorithm instance. This is the third
                argument yielded by all_algorithms().
        """
        if not self.spool_dir:
            return

        weight = self._algorithms[index].weight

        # This function is called when the algorithm should be loaded again:
        def load():
            _, loaded_algo = dump.load(self._file_for(index))
            return AlgorithmTuple(loaded_algo, weight)

        self._algorithms[index] = load
        # We no longer reference the algorithm, so try to garbage collect it
        gc.collect()

    def _open(self, index: int) -> AlgorithmTuple:
        """
        Fetch the given algorithm.

        This method automatically loads the algorithm from disk if this is
        necessary. Otherwise, it is simply returned.

        If self.spool_dir is None, this method will not try to load from disk.

        Args:
            index: Number identifying the algorithm instance. This is the third
                argument yielded by all_algorithms().

        Returns:
            The AlgorithmTuple associated with the given index.
        """
        if not self.spool_dir:
            return self._algorithms[index]

        instance = self._algorithms[index]

        if callable(instance):
            self._algorithms[index] = instance()

        return self._algorithms[index]

    def _file_for(self, index):
        """
        Create filename for where to store the identified algorithm.

        Args:
            index: Number identifying the algorithm instance. This is the third
                argument yielded by all_algorithms().

        Returns:
            Absolute path to where the given algorithm should be stored.
        """
        abs_dir = path.abspath(self.spool_dir)
        filename = "serialized_{pid}_{index}"\
            .format(pid=getpid(), index=index)
        return path.join(abs_dir, filename)

    def all_algorithms(self):
        """
        Iterate over all the algorithms in a memory-conserving manner.

        Each algorithm will be loaded from the file on disk, if it's not in
        memory already. It will then be returned from the generator. Afterwards,
        the algorithm will be removed from memory (without saving to disk).

        Make sure you don't have a hanging reference to the algorithm after
        you are done, since that will hinder our ability to remove it from
        memory.

        If you wish to persist any changes to the returned algorithm, you must
        call self._write(index) yourself.

        Example:
            for algorithm, weight, index in self.all_algorithms():
                print(self._get_algorithm_name(algorithm), weight, index)

        If self.spool_dir is None, this will simply iterate over the algorithms
        in memory.

        Yields:
            Tuple of (algorithm, weight, index).
            algorithm is the instance of the algorithm, subclass of AlgoBase.
            weight is an integer or float('inf') detailing how much to weight
                this algorithm's prediction.
            index is the index of this algorithm in the underlying list, and is
                used to identify it with the _open, _write and _close methods.
        """
        for index in range(len(self._algorithms)):
            yield (*self._open(index), index)
            self._close(index)

    def cleanup(self) -> None:
        """Remove stored instances of algorithms from disk."""
        if not self.spool_dir:
            return

        # Here, we do some magic. We want to remove all files, no matter which
        # index they have. We therefore use the wildcard as the index, which
        # matches all indices when used with glob.
        matching_files = glob(self._file_for('*'))
        for file in matching_files:
            remove(file)

    def fit(self, trainset):
        # Propagate the fit call to all algorithms that make up this algorithm
        self.trainset = trainset
        for (index, (algorithm, _)) in enumerate(self.all_algorithms()):
            algorithm.fit(trainset)
            self._write(index)
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

        for algorithm, weight, _ in self.all_algorithms():
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
                    self._get_algorithm_name(algorithm),
                    weight,
                    this_result,
                    extras
                ))

            except PredictionImpossible as e:
                # The algorithm failed! Register it as such
                rejected_results.append(AlgorithmResult(
                    self._get_algorithm_name(algorithm),
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
            is AlgorithmResult.
        """
        all_results = chain(*results)
        extras = {r.algorithm: r for r in all_results}
        return extras

    @staticmethod
    def _get_algorithm_name(algo):
        return type(algo).__name__

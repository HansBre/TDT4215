"""
Module with Surprise-compatible algorithm for combining multiple Surprise-
algorithms.
"""
from itertools import chain

import gc
from os import path, getpid, remove
import operator
from glob import glob
from collections import namedtuple, defaultdict
from functools import reduce
from surprise import AlgoBase, PredictionImpossible, dump, Prediction

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

JobRequest = namedtuple(
    'JobRequest',
    ['uid', 'iid', 'r_ui', 'iuid', 'iiid'],
)

JobResult = namedtuple(
    'JobResult',
    ['uid', 'iid', 'r_ui', 'est'],
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
        self.job_requests = []
        self.estimations = dict()
        self.job_responses = []

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
        for algorithm, _, index in self.all_algorithms():
            algorithm.fit(trainset)
            self._write(index)
        return self

    def test(self, testset, verbose=False):
        # We override the default test method so that we can do ask the same
        # algorithm to predict multiple cases at once, instead of constantly
        # switching between algorithms. We do this by not using predict() and
        # estimate() at all, re-implementing predict() inside create_job() and
        # process_result() and re-implementing estimate inside
        # run_child_algos_on_jobs() and run_combiner().

        # First, translate the raw user ID and item ID into the internal user
        # ID and item ID (this is the same that the original test and predict
        # does).
        jobs = []
        for (uid, iid, r_ui_trans) in testset:
            jobs.append(self.create_job(uid, iid, r_ui_trans - self.trainset.offset))

        # Next, iterate through each algorithm. For each algorithm, get its
        # prediction for each job created above.
        results = self.run_child_algos_on_jobs(jobs)

        # Now, for each job created above, we combine the results from the
        # different algorithms.
        job_responses = self.run_combiner(jobs, results)
        del jobs

        # Finally, we go from what combine() returns to Prediction objects
        predictions = [self.process_result(result)
                       for result in job_responses]
        return predictions

    def create_job(self, uid, iid, r_ui) -> JobRequest:
        """
        Create a JobRequest, translating from raw to internal user and item IDs.

        Args:
            uid: Raw user ID.
            iid: Raw item ID.
            r_ui: Actual result.

        Returns:
            Instance of JobRequest with all values set.
        """
        # Adaptation of first part of predict() from AlgoBase
        try:
            iuid = self.trainset.to_inner_uid(uid)
        except ValueError:
            iuid = 'UKN__' + str(uid)

        try:
            iiid = self.trainset.to_inner_iid(iid)
        except ValueError:
            iiid = 'UKN__' + str(iid)

        return JobRequest(uid, iid, r_ui, iuid, iiid)

    def run_child_algos_on_jobs(self, jobs):
        """
        Collect each algorithm's prediction for each job.

        Args:
            jobs: List of JobRequest. These are the user/item pairs we want to
                collect predictions for.

        Returns:
            Dict where key is (inner user ID, inner item ID) and value is a
            dictionary consisting of results, total_weights and
            rejected_results, as expected by combine().
        """
        def create_empty_result_dict():
            return {
                'results': [],
                'total_weights': self.sum_weights,
                'rejected_results': []
            }
        # TODO: Use a list instead with indices matching those of jobs,
        # since the same user ID and item ID pair may appear multiple times
        results = defaultdict(create_empty_result_dict)

        # Go though one algorithm at a time
        for algorithm, weight, _ in self.all_algorithms():
            # Don't fetch the name for every job
            algorithm_name = self._get_algorithm_name(algorithm)
            # Iterate through the job requests, and make a prediction for each
            for job in jobs:
                u = job.iuid
                i = job.iiid
                key = (u, i)

                try:
                    this_result = algorithm.estimate(u, i)
                    # Did we get just prediction or prediction and extras?
                    extras = None
                    if isinstance(this_result, tuple):
                        this_result, extras = this_result

                    if this_result == self.trainset.global_mean:
                        # Though the algorithm did not admit it, it failed to
                        # produce a result different than the global mean (a
                        # symptom that a prediction was impossible)
                        raise PredictionImpossible(
                            'Algorithm prediction equals global mean'
                        )

                    # If we are here, the algorithm managed to produce a result!
                    results[key]['results'].append(AlgorithmResult(
                        algorithm_name,
                        weight,
                        this_result,
                        extras
                    ))
                except PredictionImpossible as e:
                    # The algorithm failed! Register it as such
                    results[key]['rejected_results'].append(AlgorithmResult(
                        algorithm_name,
                        weight,
                        None,
                        e
                    ))
                    # Don't use this algorithm's weight when weighting
                    if weight != float('inf'):
                        results[key]['total_weights'] -= weight
        # Make it so results throws KeyError when non-existing key is accessed
        results.default_factory = None
        return results

    def run_combiner(self, jobs, algorithm_results):
        """
        Combine predictions per job so we have one hybrid prediction.

        Args:
            jobs: List of JobRequest. User/item pairs we want to predict.
            algorithm_results: Dictionary of dictionaries, as returned by
                run_child_algos_on_jobs().

        Returns:
            List of JobResult, that are the final output of the hybrid
                recommender system.
        """
        job_results = []
        for job in jobs:
            key = (job.iuid, job.iiid)
            this_result = algorithm_results[key]
            results = this_result['results']
            rejected_results = this_result['rejected_results']
            total_weights = this_result['total_weights']

            try:
                est = self.combine(results, rejected_results, total_weights)
            except PredictionImpossible as e:
                est = e
            job_results.append(self.create_job_result(job, est))
        return job_results

    @staticmethod
    def create_job_result(job_request, est):
        """
        Create JobResult using JobRequest and an estimate.

        Args:
            job_request: Instance of JobRequest, which this estimate pertains
                to.
            est: The estimate, in same format as the return value of estimate().

        Returns:
            Instance of JobResult for this job_request and est.
        """
        uid, iid, r_ui, iuid, iiid = job_request
        return JobResult(uid, iid, r_ui, est)

    def process_result(self, job_response):
        """
        Transform the return values of combine() into Prediction.

        This is equal to the post-processing in predict().

        Args:
            job_response: Instance of JobResponse.

        Returns:
            List of Prediction.
        """
        # Adaptation of second part of predict() from AlgoBase
        uid, iid, r_ui, est = job_response
        details = {}
        if isinstance(est, PredictionImpossible):
            error = str(est)
            est = self.trainset.global_mean
            details['was_impossible'] = True
            details['reason'] = error
        else:
            if isinstance(est, tuple):
                est, details = est

            details['was_impossible'] = False

        # Remap the rating into its initial rating scale
        est -= self.trainset.offset

        # clip estimate
        lower_bound, higher_bound = self.trainset.rating_scale
        est = min(higher_bound, est)
        est = max(lower_bound, est)

        return Prediction(uid, iid, r_ui, est, details)

    def combine(self, results, rejected_results, total_weights):
        """
        Combine the predictions from different algorithms into one estimate.

        Args:
            results: List of AlgorithmResult which produced a prediction.
            rejected_results: List of AlgorithmResult which did not produce a
                prediction.
            total_weights: Integer, sum of weights for all results in results.

        Returns:
            Tuple of estimate and extras, like AlgoBase.estimate().
        """
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
        extras = {r.algorithm_name: r for r in all_results}
        return extras

    @staticmethod
    def _get_algorithm_name(algo):
        return type(algo).__name__

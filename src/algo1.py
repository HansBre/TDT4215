import argparse
from os import path
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import KFold
import statistics
import csv
from tools.prediction_tools import precision_recall_at_k, get_f1
from sklearn.metrics import roc_auc_score
import numpy as np
from src.hybrid import Hybrid, AlgorithmTuple


parser = argparse.ArgumentParser(
    description='Make predictions using the SVD algorithm. '
                'Prints median prediction error per split. '
                'The active time is the variable being predicted.',
)
parser.add_argument(
    '--input', '-i',
    help='Preprocessed file with user, item and active time. '
         '(Default: %(default)s)',
    default='dataset1.txt',
)
parser.add_argument(
    '--csv', '-o',
    help='When given, create a comma separated values file suitable for '
         'programs like Excel and Libreoffice Calc. Includes item and user ID,'
         'original active time, predicted active time and difference between '
         'those two.',
    default=False,
)

parser.add_argument(
    '--metrics', '-m',
    help='When given prints metrics',
    action='store_true',
)

parser.add_argument(
    '--verbose', '-v',
    help='Verbose',
    action='store_true',
)

args = parser.parse_args()

reader = Reader(line_format='user item rating', sep='\t')
data = Dataset.load_from_file(args.input, reader=reader)

algo = Hybrid(
    (
        # Singular Value Decomposition (SVD) is used for this example
        AlgorithmTuple(SVD(), 1),
    ),
    path.join(path.dirname(path.dirname(__file__)), 'spool'),
)

n_folds = 5
kf = KFold(n_splits=n_folds)

if args.csv:
    output_file = open(args.csv, 'wt', newline='')
    try:
        fields = ['uid', 'iid', 'estimated', 'actual', 'error']
        csv_writer = csv.DictWriter(output_file, fieldnames=fields,
                                    dialect='unix')
        csv_writer.writeheader()
    except Exception:
        output_file.close()
        raise

try:
    if args.verbose:
        print("Starting Folding")
    n_fold = 0
    sum_precision, sum_recall, sum_f1, sum_roc_auc_score = 0, 0, 0, 0
    for trainset, testset in kf.split(data):
        n_fold += 1
        if args.verbose:
            print("Starting Fold", n_fold)
        algo.fit(trainset)
        predictions = algo.test(testset)

        if args.csv:
            def float2csv(f):
                return str(f).replace('.', ',')

            for p in predictions:
                csv_writer.writerow({
                    'uid': p.uid,
                    'iid': p.iid,
                    'estimated': float2csv(p.est),
                    'actual': float2csv(p.r_ui),
                    'error': float2csv(abs(p.est-p.r_ui)),
                })
        else:
            errors = map(lambda p: abs(p.r_ui - p.est), predictions)
            if args.verbose:
                print("Avg. Rating Error:", statistics.median(errors))

        if args.metrics:
            # Evaluate Metrics
            precision_pr_user, recall_pr_user = precision_recall_at_k(predictions)
            fold_average_precision = np.average(list(precision_pr_user.values()))
            fold_average_recall = np.average(list(recall_pr_user.values()))
            fold_average_f1 = get_f1(fold_average_precision, fold_average_recall)
            # Build prediction sets grouped by user for roc_auc metric evaluation
            true, test = [], []
            users_predictions = dict()
            for prediction in predictions:
                true.append(prediction.r_ui)
                test.append(prediction.est)
                if prediction.uid in users_predictions.keys():
                    users_predictions[prediction.uid].append(prediction)
                else:
                    users_predictions[prediction.uid] = [prediction]
            # fold_roc_auc_score = roc_auc_score(true, test)

            # MHRH TODO
            # for user_id, predictions in users_predictions.items():
                # predictions.sort(key=lambda x: x.est, reverse=True)
                # Now we chan calculate mhrh over the first k elements in the sorted list.

            sum_precision += fold_average_precision
            sum_recall += fold_average_recall
            sum_f1 += fold_average_f1
            # sum_roc_auc_score += fold_roc_auc_score

    if args.metrics:
        average_recall = sum_recall / n_folds
        average_precision = sum_precision / n_folds
        average_f1 = sum_f1 / n_folds
        average_roc_auc = sum_roc_auc_score / n_folds
        print("---RESULT---")
        print("Average Precision:", average_precision)
        print("Average Recall:", average_recall)
        print("Average F1:", average_f1)

        # print("Average ROC AUC:", average_roc_auc)

finally:
    if args.csv:
        output_file.close()
    if algo:
        algo.cleanup()

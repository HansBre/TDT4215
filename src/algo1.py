import argparse
from math import inf
from datetime import datetime

import random
from os import path
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import KFold, train_test_split

import statistics
import csv
from tools.prediction_tools import precision_recall_at_k, get_f1, get_top_n
from sklearn.metrics import roc_auc_score
import numpy as np
from src.hybrid import Hybrid, AlgorithmTuple, AlgorithmResult
from src.db import FulltextDb
from src.datefactor import DateFactor

parser = argparse.ArgumentParser(
    description='Make predictions using a hybrid algorithm.',
    add_help=False,
)
FulltextDb.populate_argparser(parser)
parser.add_argument(
    '--input', '-i',
    help='Preprocessed file with user, item and active time. '
         '(Default: %(default)s)',
    default='dataset1.txt',
)
parser.add_argument(
    '--csv', '-o',
    help='When given, create a comma separated values file suitable for '
         'programs like Excel and Libreoffice Calc. Includes item and user ID, '
         'original rating and predicted ratings from both hybrid and all '
         'included algorithms.',
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

parser.add_argument(
    '--seed', '-s',
    help='Random seed to use. Set this to have reproducible experiments, so '
         'that you can change one variable and keep everything else (folding, '
         'random initial variables) the same. By default, a new seed is used '
         'every time.',
    default=None,
    type=int,
)


def is_relevant(testset_by_user, uid, iid):
    if uid in testset_by_user and iid in testset_by_user[uid]:
        if testset_by_user[uid][iid] >= relevance_threshold:
            return True
    return False


args = parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)

reader = Reader(line_format='user item rating', sep='\t')
data = Dataset.load_from_file(args.input, reader=reader)

db = FulltextDb.create_from_args(args)
algo = Hybrid(
    (
        # Singular Value Decomposition (SVD) is used for this example
        AlgorithmTuple(SVD(), 1),
        AlgorithmTuple(DateFactor(
            db,
            cut_after=datetime(2017, 1, 8),  # Change if using three-week set
            oldest_date=datetime(2013, 1, 1),
            weight=0.5,
        ), inf)
    ),
    path.join(path.dirname(path.dirname(__file__)), 'spool'),
)
db.close()
del db

n_folds = 2
kf = KFold(n_splits=n_folds)

num_testing_batches = 10

# Threshold for saying an article is relevant for a user or not.
# If estimated rating is higher than threshold, we say article is relevant for that user, else not.
relevance_threshold = 2.5

# Get all article keywords
# keywords = dict()
# for line in open('article_keywords'):
# values = line.split('*-*')
# keywords[values[0]] = values[1]

trainset = data.build_full_trainset()

if args.csv:
    output_file = open(args.csv, 'wt', newline='')
    try:
        algorithms = ['SVD', 'DateFactor']
        fields = ['uid', 'iid', 'estimated', *algorithms, 'actual', 'error']
        csv_writer = csv.DictWriter(output_file, fieldnames=fields,
                                    dialect='unix')
        csv_writer.writeheader()
    except Exception:
        output_file.close()
        raise
"""
user_gen = trainset.all_users()
dic = {}
for user in user_gen:
    print(len(trainset.ur[user]))

"""
try:
    if args.verbose:
        print("Starting Folding")
    n_fold = 0
    sum_precision, sum_recall, sum_f1, sum_roc_auc_score, sum_ctr, sum_mhrh = 0, 0, 0, 0, 0, 0

    for trainset, testset in kf.split(data):
        n_fold += 1
        if args.verbose:
            print("Fold", n_fold)
        item_gen = trainset.all_items()
        user_gen = trainset.all_users()

        algo.fit(trainset)

        # Key articles by userIDS so we can retrieve them and merge them with anti testset later
        # After this dict is build it will contain for each user : (uid, articleID, rating) tuples for each article in
        # testset on that user
        testset_by_user = dict()
        articles_clicked_by_user = dict()
        for t in testset:

            # t[0] = uid, t[1] = article id, t[2] = true rating
            if t[0] in testset_by_user:
                testset_by_user[t[0]]['raw'].append(tuple(t))
            else:
                raw_tuples = [t]
                testset_by_user[t[0]] = dict()
                # Now we can find true rating from user and artcle id to determine relevance.
                testset_by_user[t[0]]['raw'] = raw_tuples
                articles_clicked_by_user[t[0]] = set()
            testset_by_user[t[0]][t[1]] = t[2]
            articles_clicked_by_user[t[0]].add(t[1])

        # This is where we'll store our top 10 predictions per user
        # ex. users_top_10
        users_top_10 = {}

        n = 0
        batch = 100
        b = 0
        fold_sum_mhrh = 0

        # while n < trainset.n_users:
        while n < (num_testing_batches * batch):
            batch_testset = []
            b += 1
            if (args.verbose):
                print('Starting building batch', b)
            for u in range(n, n + batch):
                try:
                    user_items = set([j for (j, _) in trainset.ur[u]])
                    batch_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), 0.0) for
                                      i in trainset.all_items() if
                                      i not in user_items]
                    try:
                        batch_testset += testset_by_user[trainset.to_raw_uid(u)]['raw']
                    except KeyError:
                        continue

                except ValueError:
                    # This means we have gone through all the users
                    break
            if (args.verbose):
                print('Starting testing')
            predictions = algo.test(batch_testset)
            
            if args.csv:
                def float2csv(f):
                    return str(f).replace('.', ',')

                for p in predictions:
                    details = p.details
                    individual_p = {res.algorithm_name: float2csv(res.prediction)
                                    for res in details.values()
                                    if isinstance(res, AlgorithmResult)}
                    csv_writer.writerow({
                        'uid': p.uid,
                        'iid': p.iid,
                        'estimated': float2csv(p.est),
                        **individual_p,
                        'actual': float2csv(p.r_ui),
                        'error': float2csv(abs(p.est-p.r_ui)),
                    })
                    
            if (args.verbose):
                print('Getting top_n')
            top_n = get_top_n(predictions, n=10)

            for uid, user_ratings in top_n.items():
                users_top_10[uid] = user_ratings

            if args.metrics:
                if (args.verbose):
                    print('Calculating metrics for batch')
                # Evaluate Metrics
                # CTR
                batch_sum_ctr = 0

                for uid in users_top_10:
                    clicks = 0
                    # Investigate correlations between recomennded articles keywords
                    # for iid, _ in users_top_10[uid]:
                    # print(keywords[iid])
                    user_mhrh_sum = 0
                    denominator = 1
                    for article_id, _ in users_top_10[uid]:  # mhrh & ctr
                        numerator = 1 if is_relevant(testset_by_user, uid, article_id) else 0
                        fraction = numerator / denominator
                        user_mhrh_sum += fraction
                        denominator += 1
                        fold_sum_mhrh += user_mhrh_sum
                        if uid in articles_clicked_by_user:
                            if article_id in articles_clicked_by_user[uid]:
                                clicks += 1
                    ctr = clicks / len(users_top_10[uid])
                    batch_sum_ctr += ctr
                fold_average_ctr = batch_sum_ctr / batch

                # Precision Recall
                precision_pr_user, recall_pr_user = precision_recall_at_k(predictions, k=10, threshold=relevance_threshold)
                fold_average_precision = np.average(list(precision_pr_user.values()))
                fold_average_recall = np.average(list(recall_pr_user.values())) if len(
                    list(recall_pr_user.values())) > 0 else 1
                fold_average_f1 = get_f1(fold_average_precision, fold_average_recall)

                # Build prediction sets grouped by user for roc_auc metric evaluation & MHRH
                true, test = [], []
                for prediction in predictions:
                    relevant = 1 if prediction.r_ui > relevance_threshold else 0
                    estimated_relevant = 1 if prediction.est > relevance_threshold else 0
                    true.append(relevant)
                    test.append(estimated_relevant)

                # Add for Kfold average
                sum_precision += fold_average_precision
                sum_recall += fold_average_recall
                sum_f1 += fold_average_f1
                sum_ctr += fold_average_ctr
                sum_mhrh += fold_sum_mhrh
                # sum_roc_auc_score += fold_roc_auc_score
            n += batch
        # print(round(n / trainset.n_users, 4), "%", end='       \r')
        # newline
        # print()

    if args.metrics:
        average_recall = sum_recall / (n_folds * num_testing_batches)
        average_precision = sum_precision / (n_folds * num_testing_batches)
        average_f1 = sum_f1 / (n_folds * num_testing_batches)
        average_roc_auc = sum_roc_auc_score / (n_folds * num_testing_batches)
        average_ctr = sum_ctr / (n_folds * num_testing_batches)
        average_mhrh = sum_mhrh / (n_fold * num_testing_batches * 100)
        print("---RESULT---")
        print("Average Precision:", average_precision)
        print("Average Recall:", average_recall)
        print("Average F1:", average_f1)
        print("Average ROC AUC:", average_roc_auc)
        print("Average CTR:", average_ctr)
        print("Average MHRH:", average_mhrh)

finally:
    if args.csv:
        output_file.close()
    if algo:
        algo.cleanup()
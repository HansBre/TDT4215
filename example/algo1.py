import argparse
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import KFold


parser = argparse.ArgumentParser(
    description='Make predictions using the SVD algorithm. '
                'Prints mean prediction error per split. '
                'The active time is the variable being predicted.',
)
parser.add_argument(
    '--input', '-i',
    help='Preprocessed file with user, item and active time. '
         '(Default: %(default)s)',
    default='dataset1.txt',
)

args = parser.parse_args()

reader = Reader(line_format='user item rating', sep='\t')
data = Dataset.load_from_file(args.input, reader=reader)

algo = SVD()  # Singular Value Decomposition (SVD) is used for this example

kf = KFold(n_splits=5)

for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)

    mean_error = abs(predictions[0].r_ui - predictions[0].est)
    for prediction in predictions:
        this_error = abs(prediction.r_ui - prediction.est)
        mean_error = (mean_error + this_error) / 2
    print(mean_error)




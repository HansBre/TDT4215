import argparse
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import KFold
import statistics
import csv


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

args = parser.parse_args()

reader = Reader(line_format='user item rating', sep='\t')
data = Dataset.load_from_file(args.input, reader=reader)

algo = SVD()  # Singular Value Decomposition (SVD) is used for this example

kf = KFold(n_splits=5)

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
    for trainset, testset in kf.split(data):
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
            print(statistics.median(errors))

finally:
    if args.csv:
        output_file.close()

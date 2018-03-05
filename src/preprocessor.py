import json
import os
import gzip
import argparse
from tools.rater import rate_article

parser = argparse.ArgumentParser(
    description='Preprocess the data set',
)
parser.add_argument(
    '--input', '-i',
    help='Data set file to process. Files ending with .gz will be decompressed '
         'only if the archive contains a single file. (Default: %(default)s)',
    default='test.data',
)
parser.add_argument(
    '--dataset1', '-1',
    type=argparse.FileType('wt', encoding='utf8'),
    help='File to write user, item and estimated rating to. (Default: %(default)s)',
    default='dataset1.txt',
)
parser.add_argument(
    '--dataset2','-2',
    type=argparse.FileType('wt', encoding='utf8'),
    help='File to write user, item and keywords to. (Default: %(default)s)',
    default='dataset2.txt',
)

args = parser.parse_args()

with args.dataset1 as f1, args.dataset2 as f2:
    print('>>>Start reading file...')

    # Should we decompress with Gzip?
    if args.input.endswith('.gz'):
        input_func = gzip.open
    else:
        input_func = open

    users = dict()

    print("Building Keyword preferences")
    with input_func(args.input, 'rt', encoding='utf8') as input_file:
        for line in input_file:
            obj = json.loads(line.strip())
            try:
                uid = obj['userId']
                if uid not in users:
                    users[uid] = dict()
                keywords = obj['keywords'] if 'keywords' in obj else 'None'
                active_time = str(obj['activeTime']) if 'activeTime' in obj else '0'
                if keywords is not 'None':
                    # If a user clicked on an article, we assume that means he is at least somewhat intrested
                    # And thus we add the keyword to the users preferred keywords.
                    for keyword in keywords.split(','):
                        k = keyword.lower()
                        if k in users[uid]:
                            users[uid][k] += 1
                        else:
                            users[uid][k] = 1

            except Exception as e:
                continue

    # Normalize keyword preferences
    for user_keywords in users.values():
        if len(user_keywords) > 0:
            m = max(user_keywords.values())
            for keyword, n in user_keywords.items():
                user_keywords[keyword] = round(1 + ((n / m) * 4))

    # Now do the opening
    with input_func(args.input, 'rt', encoding='utf8') as input_file:
        for line in input_file:
            obj = json.loads(line.strip())
            try:
                uid, iid = obj['userId'], obj['id']
                keywords = obj['keywords'] if 'keywords' in obj else 'None'
                keywords.split(',')
                for k in keywords:
                    k.lower()
                active_time = str(obj['activeTime']) if 'activeTime' in obj else '0'
                article = dict()
                article['time'] = active_time
                article['keywords'] = keywords

            except KeyError as e:
                # A field was missing for this JSON object, skip
                continue
            if not keywords == 'None':
                print('\t'.join([uid, iid, keywords]), file=f2)
            if not active_time == '0':
                rating = rate_article(users[uid], article)
                print('\t'.join([uid, iid, rating]), file=f1)
print('>>>Done!')

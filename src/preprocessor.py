import json
import os.path
import gzip
import argparse
from tools.rater import rate_article

parser = argparse.ArgumentParser(
    description='Preprocess the data set',
)
parser.add_argument(
    '--input', '-i',
    help='Data set file to process. Files ending with .gz will be decompressed '
         'only if the archive contains a single file.'
         'If directory then processor will iterate over the directories content. (Default: %(default)s)',
    default='test.data',
)
parser.add_argument(
    '--dataset1', '-1',
    type=argparse.FileType('wt', encoding='utf8'),
    help='File to write user, item and estimated rating to. (Default: %(default)s)',
    default='dataset1.txt',
)

parser.add_argument(
    '--dataset2', '-2',
    type=argparse.FileType('wt', encoding='utf8'),
    help='File to write user, item and keywords to. (Default: %(default)s)',
    default='dataset2.txt',
)
args = parser.parse_args()


def update_users_from_file(users_dict, f):
    with input_func(f, 'rt', encoding='utf8') as input_file:
        for line in input_file:
            obj = json.loads(line.strip())
            try:
                uid = obj['userId']
                if uid not in users_dict:
                    users_dict[uid] = dict()
                keywords = obj['keywords'] if 'keywords' in obj else 'None'
                if keywords is not 'None':
                    # If a user clicked on an article, we assume that means he is at least somewhat interested
                    # And thus we add the keyword to the users preferred keywords.
                    for keyword in keywords.split(','):
                        k = keyword.lower()
                        if k in users_dict[uid]:
                            users_dict[uid][k] += 1
                        else:
                            users_dict[uid][k] = 1

            except Exception:
                continue

    return users_dict


def normalize_keyword_preferences(users_dict):
    # Normalize keyword preferences
    for user_keywords in users_dict.values():
        if len(user_keywords) > 0:
            m = max(user_keywords.values())
            for keyword, n in user_keywords.items():
                user_keywords[keyword] = round(1 + ((n / m) * 4))
    return users_dict


def print_to_output(f, users_dict, o1, o2):
    with input_func(f, 'rt', encoding='utf8') as input_file:
        for line in input_file:
            obj = json.loads(line.strip())
            try:
                uid, iid = obj['userId'], obj['id']
                keywords = obj['keywords'] if 'keywords' in obj else 'None'
                keywords = keywords.split(',')
                for k in keywords:
                    k.lower()
                active_time = str(obj['activeTime']) if 'activeTime' in obj else '0'
                article = dict()
                article['time'] = active_time
                article['keywords'] = keywords

            except KeyError as e:
                # A field was missing for this JSON object, skip
                continue

            # TODO:
            # If No keywords AND no active_time -> Pass
            # If Only keywords -> Pass
            # If Only active_time -> rate by only active time.
            # If both -> rate by both.

            # if not keywords == 'None':
                 # print('\t'.join([uid, iid, keywords]), file=o2)
            if not active_time == '0':
                rating = str(rate_article(users_dict[uid], article))
                print('\t'.join([uid, iid, rating]), file=o1)


print('>>>Start reading file...')

# Should we decompress with Gzip?
if args.input.endswith('.gz'):
    input_func = gzip.open
else:
    input_func = open

print("Building Keyword preferences")
users = dict()
if os.path.isdir(args.input):
    for file in os.listdir(args.input):
        print("Building keywords from file:", file)
        users = update_users_from_file(users, os.path.join(args.input, file))
else:
    users = update_users_from_file(users, args.input)
users = normalize_keyword_preferences(users)

if os.path.isdir(args.input):
    for file in os.listdir(args.input):
        print("Printing results from file:", file)
        print_to_output(os.path.join(args.input, file), users, args.dataset1, args.dataset2)
else:
    print_to_output(args.input, users, args.dataset1, args.dataset2)

print('>>>Done!')

"""
Created on Thu Feb  8 14:46:23 2018

@author: zhanglemei
"""

import json
import os
import gzip
import argparse

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
    help='File to write user, item and active time to. (Default: %(default)s)',
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

    # Now do the opening
    with input_func(args.input, 'rt', encoding='utf8') as input_file:
        for line in input_file:
            obj = json.loads(line.strip())
            try:
                uid, iid = obj['userId'], obj['id']
                keywords = obj['keywords'] if 'keywords' in obj else 'None'
                active_time = str(obj['activeTime']) if 'activeTime' in obj else '0'
            except KeyError as e:
                # A field was missing for this JSON object, skip
                continue
            if not keywords == 'None':
                print('\t'.join([uid, iid, keywords]), file=f2)
            if not active_time == '0':
                print('\t'.join([uid, iid, active_time]), file=f1)
print('>>>Done!')

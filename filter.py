#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 14:46:23 2018

@author: zhanglemei
"""

import json
import os
from contextlib import nested

output_fname1 = 'dataset1.txt'
output_fname2 = 'dataset2.txt'

input_fname = 'test.data'
rootPath = os.path.abspath('.')
input_file = rootPath + os.sep + input_fname

print '>>>Start reading file...'
with nested(open(output_fname1, 'a'), open(output_fname2, 'a')) as (f1,f2):
    for line in open(input_file):
        obj = json.loads(line.strip())
        try:
            uid, iid = obj['userId'], obj['id']
            keywords = obj['keywords'] if 'keywords' in obj else 'None'
            active_time = str(obj['activeTime']) if 'activeTime' in obj else '0'
        except Exception as e:
            continue
        if not keywords=='None':
            print >>f2, '\t'.join([uid, iid, keywords]).encode('utf8')
        if not active_time=='0':
            print >>f1, '\t'.join([uid, iid, active_time]).encode('utf8')
print '>>>Done!'

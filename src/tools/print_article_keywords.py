import os
import json
output = open('article_keywords', 'wt', encoding='utf8')
used = set()
with open('one_week/20170101', 'rt', encoding='utf8') as input_file:
    for line in input_file:
        obj = json.loads(line.strip())
        try:
            uid, iid = obj['userId'], obj['id']
            keywords = obj['keywords'] if 'keywords' in obj else 'None'
            if iid not in used:
                used.add(iid)
                print(iid + "*-*" + keywords, file=output)
        except Exception:
            continue

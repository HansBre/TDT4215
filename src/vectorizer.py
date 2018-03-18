import json
import os.path
import gzip
import argparse
from tools.rater import rate_article
from sklearn.feature_extraction.text import TfidfVectorizer


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
    '--treshold','-t',
    help = 'Sets the treshold for views required by an article, defaults to 30',
    default=30,
)
args = parser.parse_args()

if args.input.endswith('.gz'):
    input_func = gzip.open
else:
    input_func = open

article_keywords = dict()
user_articles = dict()

def update_keywords_from_file(keywords_dict,user_dict,f):
    with input_func(f,'rt',encoding ='utf8') as input_file:
        for line in input_file:
            obj= json.loads(line.strip())
            try:
                title = obj['title'] if 'title' in obj else 'None'
                uid = obj['userId'] if 'userId' in obj else 'None'
                keywords = obj['keywords'] if 'keywords' in obj else 'None'
                if (title is not 'None') and (keywords is not 'None'):
                    if title not in article_keywords:
                        article_keywords[title] = dict()
                        article_keywords[title]['count'] = 0
                        article_keywords[title]['keywords'] = keywords.split(',')
                        user_dict[title] = []
                    article_keywords[title]['count']+=1
                    if uid!='None':
                        user_dict[title].append(uid)
            except Exception:
                continue


if os.path.isdir(args.input):
    for file in os.listdir(args.input):
        print("Building keywords from file:", file)
        update_keywords_from_file(article_keywords,user_articles, os.path.join(args.input, file))
else:
    update_keywords_from_file(article_keywords,user_articles, args.input)

print(len(article_keywords.keys()))



keys = list(article_keywords.keys())
for title in keys:
    if article_keywords[title]['count']<30:
        del article_keywords[title]
        del user_articles[title]
print(len(article_keywords.keys()))
print(len(user_articles.keys()))
article_corpus = []
for title in article_keywords:
    article_corpus.append(" ".join(article_keywords[title]['keywords']))
vectorizer = TfidfVectorizer()
vectorizer.fit_transform(article_corpus)
idf = vectorizer.idf_
print(dict(zip(vectorizer.get_feature_names(), idf)))
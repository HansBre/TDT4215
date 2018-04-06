import json
import os.path
import gzip
import threading
import argparse
from operator import itemgetter
from tools.rater import rate_article
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


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

parser.add_argument(
    '--level','-l',
    help='The level of which to measure recall/presicion, 10 will measure among top 10 documents retrieved etc, default 10',
    default=10
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
    if article_keywords[title]['count']<args.treshold:
        del article_keywords[title]
        del user_articles[title]
print(len(article_keywords.keys()))
print(len(user_articles.keys()))
print("Finished disk reading, time for them numbers")
article_corpus = []
article_tags = []
for title in article_keywords:
    article_corpus.append(" ".join(article_keywords[title]['keywords']))
    article_tags.append(title)
vectorizer = TfidfVectorizer(norm='l2',smooth_idf=False,min_df=1)
tf_idf_articles =  vectorizer.fit_transform(article_corpus)

user_keywords = dict()
for title in user_articles:
    for user in user_articles[title]:
        if user not in user_keywords:
            user_keywords[user] = ''
        user_keywords[user] = user_keywords[user] + ' '+ ' '.join(article_keywords[title]['keywords'])
user_corpus = []
user_tags = []
for user in user_keywords:
    user_corpus.append(user_keywords[user])
    user_tags.append(user)
tf_idf_users = vectorizer.fit_transform(user_corpus)
similarity_matrix = linear_kernel(tf_idf_articles,tf_idf_users)
def calculate_cosine_sim(uid,k_value):
        
    

    user_tag = user_tags[uid]
    ratings = []


    for entry in range(0,similarity_matrix.shape[0]):
        if similarity_matrix[entry][uid]!=0:
            ratings.append((article_tags[entry],similarity_matrix[entry][uid]))
    ratings = sorted(ratings,key=itemgetter(1),reverse=True)
    global_relevant = 0
    for title,score in ratings:
        if user_tag in user_articles[title]:
            global_relevant+=1
    global_relevant = float(global_relevant)
    n_retrieved = []
    max_rating = k_value
    for counter in range(0,max_rating):
        if not len(ratings)<=counter:
            n_retrieved.append(ratings[counter])


    local_relevant = 0
    for title,score in n_retrieved:
        if user_tag in user_articles[title]:
            local_relevant+=1
        
    presicion = local_relevant/max_rating
    return presicion
calculate_cosine_sim(1,10)
total_precision = 0
add_lock = threading.Lock()
tracer = 0
for counter in range(0,similarity_matrix.shape[1]):
    total_precision+=calculate_cosine_sim(counter,5)
    if counter>tracer:
        print("Finished "+repr(counter)+" vectors , "+repr(similarity_matrix.shape[1]-counter)+" remaining")
        tracer+=10000



class myThread (threading.Thread):
    def __init__(self, threadID,chunk_size):
      threading.Thread.__init__(self)
      self.threadID = threadID
      self.chunk_size = chunk_size
      self.offset = self.threadID*self.chunk_size
      self.range = self.offset+self.chunk_size
      self.local_precision = 0
    def run(self):
        tracer = 100
        for counter in range(self.offset,self.range):
            self.local_precision+=calculate_cosine_sim(counter,10)
            if((counter-self.offset)>tracer):
                print("Thread "+repr(self.threadID)+"finished "+repr(counter-self.offset)+" vectors "+repr(chunk_size-(counter-self.offset))+" to go")
                tracer+=100
        add_lock.acquire()
        global total_precision
        total_precision+=self.local_precision
        add_lock.release()


num_threads = 16
chunk_size = int(similarity_matrix.shape[1]/num_threads)
threads = []
"""
for counter in range(0,num_threads):
    thread = myThread(counter,chunk_size)
    threads.append(thread)
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
"""
print(total_precision)
print(total_precision/similarity_matrix.shape[1])

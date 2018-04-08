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

parser.add_argument(
    '--parallel','-p',
    help="Utilize parallelization for metric computation",
    default=False
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

recommendation_count = dict()
def calculate_cosine_sim(uid,k_value,r_count):
    
    user_tag = user_tags[uid]
    ratings = []
    for entry in range(0,similarity_matrix.shape[0]):
        if similarity_matrix[entry][uid]!=0:
            ratings.append((article_tags[entry],similarity_matrix[entry][uid]))
    ratings = sorted(ratings,key=itemgetter(1),reverse=True)
    global_relevant_titles = []
    global_relevant = 0
    for title,score in ratings:
        if user_tag in user_articles[title]:
            global_relevant+=1
            global_relevant_titles.append(title)
    global_relevant = float(global_relevant)
    n_retrieved = []
    max_rating = k_value

    for counter in range(0,max_rating):
        if not len(ratings)<=counter:
            n_retrieved.append(ratings[counter])


    local_relevant = 0
    local_irrelevant = 0
    true_p = dict()
    false_p = dict()
    local_relevant_titles = []
    for title,score in n_retrieved:
        if title in global_relevant_titles:
            local_relevant+=1
            local_relevant_titles.append(title)
        else:
            local_irrelevant+=1
        true_p[score] = float(local_relevant)/len(n_retrieved)
        false_p[score] = float(local_irrelevant)/len(n_retrieved)
        
        if title not in r_count:
            r_count[title] = 0
        r_count[title]+=1
    presicion = local_relevant/max_rating

total_precision = 0

tracer = 10000
total_ctr = 0

def calculate_total_roc(true_pr,false_pr,tot_roc,step_size):
    steps = len(tot_roc)
    rates = []
    for counter in range(0,steps):
        limit = counter*step_size
        local_true = 0
        local_false = 0
        false_rate = 0
        true_rate = 0
        for score in true_pr:
            if  limit<score:
                true_rate+=true_pr[score]
                local_true+=1
        for score in false_pr:
            if limit < score:
                false_rate+=false_pr[score]
                local_false+=1
        res_true = float(true_rate/local_true) if local_true!=0 else 0
        res_false = float(false_rate/local_false) if local_false!=0 else 0
        
        rates.append((res_true,res_false))
    if not len(tot_roc)==0:
        for i in range(0,len(tot_roc)):
            tot_roc[i] = (tot_roc[i][0]+rates[i][0],tot_roc[i][1]+rates[i][1])
        return tot_roc
    else:
        return rates
def calculate_aoc(roc):
    area_under = 0
    for i in range(0,len(roc)-1):
        (x0,y0) = roc[i]
        (x1,y1) = roc[i+1]
        dx = x0-x1 
        dy = y0-y1
        area_under+=(dx*dy)
        print('dx',dx,'dy',dy,'area under',area_under)
        x0 = x1
        y0 = y1
    area = roc[0][0]*roc[0][1]
    return area_under/area





        

                

    
    

if not args.parallel:
    total_roc = []
    for counter in range(0,10):
        total_roc.append((0,0))
    for counter in range(0,similarity_matrix.shape[1]):
        (local_precision,true_p_r,false_p_r) = calculate_cosine_sim(counter,5,recommendation_count)
        total_roc = calculate_total_roc(true_p_r,false_p_r,total_roc,0.1)
        total_precision+=local_precision

        if counter>tracer:
            print("Finished "+repr(counter)+" vectors , "+repr(similarity_matrix.shape[1]-counter)+" remaining")
            
            #tracer+=10000
    for title in recommendation_count:
        total_ctr+= float(len(user_articles[title]))/float(recommendation_count[title])
    average_ctr = total_ctr/similarity_matrix.shape[0]
    average_roc = []
   
    for (true_r,false_r) in total_roc:
        average_roc.append((float(true_r/similarity_matrix.shape[1]),float(false_r/similarity_matrix.shape[1])))
    print("ROC")
    for (true_r,false_r) in average_roc:
        print("True",true_r,"False",false_r)
    print("ROC",calculate_aoc(average_roc))
    
    print("Avg. Precision",total_precision/similarity_matrix.shape[1])
    print("CTR",average_ctr)

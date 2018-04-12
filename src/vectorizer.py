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
    '--threshold','-t',
    help = 'Sets the treshold for views required by an article in order for it not to be pruned, defaults to 30',
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
args.level = int(args.level)
args.threshold = int(args.threshold)


if args.input.endswith('.gz'):
    input_func = gzip.open
else:
    input_func = open


article_keywords = dict()
user_articles = dict()
#This function maps users to articles, maps keywords to articles and counts how many times a particular article appears in the collection.
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



#Prune articles with less views than specified by the threshold
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

#Extract keywords from articles and create a TF-IDF weighted sparse matrix
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

#Create user keyword vectors by examining which articles the user has read, and merging their keywords into a single vector.
#Create TF-IDF weighted sparse matrix
user_corpus = []
user_tags = []
for user in user_keywords:
    user_corpus.append(user_keywords[user])
    user_tags.append(user)
tf_idf_users = vectorizer.fit_transform(user_corpus)
#Calculate similarity between users and articles
similarity_matrix = linear_kernel(tf_idf_articles,tf_idf_users)

recommendation_count = dict()

#Calculate the ROC for a specific user's corresponding article ratings.
def calculate_roc(intervals,ratings,global_relevant_titles):
    roc = []
    for i in range(0,intervals):
        score_tres = i*0.1
        positives = len(global_relevant_titles)
        right_positives = 0
        right_negatives = 0
        negatives = len(ratings) - len(global_relevant_titles)
        for title,score in ratings:
            
            
            if title in global_relevant_titles:
                if score_tres<score:
                    right_positives+=1
            else:
                
                if score_tres<score:
                    right_negatives+=1
        res_true = float(right_positives)/float(positives) if positives!=0 else 0
        res_false = float(right_negatives)/float(negatives) if negatives!=0 else 0
        roc.append((res_true,res_false))
    return roc

#Sorts the ratings, returns top K and calculate metrics
def calculate_cosine_sim_metrics(uid,k_value,r_count):
    
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
    local_relevant_titles = []
    mrhr = 0
    rank = 1
    for title,score in n_retrieved:
        if title in global_relevant_titles:
            local_relevant+=1
            local_relevant_titles.append(title)
            mrhr = float(1/rank) if mrhr == 0 else mrhr
        else:
            local_irrelevant+=1
        if title not in r_count:
            r_count[title] = 0
        r_count[title]+=1
        rank+=1

    presicion = local_relevant/max_rating
    recall = local_relevant/global_relevant
    roc = calculate_roc(10,ratings,global_relevant_titles)
    return (recall,presicion,mrhr,roc)



tracer = 10000
total_ctr = 0

def sum_roc(new_roc,old_roc):
    if len(old_roc)!=0:
        for i in range(0,len(new_roc)):
            new_roc[i] = ((new_roc[i][0]+old_roc[i][0]),new_roc[i][1]+old_roc[i][1])
    return new_roc

def avg_roc(roc,n):
    avg_roc = []
    for entry in roc:
        avg_roc.append((entry[0]/n,entry[1]/n))
    return avg_roc

def calculate_aoc(roc):
    area_under = 0
    for i in range(0,len(roc)-1):
        (y0,x0) = roc[i]
        (y1,x1) = roc[i+1]
        dx = x0-x1 
        dy = y0-y1
        area_under+= ((dx*dy)/2)+dx*y0
        x0 = x1
        y0 = y1
    return area_under/1.0

def calculate_total_ctr(user_articles,recommendation_count):
    total_ctr = 0
    for title in recommendation_count:
        total_ctr+= float(len(user_articles[title]))/float(recommendation_count[title])
    return total_ctr



total_roc = []
for counter in range(0,10):
    total_roc.append((0,0))
total_precision = 0
total_recall = 0
total_mrhr = 0

#For every user row in the similarity matrix, sort ratings and calculate metrics
for counter in range(0,similarity_matrix.shape[1]):
    (local_recall,local_precision,local_mrhr,roc) = calculate_cosine_sim_metrics(counter,args.level,recommendation_count)
    total_precision+=local_precision
    total_roc = sum_roc(roc,total_roc)
    total_recall+=local_recall
    total_mrhr+=local_mrhr

    if counter>tracer:
        print("Finished "+repr(counter)+" vectors , "+repr(similarity_matrix.shape[1]-counter)+" remaining")
        
        #tracer+=10000




total_ctr = calculate_total_ctr(user_articles,recommendation_count)
average_ctr = total_ctr/similarity_matrix.shape[0]
average_roc = avg_roc(total_roc,similarity_matrix.shape[1])
average_precision = total_precision/similarity_matrix.shape[1]
average_recall = total_recall/similarity_matrix.shape[1]
average_mrhr = total_mrhr/similarity_matrix.shape[1]
averarage_f1 = (2*average_precision*average_recall)/(average_precision+average_recall)
print("ROC")
for (true_r,false_r) in average_roc:
    print("True",true_r,"False",false_r)
print("AOC",calculate_aoc(average_roc))
print('Avg. Recall',average_recall)
print('Avg. MRHR', average_mrhr)
print("Avg. Precision",average_precision)
print('Avg. f1', averarage_f1)
print('Avg. CTR', average_ctr)


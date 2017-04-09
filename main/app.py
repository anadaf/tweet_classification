"""
Core of Hot news from twitter feeds
------------------------------------------------------

@author: Ali Nadaf

[1] The core initially reads data and transforms it into a csv file.
 [2] Then, it trains the classifiers using BBC training data for topic_classifier and pre-trained w2v data for hot_classifier. 
 [3] After that, for every 10 minutes, it downloads and stores new tweets and [4] predicts their topics and (hot) classes 
 (whether it is hot or not). 
[5] The hot (best) tweet for each topic is identified. 
[6] It selects a sample of users  randomly.
[7] Users are notified with the hot tweets of the topic based on their interest topics.
 Note that, users do not receive more than 2 tweets per day.
 [8] The result of recommended (notified) tweets to the users is stored in a csv file. 
 
 Note: Based on this explanation, the core covers all the questions posted in 
 https://docs.google.com/document/d/1ziUlEDtOBChJzHvArc4GzQKJKG1s-Ut9IkzGAyzAdJI/edit#heading=h.wv9t71lmeyft
"""

import twitter
from tweet_data.collect_data import collect_data,get_tweets
from tweet_data.data_transformer import preprocessing_data
import pandas as pd
from classifiers.topic_classifier import topic_classifier
from tweet_data.test_utils import sample_user_generator,generate_dict
import os,csv
import random
from classifiers.hot_classifier import hot_classifier
import numpy as np
import time


#--------------------------------------------------------------------------------
# These tokens are needed for user authentication.
# Credentials can be generates via Twitter's Application Management:
#	https://apps.twitter.com/app/new
#--------------------------------------------------------------------------------
consumer_key = '8AIoVRM92Z0GiOI0ImZDEQzGz'
consumer_secret = 'LNn2koDSBDtFOdyhiJg3JmpJZBL06z11ARwsm1XoDY1qbspEYP'
access_token='850101633870475264-2CfPuTo3Et96RGENTwEs6xRxCCliLaB'
access_token_secret = 'GlWbsyxeM7dbfMQTJ1nc0hTF5LezR5wOuIob3muzcA2x2'
twitter_api = twitter.Api(consumer_key=consumer_key,
                              consumer_secret=consumer_secret,
                              access_token_key=access_token,
                              access_token_secret=access_token_secret)

#load tweeet data using twitter_api
def load_data(twitter_api):
    collect_data(twitter_api)
    preprocessing_data()
    pass

#training the classifiers
def training_clasifiers():
    db_news_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    db_news_dir = os.path.join(db_news_dir, 'classifiers', 'news_data')
    topics = ['business', 'entertainment', 'politics', 'sport', 'tech']
    # create a topic classifier object
    tc = topic_classifier(db_news_dir, topics,'mulNB')
    tc.training()


    #tweet data
    db_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    db_dir = os.path.join(db_dir, 'tweet_data', 'twitter_data', 'clean_tweets.csv')
    #pre-trained word2vec data
    w2v_db_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    w2v_db_dir = os.path.join(w2v_db_dir, 'tweet_data', 'w2v_trained_data', 'vectors.text')
    #create a hot tweets classifier object
    hc = hot_classifier(db_dir, w2v_db_dir, 'lsi')
    hc.data_normalizer()
    return tc,hc

def main():
    # [1] collecting tweet data
    print('loading tweets data ...',end="")
    load_data(twitter_api)
    print('Done!')

    #[2] training the classifiers
    print("loading the classifiers...",end="")
    tc,hc=training_clasifiers()
    print("Done!")

    # a path for the output
    output_rec_dir=os.path.join(os.path.dirname(__file__),'rec_output.csv')
    rec_user=[]
    list_user_id=[]
    rec_items=[]

    # generate sample users
    print('generating sample users ...',end="")
    sample_users = sample_user_generator()
    users_dict = generate_dict(sample_users)
    print('Done!')
    while True:
        #[3] collecting new tweets
        print('loading new tweets ...', end="")
        tweets = get_tweets(twitter_api)
        print('Done!')
        list_data=[]
        #[4] predicting topics and classes for new tweets
        for tweet in tweets:
            id=tweet['id']
            text = tweet['text']
            date=tweet['created_at']
            url=tweet['urls']
            tweet_counts = tc.method.transform([text])
            #predicting
            topic_prediction = tc.classifier.predict(tweet_counts)
            print([text],topic_prediction)
            queries_class, queries_hot_scores = hc.find_query_hot_class([text])
            print(queries_class, queries_hot_scores)
            list_data.append([id,text,date,topic_prediction[0],queries_class[0],queries_hot_scores[0],url])


        predict_output_DF = pd.DataFrame({'id':[list(x) for x in zip(*list_data)][0],
                                'tweet':[list(x) for x in zip(*list_data)][1],
                               'date':[list(x) for x in zip(*list_data)][2],
                               'topic':[list(x) for x in zip(*list_data)][3],
                                'news_class': [list(x) for x in zip(*list_data)][4],
                                'hot_score': [list(x) for x in zip(*list_data)][5],
                                'url': [list(x) for x in zip(*list_data)][6]})

        # [5] Determing top tweets in each topic
        top_news=predict_output_DF.groupby('topic').apply(lambda x: x.nlargest(1,'hot_score')).reset_index(drop=True)
        top_topics=top_news.topic.unique()
        for i in range(len(top_news)):
            print("Top News in ", top_news['topic'].ix[i],':',top_news['tweet'].ix[i])

        # [6] choosing a sample from users for notifying
        rand_int=random.randint(1,len(list(users_dict.keys())))
        rand_users=random.sample(list(users_dict.keys()), rand_int)

        # [7] notifying users with the hot tweets
        print('notifying users with the hot tweets ...', end="")
        for user in rand_users:
             if users_dict[user]['rec_date'] != time.strftime("%d/%m/%Y"):
                 users_dict[user]['n_rec'] =0
             if users_dict[user]['interest_topics'] in top_topics and users_dict[user]['n_rec']<2:
                 top_item=top_news[top_news['topic'] == users_dict[user]['interest_topics']]
                 rec_id=top_item['id']
                 if users_dict[user]['rec_tweet']!=rec_id.values[0]:
                     # print('item rec id ',rec_id.values[0],'for user',user)
                     users_dict[user]['rec_tweet']=rec_id.values[0]
                     users_dict[user]['rec_date'] = time.strftime("%d/%m/%Y")
                     users_dict[user]['n_rec']+=1
                     # print('number of recommend items to ', users_dict[user]['n_rec'], 'for user', user)
                     list_user_id.append(user)
                     rec_user.append(users_dict[user])
                     rec_items.append(top_item)
                 else:
                     continue

        print("Done!")
        # print('rec item to user 108: ',users_dict[108]['rec_tweet'])
        # print('number of rec to user 108: ',users_dict[108]['n_rec'])

        #[8] writing the results to disk
        print("Writing the results to disk...", end='')
        output_recommender=pd.DataFrame({'user ID':list_user_id,
                                         'rec_date':[rec_user[i]['rec_date'] for i in range(len(rec_user))],
                                         'tweet_id':[rec_items[i]['id'].values[0] for i in range(len(rec_items))],
                                         'tweet':[rec_items[i]['tweet'].values[0] for i in range(len(rec_items))],
                                         'tweet_date':[rec_items[i]['date'].values[0] for i in range(len(rec_items))],
                                         'topic':[rec_items[i]['topic'].values[0] for i in range(len(rec_items))],
                                         'tweet_class':[rec_items[i]['news_class'].values[0] for i in range(len(rec_items))],
                                         'tweet_score':[rec_items[i]['hot_score'].values[0] for i in range(len(rec_items))],
                                         'tweet_url':[rec_items[i]['url'].values[0] for i in range(len(rec_items))],
                                         'number rec per user': [rec_user[i]['n_rec'] for i in range(len(rec_user))]
                                         })
        output_recommender.to_csv(output_rec_dir)
        print("Done!")
        #waiting time = 10 minutes
        time.sleep(600)

if __name__ == "__main__":
     main()






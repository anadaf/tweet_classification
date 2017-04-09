"""
A tool to pull out the data from the tweets dumped into the disk and clean it and place the cleaned one into
another database
------------------------------------------------------

@author: Ali Nadaf
"""

import os

# return retweet_count and favorite_count
def get_cluster_labels(tweet, labels):
    return (tweet.get(labels[0], 0), tweet.get(labels[1], 0))

# return tweet, description and user info
def get_item_features(tweet, features):
    return (tweet.get(features[0], 'Missing tweet body'), tweet.get(features[1], 'Missing description'), tweet.get('user', 'Missing user id').get('screen_name', 'Missing user id'))

"""
a transformer to convert the json data into csv 

***parameter***
 - input_file_path: the file path of the dumped tweets in json  
 - output_file_path: the file path of the tweets in csv  
"""

def preprocessing(input_file_path, output_file_path):
    import csv, json
    output_file = open(output_file_path, 'w+', encoding='utf-8')
    writer = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    with open(input_file_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            screen_name = record.get('user', '').get('screen_name', '')
            location=record.get('user','').get('location','')
            time=record.get('user','').get('created_at','')
            user_id = record.get('user', '').get('id', 0)
            favorite_count = record.get('favorite_count', 0)
            retweet_count = record.get('retweet_count', 0)
            urls = record.get('urls', '')
            text = record.get('text', '')
            writer.writerow([screen_name,user_id,time, favorite_count, retweet_count, text, urls,location])
    output_file.close()

def preprocessing_data():
    cur_dir = os.path.dirname(__file__)
    data_path = os.path.join(cur_dir,'twitter_data', 'news_tweets.dat')
    output_path = os.path.join(cur_dir, 'twitter_data', 'clean_tweets.csv')
    preprocessing(data_path, output_path)
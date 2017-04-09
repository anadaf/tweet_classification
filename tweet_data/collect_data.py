"""
core for collecting tweets using twitter api tools
------------------------------------------------------

@author: Ali Nadaf
"""


import os
from tweet_data.data_mining import NewsFeed
import datetime
"""
create a dictionary for all news agency twitter links containing their twitter links
The list of twitter news agencies are:
New York Times, The Sun, The Times, The Associated Press, CNN, BBC NEWS, CNET, MSN UK, Telegraph,
USAToday, Wall Street Journal, Washington Post, Boston Globe, NEWS.com.au, Sky News, SFGate, Al-Jazeera,
Independent, UK, Guardian.co.uk, LA Times, Reuters, ABC News, Bloomberg, Business Week, Time

**parameters**
- input:
 ``file_path``: the file path of tweet_list.dat containing list of news agencies
- output:
 user_set: a dictionary with name of authenticating news agencies and their twitter url 
"""
def load_news_agencies(file_path):
    user_set = {}
    with open(file_path, 'r') as f:
        counter = 0
        user_name = None
        for line in f:
            line = line.strip()
            if counter % 2 == 0:
                user_name = line
            else:
                user_id = line
                user_set[user_name] = user_id.replace('https://twitter.com/', '')
            counter += 1
    # print(user_set)
    return user_set

#get the most recent tweets posted from the authenticating user
def get_tweets(api):
    cur_dir = os.path.dirname(__file__)
    user_set_path = os.path.join(cur_dir, 'tweet_list.dat')
    user_ids = load_news_agencies(user_set_path)
    username=user_ids
    # authenticating users
    for user in username:
        tweets = api.GetUserTimeline(36511031)
        buffer=[]
        for tweet in tweets:
            tweet = tweet.AsDict()
            # print( tweet['created_at'])
            tweet_time = datetime.datetime.strptime(tweet['created_at'],'%a %b %d %H:%M:%S +0000 %Y')
            minute_lapse = (datetime.datetime.now() - tweet_time).total_seconds() / 60
            # print all tweets posted during 10-minute waiting period
            if minute_lapse < 10:
                # print(tweet['text'].encode("utf-8"))
                buffer.append(tweet)
    return buffer

# Scraping and mining data from twitter via the Python Twitter Tools module
def collect_data(twitter_api):
    cur_dir = os.path.dirname(__file__)
    user_set_path = os.path.join(cur_dir,'tweet_list.dat')
    user_ids = load_news_agencies(user_set_path)
    dump_path = os.path.join(cur_dir,'twitter_data', 'news_tweets.dat')
    scraper = NewsFeed(twitter_api=twitter_api)
    scraper.mining_data(user_ids, dump_path)




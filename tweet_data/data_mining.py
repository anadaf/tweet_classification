"""
a util for mining the tweets and dumping them into the dis
------------------------------------------------------

@author: Ali Nadaf
"""


import json, twitter

class NewsFeed(object):
    def __init__(self, twitter_api):
        self.twitter_api = twitter_api
        pass

    # mining tweets from the twitter
    def mining_data(self, user_map, dump_path):
        file_stream = open(dump_path, 'w+')
        counter = 0
        print("list of users to collect", user_map)
        for user_name in user_map:
            screen_name = user_map[user_name]
            user_not_exist = False
            print('Scripting user:', user_name, 'screen name:', screen_name)
            for i in range(0, 16):  ## iterate through 16 times to get max No. of tweets
                try:
                    user_timeline = self.twitter_api.GetUserTimeline(screen_name=screen_name, count=200)
                except twitter.error.TwitterError:
                    user_not_exist = True
                if user_not_exist:
                    print(screen_name,"doesn't exist")
                    break
                for tweet in user_timeline:
                    counter += 1
                    # dump the data into disk
                    tweet_str = json.dumps(tweet.AsDict())
                    file_stream.write(tweet_str+'\n')
                    # print(dir(tweet))
                    if counter % 1000 == 0:
                        print(counter, 'tweets collected')
        file_stream.close()
        pass






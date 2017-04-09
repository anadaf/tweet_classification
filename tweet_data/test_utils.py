
"""

sample users generator
@author: Ali Nadaf
"""

import numpy as np
import scipy.sparse as sp
import time
import random
import pandas as pd

# Sample data containing user ids, their interest news topics and the recommendation data
def sample_user_generator():
    users=[ x for x in range(100,120)]
    topics = ['business', 'entertainment', 'politics', 'sport', 'tech']
    interest_topics=[random.choice(topics) for i in range(20)]
    dates=[time.strftime("%d/%m/%Y") for i in range(20)]
    return [users,interest_topics,dates]

"""
generate a dictionary for all users in the database with following keys  
 * user_id: an id for connecting the system to the respective users; 
 * interest_topics: a news category that user wish to get notification about;
 * rec_date : date of the sent notification from the system to the user;
 * n_rec : number of notifications received by user in one day;
"""

def generate_dict(users_data):
    user_dicts={}
    users=users_data[0]
    interest_topics=users_data[1]
    dates=users_data[2]
    for i in range(len(users)):
        user=users[i]
        user_dicts[user]={}
        user_dicts[user]['interest_topics']=interest_topics[i]
        user_dicts[user]['rec_tweet']=''
        user_dicts[user]['rec_date']=dates[i]
        user_dicts[user]['n_rec']=0
    return user_dicts



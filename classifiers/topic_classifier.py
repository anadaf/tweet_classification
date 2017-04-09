"""
Topic classifier

------------------------------------------------------

@author: Ali Nadaf

------------------------------------------------------
The classifier takes the tweets and place them into one of k topic classes.The classifier uses the database consisting 
of 2225 documents from the BBC news website corresponding to stories in five topical areas from 2004-2005.
Natural Classes: 5 (business, entertainment, politics, sport, tech)

 In this project, the classifier uses a number of scikit learn multi-class classifiers such as 
 - OneVsRestClassifier: One-vs-the-rest (OvR) multiclass/multilabel strategy (http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html)
 - SVC: C-Support Vector Classification. (http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
 - DecisionTreeClassifier: A decision tree classifier. (http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
 - RandomForestClassifier: A random forest classifier. (http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
 - NearestCentroid: Nearest centroid classifier. (http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html)
 - MLPClassifier: Multi-layer Perceptron classifier. (http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
 
 Note: The classifier can utilize the word2vec approach described in hot_classifier class. 
 
Reference:
- D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006.

"""


import os
import glob
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from textblob.classifiers import NaiveBayesClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.nearest_centroid import NearestCentroid

class topic_classifier():
    """
    ***parameter***
    - file_dir: File path of BBC training data
    - topics: tweet topic from one of these topics: 'business','entertainment','politics','sport' and 'tech'
    - classifier: Machine learning multi-class classifier from one of the following classifiers
         +'mulNB': Naive Bayes 
         +'svc': SVC
         +'dec_tree': Decision Tree
         +'rand_forest': Random Forest
         +'random_sample': Random Sample
         +'nearest_cent': Nearest centroid
         +'mlp': Multi-layer Perceptron
    """
    def __init__(self,file_dir,topics,classifier):
        self.file_dir=file_dir
        self.topics=topics
        self.classifier=None
        self.algorithm=classifier
        self.method=None



    #extracting training texts from different folders and dumping them into a dataframe
    def train_topics_gen(self):
        content=[]
        classes=[]
        for topic in self.topics:
            user_set_path = os.path.join(self.file_dir,topic)
            os.chdir(user_set_path)
            files=glob.glob("*.txt")
            for file in files:
                with open(file) as f:
                    content.append(f.read())
                    classes.append(topic)
        DF = pd.DataFrame({'class': classes,'content': content})
        return DF


    # training the classifier using the BBC training data
    def training(self):
        if self.algorithm=='mulNB':
            self.classifier = MultinomialNB()
        elif self.algorithm=='svc':
            self.classifier=OneVsRestClassifier(SVC())
        elif self.algorithm=='dec_tree':
            self.classifier=DecisionTreeClassifier()
        elif self.algorithm=='rand_forest':
            self.classifier=RandomForestClassifier()
        elif self.algorithm=='random_sample':
            self.classifier=RandomForestClassifier()
        elif self.algorithm=='nearest_cent':
            self.classifier=NearestCentroid()
        elif self.algorithm=='mlp':
            self.classifier=MLPClassifier()

        # BBC training dataset
        df=self.train_topics_gen()
        # vectorizing the contents of the data
        self.method = CountVectorizer()
        counts = self.method.fit_transform(df['content'].values)
        targets = df['class'].values
        self.classifier.fit(counts, targets)
        return self




if __name__ == "__main__":
    #read the training data
    cur_dir = os.path.dirname(__file__)
    cur_dir=os.path.join(cur_dir,'news_data')
    topics=['business','entertainment','politics','sport','tech']
    topic_model=topic_classifier(cur_dir,topics,'mulNB')
    topic_model.training()

    #tweets data
    db_dir=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
    db_dir=os.path.join(db_dir, 'tweet_data', 'twitter_data', 'clean_tweets.csv')
    db = pd.read_csv(db_dir, skiprows=1,
                     names=['screen_name', 'user_id', 'time', 'favorite_count', 'retweet_count', 'text', 'urls',
                            'location'])

    # predict the topic classs for each tweet in tweets data
    tweet_counts = topic_model.method.transform(db.text)
    predictions = topic_model.classifier.predict(tweet_counts)
    db['topic']=predictions


    # an example which takes a query (a tweet) and predict the topic class associated with it
    tweet='Bombers strike Syrian town hit in chemical attack https://t.co/RZRaeSjpuh https://t.co/J10FJKnUcQ'
    query_counts=topic_model.method.transform([tweet])
    query_pred = topic_model.classifier.predict(query_counts)
    print([tweet],query_pred)
    predictions


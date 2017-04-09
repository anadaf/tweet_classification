"""
A classifier to identify hot, urgent and worthy tweets.

------------------------------------------------------

@author: Ali Nadaf

------------------------------------------------------
The classifier takes tweets and identifies hot ones. For this purpose, the classifier determines
 the importance of tweets based on their content and puts them into different classes. The classes are 'cold', 'normal',
 'warm', 'hot' and 'breaking'. The relevant class to a tweet is identified based on the number of favorites and retweets 
 of the tweet. 
 
 ***Algorithm***
 For each authenticating news agency, the classifier initially normalizes the number of favorites and retweets.
  Then, a linear combination of these two normalized values is determined as
   importance_score=\alpha*\norm(number_favorite)+(1-\alpha)*\norm(number of retweets)
   and stored in importance_score. This score is utilized as a labelled data showing the importance of the
   tweets. Following thresholds are used to classify the training tweets:
   
   0.0 < importance_score < 0.1  --------> 'cold'
   0.1 < importance_score < 0.4  --------> 'normal'
   0.4 < importance_score < 0.6  --------> 'warm'
   0.6 < importance_score < 0.8  --------> 'hot'
   0.8 < importance_score < 1.0  --------> 'breaking'
   
   For the dumped tweets, the class and the corresponding score to each tweet are determined. These tweets are used as
    labelled tweets. For each query (new tweet), the classifier calculates the distance between the query and the 
    other tweets using following methods:
    - word2vec: Word2vec model. 
    Each tweet contains number of words. In this model, the classifier uses a pre-trained model (vector.text - 100MB) to produce 
    word embedding vector for the words. By averaging these vectors, an embedding vector representation for 
    the tweet is obtained.  For this purpose, classifier uses gensim (https://radimrehurek.com/gensim/) for word embedding.
    After determining tweet embedding vectors, different classifiers are used as follows:
                ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process", "Decision Tree", 
                "Random Forest", "Neural Net", "AdaBoost","Naive Bayes", "QDA"]
    - tfidf: TF-IDF model (more info: https://radimrehurek.com/gensim/models/tfidfmodel.html)
    - lsi: Latent Semantic Indexing (more info: https://radimrehurek.com/gensim/models/lsimodel.html)
    - rp: Random Projections (more info: https://radimrehurek.com/gensim/models/rpmodel.html)
    - hdp: Hierarchical Dirichlet Process (more info: https://radimrehurek.com/gensim/models/hdpmodel.html)
    - lda: Latent Dirichlet Allocation (more info: https://radimrehurek.com/gensim/models/ldamodel.html)
    - lem: LogEntropy model (more info: https://radimrehurek.com/gensim/models/logentropy_model.html)

Reference:
- Radim Řehůřek and Petr Sojka (2010). Software framework for topic modelling with large corpora. Proc. LREC Workshop on New Challenges for NLP Frameworks
Jump up 
"""



from sklearn import svm
from sklearn import preprocessing
from tweet_data.cbf import *
import gensim
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import warnings
import os
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

# Defining tweet class based on the importance_score (x)
def news_class(x):
    if x<0.1:
        return 'cold'
    elif x<0.4:
        return 'normal'
    elif x<0.6:
        return 'warm'
    elif x<0.8:
        return 'hot'
    else:
        return 'breaking'

#loading in a format compatible with the original word2vec implementation
def load_w2v_model(model_path):
    w2v_mod = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False)
    return w2v_mod

class hot_classifier():
    """
    ***parameters***
     - data_dir: file path of tweet data
     - w2v_trained_data_path: file path of pre-trained word2vec model
     - algorithm: algorithm to determine distance between the tweets. The list of algorithms is as follows:
        * 'tfidf': TF-IDF model (default)
        * 'lsi': Latent Semantic Indexing 
        * 'rp': Random Projections
        * 'hdp': Hierarchical Dirichlet Process
        * 'lda': Latent Dirichlet Allocation
        * 'lem': LogEntropy model 
        
    """
    def __init__(self,data_dir,w2v_trained_data_path,algorithm='tfidf'):
        self.data_dir=data_dir
        self.method=algorithm
        self.w2v_model=load_w2v_model(w2v_trained_data_path)
        self.data=None

    """
    Determines linear combination of normalized #favorite and #retweet
    ***parameter***
     - alpha: a weight between 0 and 1
    """
    def hybrid(self,alpha):
        if 0<= alpha<=1:
            return self.data.favorite_count.values*alpha+self.data.retweet_count.values*(1-alpha)
        else:
            raise ValueError("The parameter alpha should be between 0 and 1.")

    #load and clean the tweet data
    def load_data(self):
        data = pd.read_csv(self.data_dir, names=['screen_name', 'user_id', 'time', 'favorite_count', 'retweet_count', 'text', 'urls',
                                        'location'])
        data = data.drop_duplicates()
        self.data = data.reset_index()
        return self

    #normalize favorite and retweet
    def data_normalizer(self):
        self.load_data()
        data=self.data
        screen_name_list = data['screen_name'].unique()
        for names in screen_name_list:
            for column in ['favorite_count', 'retweet_count']:
                mx = data.loc[data['screen_name'] == names][column].max()
                normalized_values = data.loc[data['screen_name'] == names][column] / mx.astype(np.float64)
                data.loc[data['screen_name'] == names, column] = normalized_values
        data['hot_score'] = self.hybrid(alpha=0.5)
        data['class'] = data['hot_score'].map(lambda x: news_class(x))
        corpus= data['text'].unique()
        self.data=data
        self.corpus=corpus
        return self

    """
     predict the class of queries 
     ***parameter***
      input:
      - queries : list of queries
      output:
      - queries_class:  the predicted class for the queries
      - queries_hot_scores: the corresponding prediction scores 
    """
    def find_query_hot_class(self,queries):
        index, sim = query_similarity(queries, self.corpus, self.method)
        queries_ng = []
        queries_hot_scores = []
        queries_class = []
        for i in range(len(index)):
            queries_ng.append(self.data['hot_score'].ix[index[i]].tolist())
            queries_hot_scores.append(np.dot(queries_ng[i], sim[i]))
            queries_class.append(news_class(queries_hot_scores[i]))
        return queries_class,queries_hot_scores

    # Determine a vector representation for tweets using word2vec
    def w2v_doc_vec(self):
        w2v_corpus=build_w2v_corpusdic(self.corpus)
        doc_vec=[]
        mask=[]
        for i in range(len(w2v_corpus)):
            word_vec = []
            for j in range(len(w2v_corpus[i])):
                try:
                    word_vec.append(self.w2v_model.word_vec(w2v_corpus[i][j]))
                except:
                    continue
            if len(word_vec)!=0:
                doc_vec.append(np.mean(word_vec,0).tolist())
                mask.append(i)
        return doc_vec,mask

    # Determine a vector representation for queries using word2vec
    def w2v_query_vec(self,query):
        w2v_corpus=build_w2v_corpusdic(query)
        doc_vec=[]
        mask=[]
        for i in range(len(w2v_corpus)):
            word_vec = []
            for j in range(len(w2v_corpus[i])):
                try:
                    word_vec.append(self.w2v_model.word_vec(w2v_corpus[i][j]))
                except:
                    continue
            if len(word_vec)!=0:
                doc_vec.append(np.mean(word_vec,0).tolist())
                mask.append(i)
        return doc_vec,mask

    """
    predict the class of queries using word2vec method
    ***parameters***
    inputs:
    - query: list of queries
    - names: classifier name from 
     ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process","Decision Tree", "Random Forest", "Neural Net",
      "AdaBoost", "Naive Bayes", "QDA"]
    - classifiers: list of classifiers correspond to the classifier name 
      [ KNeighborsClassifier(10), SVC(kernel="linear", C=0.025), SVC(gamma=2, C=1),
      GaussianProcessClassifier(1.0 * RBF(2.0),warm_start=True), DecisionTreeClassifier(max_depth=20),
       RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), MLPClassifier(alpha=1), AdaBoostClassifier(),
       GaussianNB(),  QuadraticDiscriminantAnalysis()]
    outputs:
    - classifier_name: name of classifier
    - classifier_pred: predicted class for the query
    - classifier_score: corresponding score to the predicted class
    """
    def predict(self,query,names,classifiers):
        classifier_name=[]
        classifier_pred=[]
        classifier_score=[]
        doc_vec, mask = self.w2v_doc_vec()
        doc_class = self.data.ix[mask]['class'].tolist()
        keys = ('cold', 'normal', 'warm','hot','breaking')
        values = (0.1,0.3,0.5,0.7,0.9)
        cat_dict = dict(zip(keys, values))
        for name, clf in zip(names, classifiers):
            clf.fit(np.array(doc_vec), doc_class)
            queries_vector,mask=self.w2v_query_vec(query)
            pred=clf.predict(queries_vector)
            score=[cat_dict[pred[i]]  for i in range(len(pred))]
            classifier_name.append(name)
            classifier_pred.append(pred)
            classifier_score.append(score)

        return classifier_name,classifier_pred,classifier_score


if __name__ == "__main__":
    #tweet data
    cur_dir = os.path.dirname(__file__)
    db_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    db_dir = os.path.join(db_dir, 'tweet_data', 'twitter_data', 'clean_tweets.csv')
    #pre-trained word2vec model
    w2v_db_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    w2v_db_dir = os.path.join(w2v_db_dir, 'tweet_data', 'w2v_trained_data', 'vectors.text')
    #creating the classifier object
    hc=hot_classifier(db_dir,w2v_db_dir,'lsi')
    hc.data_normalizer()

    # an example for predicting hot classes of queries
    tweet1='Canada has been chosen as the best country'
    tweet2='Trump evolution on Syria did not happen overnight'
    queries_class, queries_hot_scores=hc.find_query_hot_class([tweet1,tweet2])

    # SVC prediction model
    # clf = svm.SVC()
    # # le = preprocessing.LabelEncoder()
    # # doc_transformed_class=le.fit_transform(doc_class)
    # doc_vec,mask=w2v_doc_vec(corpus)
    # doc_class=db.ix[mask]['class'].tolist()
    # clf.fit(np.array(doc_vec), doc_class)
    # example_doc_vec,mask= w2v_doc_vec([tweet1,tweet2])
    # svc_pred=clf.predict(example_doc_vec)
    # print(svc_pred)

    names = ["Nearest Neighbors"]
    classifiers = [KNeighborsClassifier(10)]
    classifier_name,pred_class,pred_score=hc.predict([tweet1,tweet2],names, classifiers)
    print(classifier_name,pred_class,pred_score)
    print('Done!')





# tweet_classifier
This set of python scripts is a real-time engine which loads news feeds from some well-known news agencies twitter channels and identifies if they are imporant or not. The important news is pushed to users. The following tasks are performed by the engine:
1. Collects, reads tweets data using twitter_api and transforms it into a csv file.
2. Trains the topic classifier using external training data to predict new news feed topic. The topics are technology, politic, sport, entertainment and business (supervised learning).
3. Trains the hot classifier using a pre-trained word2vec data to distinguish the hot news from the other news (unsupervised learning).
4. For every 10 minutes, the engine downloads new news feeds and groups them into similar topics and identifies if they are important or not. 
5. Identifies best tweet in each topic.
6. Pushes at most two real-time tweets to users each day.
7. Stores the notified tweets. 

# Dependencies
- Gensim
- Twitter
- Numpy
- Scipy
- Pandas
- Sklearn
- nltk
# How to use it
- Set project root to project folder directory.
- Simply run script `main\app.py` to run the engine.
# classification algorithms
## Topic classifier (supervised learning)
The classifier takes the tweets and place them into one of k topic classes.The classifier uses the database consisting 
of 2225 documents from the BBC news website corresponding to stories in five topical areas from 2004-2005.
- Natural Classes: 5 (business, entertainment, politics, sport, tech)

 In this project, the classifier uses a number of scikit learn multi-class classifiers such as 
 - OneVsRestClassifier: One-vs-the-rest (OvR) multiclass/multilabel strategy (http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html)
 - SVC: C-Support Vector Classification. (http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
 - DecisionTreeClassifier: A decision tree classifier. (http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
 - RandomForestClassifier: A random forest classifier. (http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
 - NearestCentroid: Nearest centroid classifier. (http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html)
 - MLPClassifier: Multi-layer Perceptron classifier. (http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
 
## Hot classifier (unsupervised learning)
The classifier takes tweets and identifies hot ones. For this purpose, the classifier determines
 the importance of tweets based on their content and puts them into different classes. The classes are **'cold', 'normal',
 'warm', 'hot'** and **'breaking'**. The relevant class to a tweet is identified based on the number of favorites and retweets 
 of the tweet. 
 
 For each authenticating news agency, the classifier initially normalizes the number of favorites and retweets.
  Then, a linear combination of these two normalized values is determined as
  ```
   importance_score=\alpha*\norm(number_favorite)+(1-\alpha)*\norm(number of retweets)
   ```
   and stored in importance_score. This score is utilized as a labelled data showing the importance of the
   tweets. Following thresholds are used to classify the training tweets:
   ```
  - 0.0 < importance_score < 0.1  --------> 'cold'
  - 0.1 < importance_score < 0.4  --------> 'normal'
  - 0.4 < importance_score < 0.6  --------> 'warm'
  - 0.6 < importance_score < 0.8  --------> 'hot'
  - 0.8 < importance_score < 1.0  --------> 'breaking'
   ```
   For the dumped tweets, the class and the corresponding score to each tweet are determined. These tweets are used as
    labelled tweets. For each query (new tweet), the classifier calculates the distance between the query and the 
    other tweets using following methods:
1. word2vec: Word2vec model. 
    Each tweet contains number of words. In this model, the classifier uses a pre-trained model (vector.text - 100MB) to produce 
    word embedding vector for the words. By averaging these vectors, an embedding vector representation for 
    the tweet is obtained.  For this purpose, classifier uses gensim (https://radimrehurek.com/gensim/) for word embedding.
    After determining tweet embedding vectors, different classifiers are used as follows:
    ```
                ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process", "Decision Tree", 
                "Random Forest", "Neural Net", "AdaBoost","Naive Bayes", "QDA"]
    ```            
2. tfidf: TF-IDF model (more info: https://radimrehurek.com/gensim/models/tfidfmodel.html)
3. lsi: Latent Semantic Indexing (more info: https://radimrehurek.com/gensim/models/lsimodel.html)
4. rp: Random Projections (more info: https://radimrehurek.com/gensim/models/rpmodel.html)
5. dp: Hierarchical Dirichlet Process (more info: https://radimrehurek.com/gensim/models/hdpmodel.html)
6. lda: Latent Dirichlet Allocation (more info: https://radimrehurek.com/gensim/models/ldamodel.html)
7. lem: LogEntropy model (more info: https://radimrehurek.com/gensim/models/logentropy_model.html)


# References
1. D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006.
2. Radim Řehůřek and Petr Sojka (2010). Software framework for topic modelling with large corpora. Proc. LREC Workshop on New Challenges for NLP Frameworks

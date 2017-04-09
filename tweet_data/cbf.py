# -*- coding: utf-8 -*-
"""
Core computations and tools for Content Based System
------------------------------------------------------

@author: Ali Nadaf
"""

from textblob import TextBlob 
import re
from tweet_data.nlp_utils import stemmer, tokenizer,stopwords
import numpy as np
from collections import defaultdict
from gensim import corpora,models,similarities


#Correct spelling 
def spell_corrector(sentence):
    tb = TextBlob(sentence)
    tb = tb.correct()
    return tb.string
   

upper_case_normalizer = {}

#returns the normalized texts
def normalize(s, spell_check=False):
    
    for (k,v) in upper_case_normalizer.items():
        s = s.replace(k,v)
    
    s = s.lower()
    
    if spell_check:
        s = spell_corrector(s)
    
    s = re.sub(r'(?u)([a-z])-([a-z])',r'\1\2',s)
    
    return s
 
#breaking the texts into words, phrases, symbols and other meaningful defined characters  
def stem(s):
    global stopwords
    s = tokenizer.tokenize(s)
    s = [stemmer.stem(w) for w in s if w not in stopwords]
    return ' '.join(s)

def query_similarity(queries,corpus,method='tfidf',n_neighbors=2):
    dictionary, corpusdic = build_corpusdic(corpus)
    if method == 'lsi':
        mdl = models.LsiModel(corpusdic, id2word=dictionary, num_topics=100)
    elif method == 'tfidf':
        mdl = models.TfidfModel(corpusdic)
    elif method == 'rp':
        mdl = models.RpModel(corpusdic, num_topics=100)
    elif method == 'hdp':
        mdl = models.HdpModel(corpusdic, id2word=dictionary)
    elif method == 'lda':
        mdl = models.LdaModel(corpusdic, id2word=dictionary, num_topics=100)
    elif method == 'lem':
        mdl = models.LogEntropyModel(corpusdic)
    elif method == 'norm':
        mdl = models.NormModel(corpus, norm='l2')

    else:
        raise ValueError("There is an invalid model method in the input!")
    index = similarities.MatrixSimilarity(mdl[corpusdic])
    indx_list=[]
    sim_list=[]
    for query in queries:
        vec_bow = dictionary.doc2bow(query.lower().split())
        vec_lsi = mdl[vec_bow]  # convert the query to LSI space
        sims = index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        sims = sims[:n_neighbors]
        indx_, sim_ = np.array(sims).transpose()
        indx_list.append(indx_)
        sim_list.append(sim_)
    return indx_list, sim_list

# Returns similar document scores and indices.
def find_similar_docs(doc_matrix, query_matrix, n_neighboors=30,
                       method='nearest', return_similarity=True):
                                
                        if n_neighboors > doc_matrix.shape[0]:
                                        raise ValueError("Not enough docs to find neighboors")
                        
                      
                        # Allocating from high to low on similarity of the docs to query docs
                        # thus the `-` sign     
                        top_n_index = np.argpartition(-doc_matrix, \
                                    n_neighboors-1, axis=1)
                        top_n_similar_docs = top_n_index[:,:n_neighboors]
                        
                        if return_similarity:
                            top_n_similar_docs_score = np.zeros(top_n_similar_docs.shape)
                            for (i_, indx_) in enumerate(top_n_similar_docs):
                                top_n_similar_docs_score[i_,:] = doc_matrix[i_,indx_]

                        return (top_n_similar_docs, top_n_similar_docs_score)
"""
    ''corpus'': a list of unique item description and contents 
    
    ''method'': different models which works on training corpus to predict unseen 
                and new upcoming documents. The methods can be one of the following items
        
                - tfidf: TF-IDF model (more info: https://radimrehurek.com/gensim/models/tfidfmodel.html)
                - lsi: Latent Semantic Indexing (more info: https://radimrehurek.com/gensim/models/lsimodel.html)
                - rp: Random Projections (more info: https://radimrehurek.com/gensim/models/rpmodel.html)
                - hdp: Hierarchical Dirichlet Process (more info: https://radimrehurek.com/gensim/models/hdpmodel.html)
                - lda: Latent Dirichlet Allocation (more info: https://radimrehurek.com/gensim/models/ldamodel.html)
                - lem: LogEntropy model (more info: https://radimrehurek.com/gensim/models/logentropy_model.html)
        
   ''n_neighboors'': number of similar items to an item
   
   ''return_similarity'': If True, returns the similarity scores
   ''batch_size'': Batch processor wrapper to increase the computation speed
   (Warning: this value can be optimized for different data sizes)
   

"""   
def build_doc_similarity_table(corpus,method='tfidf', n_neighboors=3,
                            return_similarity=True, batch_size=5000,
                            doc_dtype=np.int64, score_dtype=np.float16):
    """
    Batch processor wrapper for ``find_similar_docs`` to
    find `n_neighboors` similar docs to all the docs
    *Note 1*: increasing ``batch_size`` can increase memory usage, but can be
    faster
    """
    
    # This structure should be paralleized to doc multiple CPUs
    
    dictionary, corpusdic=build_corpusdic(corpus)
    
    if method=='lsi':
        lsi = models.LsiModel(corpusdic, id2word=dictionary, num_topics=100)
    elif method=='tfidf':
        lsi=models.TfidfModel(corpusdic)
    elif method=='rp':
        lsi=models.RpModel(corpusdic, num_topics=100)
    elif method=='hdp':
        lsi=models.HdpModel(corpusdic, id2word=dictionary)
    elif method=='lda':
        lsi=models.LdaModel(corpusdic, id2word=dictionary, num_topics=100)
    elif method=='lem':
        lsi=models.LogEntropyModel(corpusdic)
    elif method=='norm':
        lsi=models.NormModel(corpus,norm='l2')
      
    else:
        raise ValueError("There is an invalid model method in the input!")

    #Determing the similarities between different documents
    index = similarities.MatrixSimilarity(lsi[corpusdic])
    vec_lsi = lsi[corpusdic]
    doc_matrix = index[vec_lsi]  
        
    doc_count = doc_matrix.shape[0]

    similarity_table = np.zeros((doc_count, n_neighboors)).astype(doc_dtype)
    
    similarity_score = None
    
    if return_similarity:
        similarity_score = np.zeros((doc_count, n_neighboors)).astype(score_dtype)
       
    start_ = 0
    while (start_ < doc_count):
        
        end_ = start_ + batch_size
        
        if (end_ > doc_count):
            end_ = doc_count
            
        query_index = np.arange(start_,end_)
        query_matrix = doc_matrix[query_index]
        
        (similarity_table[query_index], similarities_) = \
            find_similar_docs(doc_matrix, query_matrix, 
                               n_neighboors=n_neighboors,
                               method='nearest', return_similarity=return_similarity)
        
        if return_similarity:
            similarity_score[query_index] = similarities_

        start_ = start_ + batch_size
        
    return (similarity_table, similarity_score)


"""
The function initially tokenizes the documents and then convert 
tokenized documents to vectors. The result is a sparse vector.

For instance: 
        The document "Interesting book" returns the sparse vector [(0, 1), (1, 1)],
        where the words 'Interesting' (id 0) and 'book' (id 1) appear once.
"""
def build_corpusdic(corpus):
    stemmed_corpus = list()
    
#list of normalized sentences (remove the common phrases)
    for sentence_ in corpus:
        stemmed_corpus.append(stem(normalize(str(sentence_))))

#tokenize the documents
    texts = [[word for word in document.lower().split()]
             for document in stemmed_corpus]
                 
    frequency = defaultdict(int)
    
    for text in texts:
        for token in text:
            frequency[token] += 1
            
    texts = [[token for token in text if frequency[token] > 0]
              for text in texts]
                  
    dictionary = corpora.Dictionary(texts)
    corpusdic = [dictionary.doc2bow(text) for text in texts]
    return  dictionary,corpusdic


def build_w2v_corpusdic(corpus):
    stemmed_corpus = list()

    # list of normalized sentences (remove the common phrases)
    for sentence_ in corpus:
        stemmed_corpus.append(normalize(str(sentence_)))

    # tokenize the documents
    texts = [[word for word in document.lower().split()]
             for document in stemmed_corpus]

    frequency = defaultdict(int)

    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 0]
             for text in texts]

    return texts
    
def _sort_by_scores(sim_index, sim_score):
    """
    Sorts rows of ``sim_index`` matrix, using rows of ``sim_score`` matrix
    """
    for (i_, (indx_, score_)) in enumerate(zip(sim_index, sim_score)):
        sorting_indx = np.argsort(-score_)
        sorted_scores = score_[sorting_indx]
        sorted_indx = indx_[sorting_indx]
        sim_index[i_] = sorted_indx
        sim_score[i_] = sorted_scores
        
    return (sim_index, sim_score)


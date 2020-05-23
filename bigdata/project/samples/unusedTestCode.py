# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:46:56 2020

@author: Janice
"""
#%% imports TODO ckean up imports

from gensim import models, corpora
from gensim.utils import simple_preprocess
import numpy as np

import warnings
# Suppress annoying deprecation messages which I'm not going to fix yet
warnings.filterwarnings("ignore", category=DeprecationWarning)

import importlib
# importlib.import_module("rssreader.reader")
importlib.import_module("reader")
importlib.import_module("topicmap")

from gensim import models, corpora
from gensim.utils import simple_preprocess
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import ipywidgets as widgets
from IPython import display
from gensim.matutils import softcossim 
import datetime
import time

import gensim.downloader as api
from dateutil.parser import *

import pyLDAvis
import pyLDAvis.gensim
from bokeh.io import  show, output_notebook, output_file

import matplotlib.pyplot as plt

import warnings
# Suppress annoying deprecation messages which I'm not going to fix yet
warnings.filterwarnings("ignore", category=DeprecationWarning)

#%% tfidfTest TODO probably delete since it's a test method
def tfidfTest():

    
    documents = ["This is the first line",
                 "This is the second sentence",
                 "This third document"]
    
    # Create the Dictionary and Corpus
    mydict = corpora.Dictionary([simple_preprocess(line) for line in documents])
    corpus = [mydict.doc2bow(simple_preprocess(line)) for line in documents]
    
    # Show the Word Weights in Corpus
    for doc in corpus:
        print([[mydict[id], freq] for id, freq in doc])
    
    # [['first', 1], ['is', 1], ['line', 1], ['the', 1], ['this', 1]]
    # [['is', 1], ['the', 1], ['this', 1], ['second', 1], ['sentence', 1]]
    # [['this', 1], ['document', 1], ['third', 1]]
    
    # Create the TF-IDF model
        
    tfidf = models.TfidfModel(corpus, smartirs='ntc')
    
    # Show the TF-IDF weights
    for doc in tfidf[corpus]:
        print([[mydict[id], np.around(freq, decimals=2)] for id, freq in doc])
    return

#%% print_top_words

def print_top_words(model, feature_names, n_top_words):
    for index, topic in enumerate(model.components_):
        message = "\nTopic #{}:".format(index)
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1 :-1]])
        print(message)
        print("="*70)
    return

#%% cosineSimilarityTest TODO delete
def cosineSimilarityTest():

    documents = getTestDocuments()
    # Create the Document Term Matrix
    count_vectorizer = CountVectorizer(stop_words='english')
    count_vectorizer = CountVectorizer()
    sparse_matrix = count_vectorizer.fit_transform(documents)
    
    # OPTIONAL: Convert Sparse Matrix to Pandas Dataframe if you want to see the word frequencies.
    doc_term_matrix = sparse_matrix.todense()
    df = pd.DataFrame(doc_term_matrix, 
                      columns=count_vectorizer.get_feature_names(), 
                      index=['doc_trump', 'doc_election', 'doc_putin'])
    print(cosine_similarity(df, df))
    """
    [[1.         0.51480485 0.38890873]
     [0.51480485 1.         0.38829014]
     [0.38890873 0.38829014 1.        ]]
    
    Interpretation: 
    doc_trump is more similar to doc_election (0.51) than to doc_putin (0.39)
    """
    return df
#%% testTopicMaps
def testTopicMaps(stopW="english"):
    sentence = ["The stop_words_ attribute can get large and increase the model size when . This attribute is provided only for introspection and can be safely removed using delattr or set to None before pickling.",
            "I love to eat Fries"]
    vectorizer =LemmaCountVectorizer(min_df=0, stop_words=stopW)
    sentence_transform = vectorizer.fit_transform(sentence)
    
    feature_names = vectorizer.get_feature_names()

    count_vec = np.asarray(sentence_transform.sum(axis=0)).ravel()
    zipped = list(zip(feature_names, count_vec))
    return zipped
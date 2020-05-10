# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:44:18 2020

@author: Janice
"""

from nltk.stem import WordNetLemmatizer
import numpy as np

# Plotly imports
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

import importlib
importlib.import_module("rssreader.reader")
from rssreader.reader import getDocList

# Other imports
# from scipy.misc import imread
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from matplotlib import pyplot as plt

from nltk.corpus import stopwords

# Do one time only to get wordnet for Lemmatization
# import nltk
# nltk.download() # --> and choose Corpora/wordnet

# Source: https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial#3.-Topic-modelling

#%%
#Subclassing 
lemm = WordNetLemmatizer()
class LemmaCountVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(LemmaCountVectorizer, self).build_analyzer()
        return lambda doc: (lemm.lemmatize(w) for w in analyzer(doc))
    
#%%
def print_top_words(model, feature_names, n_top_words):
    for index, topic in enumerate(model.components_):
        message = "\nTopic #{}:".format(index)
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1 :-1]])
        print(message)
        print("="*70)
        
#%%
# def 
# text = list(train.text.values)
# # Calling our overwritten Count vectorizer
# tf_vectorizer = LemmaCountVectorizer(max_df=0.95, 
#                                      min_df=2,
#                                      stop_words='english',
#                                      decode_error='ignore')
# tf = tf_vectorizer.fit_transform(text)

#%%
def deriveTopicMaps(sentences, stopW=getCustomStopWords(), maxNum=30, ngram_range=(3,4)):
    """
    Using TfidfVectorizer with added Lemmatization, derive the given number
    of "Topics" from the supplied sentences.
    The implicit use of TfidfTransformer additionally scales down the impact 
    of tokens that occur very frequently in the given corpus and that are hence 
    empirically less informative than features occuring in a small fraction 
    of the corpus.

    Parameters
    ----------
    sentences : list
        list of articles from corpus
    stopW : List of stop words
        DESCRIPTION. The default is getCustomStopWords() which is based
        on "english" plus other noisy stuff.
    maxNum : int, optional
        DESCRIPTION. The maximum nr. of topics to generate (default is 30)

    Returns
    -------
    zipped : list of word:frequency tuples 
        DESCRIPTION.

    """
    vectorizer = LemmaCountVectorizer(max_df=0.85,
                                      min_df=3, 
                                      ngram_range=ngram_range, #short phrases seem to work better than single words
                                      max_features=maxNum,
                                      stop_words=stopW
                                      )
    sentence_transform = vectorizer.fit_transform(sentences)
    
    feature_names = vectorizer.get_feature_names()

    count_vec = np.asarray(sentence_transform.sum(axis=0)).ravel()
    zipped = list(zip(feature_names, count_vec))
    return zipped

#%%
def testTopicMaps(stopW="english"):
    sentence = ["The stop_words_ attribute can get large and increase the model size when pickling. This attribute is provided only for introspection and can be safely removed using delattr or set to None before pickling.", 
            "I love to eat Fries"]
    vectorizer =LemmaCountVectorizer(min_df=0, stop_words=stopW)
    sentence_transform = vectorizer.fit_transform(sentence)
    
    feature_names = vectorizer.get_feature_names()

    count_vec = np.asarray(sentence_transform.sum(axis=0)).ravel()
    zipped = list(zip(feature_names, count_vec))
    return zipped

#%%
def getCustomStopWords():
    """
    Add any expressions that need to be ignored in addition to the nltk.corpus
    stoplist for english

    Returns
    -------
    myStopWords : list

    """
    myStopWords = list(set(stopwords.words('english')))
    myStopWords.extend(["view", "entire", "post", "twitter","com", "share","story",
                        "interested", "friends","interested", "would", "also", "rt"
                        "cipher", "brief", "continue", "reading", "onenewszetu",
                        "offibiza", "linkthe", "haunting"])
    return myStopWords

#%%
def getCustomStopPhrases():
    """
    Add any expressions that need to be ignored in addition to the nltk.corpus
    stoplist for english

    Returns
    -------
    myStopWords : list

    """
    myStopWords=["view entire post", "post", "twitter.com","twitter com", "share story",
                        "interested friends","interested", "would", "also rt",
                        "cipher brief", "continue reading", "year old", "per cent", 
                        "last week", "first time", "last year", "last month",
                        "around the world", "year-old", "live blog", "like story share",
                        "story share", "think friends share", "NewsZetu", 
                        "rt.com", "Z6Mag", "onz6", "friends", "first onworld weekly news",
                        "first onworld weekly", "onworld weekly news", "offibiza",
                        "latest news"]
    return myStopWords

#%%
def plotLeadingWords(topicList):
    x, y = (list(x) for x in zip(*sorted(topicList, key=lambda x: x[1], reverse=True)))
    # Now I want to extract out on the top 15 and bottom 15 words
    Y = np.concatenate([y[0:15], y[-16:-1]])
    X = np.concatenate([x[0:15], x[-16:-1]])
    
    # Plotting the Plot.ly plot for the Top 50 word frequencies
    data = [go.Bar(
                x = x[0:50],
                y = y[0:50],
                marker= dict(colorscale='Jet',
                             color = y[0:50]
                            ),
                text='Word counts'
        )]
    
    layout = go.Layout(
        title='Top 50 Word frequencies after Preprocessing'
    )
    
    fig = go.Figure(data=data, layout=layout)
    
    py.iplot(fig, filename='basic-bar')
    
    # Plotting the Plot.ly plot for the Top 50 word frequencies
    data = [go.Bar(
                x = x[-100:],
                y = y[-100:],
                marker= dict(colorscale='Portland',
                             color = y[-100:]
                            ),
                text='Word counts'
        )]
    
    layout = go.Layout(
        title='Bottom 100 Word frequencies after Preprocessing'
    )
    
    fig = go.Figure(data=data, layout=layout)
    
    py.iplot(fig, filename='basic-bar')
    return
#%%

# allDict=loadAllFeedsFromFile()
# docl=getDocList(allDict, limit=100, reloaddocs=False, stop_list=)
# topics= deriveTopicMaps(docl)
    # Play with ngrams to find sensible topics
# topics= deriveTopicMaps(docl,ngram_range=(1,2)) # Produces nonsense
# topics= deriveTopicMaps(docl,ngram_range=(2,2)) # Produces recognisable topics but with many repetitions in different constellations
    
# topics= deriveTopicMaps(docl,ngram_range=(4,4), maxNum=20) # with 4,4 you find completely differnt tabloid type stories (possibly they are all agency copy+paste stories?)
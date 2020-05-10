# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:44:18 2020

@author: Janice
"""

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import base64
import numpy as np
import pandas as pd

# Plotly imports
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

# Other imports
from collections import Counter
# from scipy.misc import imread
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from matplotlib import pyplot as plt

# Do one time only to get wordnet for Lemmatization
# import nltk
# nltk.download() # --> and choose Corpora/wordnet

# Source: https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial#3.-Topic-modelling

#%%
#Subclassing 
lemm = WordNetLemmatizer()
class LemmaCountVectorizer(CountVectorizer):
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

def testTopicMaps():
    sentence = ["The stop_words_ attribute can get large and increase the model size when pickling. This attribute is provided only for introspection and can be safely removed using delattr or set to None before pickling.", 
            "I love to eat Fries"]
    vectorizer = LemmaCountVectorizer(min_df=0)
    sentence_transform = vectorizer.fit_transform(sentence)
    
    feature_names = vectorizer.get_feature_names()

    count_vec = np.asarray(sentence_transform.sum(axis=0)).ravel()
    zipped = list(zip(feature_names, count_vec))
    return zipped

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
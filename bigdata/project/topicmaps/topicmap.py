# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:44:18 2020

@author: Janice
"""
#%% imports

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

import pandas as pd
import numpy as np
import re
import random
from collections import Counter, defaultdict

# Plotly imports
import plotly.offline as py
py.init_notebook_mode(connected=True)
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


import importlib
importlib.import_module("reader")
from reader import loadAllFeedsFromFile,getStringContents, getAllTags
# importlib.import_module("rssreader.reader")
# importlib.import_module("reader")
# from reader import getDocList,loadAllFeedsFromFile,getStringContents

# Other imports
# from scipy.misc import imread
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from fuzzywuzzy import process # for Levenshtein  Distance calculations

# Suppress annoying deprecation messages from nltk which I'm not going to fix yet
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# TODO something for the readme.txt Do one time only to get wordnet for Lemmatization
# import nltk
# nltk.download() # --> and choose Corpora/wordnet

# Source: https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial#3.-Topic-modelling

#%% getDocList

def getDocList(allEntryDict=None, limit = None, reloaddocs= False, 
               stop_list=None, with_ids=False):
    """
    

    Parameters
    ----------
    allEntryDict : dict, optional
        DESCRIPTION. Dictionalry og RSS entries. The default is None.
    limit : int, optional
        DESCRIPTION. Max number of entries to return. The default is None.
    reloaddocs : TYPE, optional
        DESCRIPTION. The default is False.
    stop_list : list, optional
        DESCRIPTION. List of words or phrases which will be removed from all 
        finished articles. Case insensitive removal. The default is None.
    with_ids : bool False
        DESCRIPTION. True if a zip of docids and contents should be fetched)         
    Returns
    -------
    docs : TYPE
        DESCRIPTION.

    """
    
    if reloaddocs or not allEntryDict:
        allEntryDict=loadAllFeedsFromFile()
        
    docs=[]
    ids=[]
    i=0 # use to break out at limit
    for key, val in allEntryDict.items():
        i +=1
        finalVal=val.collatedContents        
        if stop_list :   #substitute all phrases for ' ' case insensitive
            finalVal = removeStopWords(finalVal, stop_words=stop_list)
        docs.append(finalVal)
        ids.append(key)
        if limit and i > limit :
            break
        
    return zip(ids,docs) if with_ids  else docs

#%% LemmaCountVectorizer
#Subclassing 
lemm = WordNetLemmatizer()
class LemmaCountVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(LemmaCountVectorizer, self).build_analyzer()
        return lambda doc: (lemm.lemmatize(w) for w in analyzer(doc))
    
#%% print_top_words

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

#%% Stop word processing
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
                        "offibiza", "linkthe", "haunting", "blogfor", "live"])
    # myStopWords.extend(getCustomStopPhrases())
    return myStopWords

#%% getCustomStopPhrases
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
                        "latest news", "blogforum"]
    return myStopWords

def removeStopWords(str, stop_words = getCustomStopWords()):
    """
    Use NLTK to remove stop words from string
    Parameters
    ----------
    str : TYPE
        DESCRIPTION. String to process
    stop_words : TYPE, optional
        DESCRIPTION. The default is set(stopwords.words('english')).

    Returns
    -------
    filtered_sentence : str
        DESCRIPTION. str with stop word removed
    """

    word_tokens = word_tokenize(str)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    
    filtered_sentence = []
    for w in word_tokens:
        if w.lower() not in stop_words:
            filtered_sentence.append(w)
    from nltk.tokenize.treebank import TreebankWordDetokenizer

    return TreebankWordDetokenizer().detokenize(filtered_sentence)

#%% deriveTopicMaps
def deriveTopicMaps(sentences, stopW=getCustomStopWords(), maxNum=30, ngram_range=(3,4)):
    """
    Using 
    with added Lemmatization, derive the given number
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

#%% testTopicMaps
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
# def plotLeadingWords(topicList):
#     x, y = (list(x) for x in zip(*sorted(topicList, key=lambda x: x[1], reverse=True)))
#     # Now I want to extract out on the top 15 and bottom 15 words
#     Y = np.concatenate([y[0:15], y[-16:-1]])
#     X = np.concatenate([x[0:15], x[-16:-1]])
    
#     # Plotting the Plot.ly plot for the Top 50 word frequencies
#     data = [go.Bar(
#                 x = x[0:50],
#                 y = y[0:50],
#                 marker= dict(colorscale='Jet',
#                              color = y[0:50]
#                             ),
#                 text='Word counts'
#         )]
    
#     layout = go.Layout(
#         title='Top 50 Word frequencies after Preprocessing'
#     )
    
#     fig = go.Figure(data=data, layout=layout)
    
#     py.iplot(fig, filename='basic-bar')
    
#     # Plotting the Plot.ly plot for the Top 50 word frequencies
#     data = [go.Bar(
#                 x = x[-100:],
#                 y = y[-100:],
#                 marker= dict(colorscale='Portland',
#                              color = y[-100:]
#                             ),
#                 text='Word counts'
#         )]
    
#     layout = go.Layout(
#         title='Bottom 100 Word frequencies after Preprocessing'
#     )
    
#     fig = go.Figure(data=data, layout=layout)
    
#     py.iplot(fig, filename='basic-bar')
#     return
#%% unzipLeftSide
def unzipLeftSide(iterable):
    return zip(*iterable).__next__()

#%% TODO Do same thing for CosineSimilarities
# TODO Do same thing for CosineSimilarities (but multiply them by 100)
# remake the matrix with the full Ids of the Documents, then write them back 
# to the allEntryDict
# need to do deletions (if at all) after testFuzz and testSoftCosine have
# been applied

#%% testFuzz

def testFuzz(topic_list, allEntryDict, limit = None, threshold=75):
    """
    Add list of topics to each entry of the given allEntryDict for each topic
    that has a fuzzy relevance of greater than the specified threshold

    Parameters
    ----------
    topic_list : list of tuple (topic , weight)
        DESCRIPTION. List of topics and their relevance for the whole corpus
    allEntryDict : dict, optional
        DESCRIPTION. Dictionalry og RSS entries. The default is None.
    limit : int, optional
        DESCRIPTION. Max number of entries to return. The default is None.
    reloaddocs : TYPE, optional
        DESCRIPTION. The default is False.
    threshold : float value for assigning relevant topics

    Returns
    -------
    void
    """
    
    topics=unzipLeftSide(topic_list) #just get the phrases)
    toBeRemoved=[]
    if not allEntryDict:
        allEntryDict=loadAllFeedsFromFile()
        
    i=0 # use to break out at limit
    for key, val in allEntryDict.items():
        html=""
        if hasattr(val , "content"):
            for line in val.content:
                html = html + " " + line.value
        elif hasattr(val , "summary_detail"):
            html = val.summary_detail.value
        else:
            continue
        i +=1
        finalVal = val.title +" " + getStringContents(html)
        # import pdb
        # pdb.set_trace()
        try:
            matchedTopics=process.extract(finalVal,topics)
        except:
            print("An exception occurred with:", key)
        goodTops = [tupl for tupl in matchedTopics if tupl[1] > threshold]
        if len(goodTops) > 0:
            val["topiclist"]=goodTops
        else:
            val["topiclist"]=None
            toBeRemoved.append(key)
        if limit and i > limit :
            break
    # for gone in toBeRemoved:
    #     allEntryDict.pop(gone)        
    return toBeRemoved

#%% displayTopicsAndFeeds
def displayTopicsAndFeeds(allItemDict, numTopics=30):
    sns.set()
    # plt.xticks(rotation=60)
    plt.xticks(rotation=45, horizontalalignment='right')
    feedTuple=getAllFeedTopics(allItemDict)
    
    feeds=[]
    allTopics=getAllTopics(allItemDict)
    c_Topics=Counter(allTopics)
    topN=c_Topics.most_common(numTopics)
    Topicnames=[item[0] for item in topN]
    for feed,nrTopics in feedTuple[0].items():
        feeds.append(feed)
    
    matr=np.zeros( (len(feeds),len(Topicnames) ) )
    df = pd.DataFrame(data= matr, columns=Topicnames, index=feeds)
    populateTopicMatrix(allItemDict, df)
    df2=makeTopicMatrix(df)
    sns.set_context("paper", font_scale=1.0)
    # sns.set_style("whitegrid", {'axes.grid' : False})
    cmap = sns.cubehelix_palette (dark = .3, light=.8, as_cmap=True)
    ax = sns.scatterplot(data=df2,x="Feeds", y="Topics", size="Number", 
                         hue="Number",sizes=(100,300), markers = False)
    ax.tick_params(labelsize=5)
    plt.title("Topic Usage in RSS Feeds")
    plt.show()
    return ax

#%% populateTopicMatrix

def populateTopicMatrix(allDocDict, feedTopicMatrix):
    """
    Calculate from allDocDict how many of the specified topics occur for each 
    FeedItem of the named feeds in feedTopicMatrix, summing them in the 
    x.y position (Feed, Tagname) in the given matrix

    Parameters
    ----------
    allDocDict : TYPE
        DESCRIPTION.
    feedTopicMatrix : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    """
    topics=[]
    for key, val in allDocDict.items():
        if hasattr(val , "topiclist") and val.topiclist:
            for topicItem in val.topiclist:
                 topic = topicItem[0]
                 if topic in feedTopicMatrix.columns and val.feed_name in feedTopicMatrix.index:
                     feedTopicMatrix[topic][val.feed_name] +=1
    return

#%% preProcessDocs
def preProcessDocs(docList):
    """
    Remove stop words and phrases

    Parameters
    ----------
    dict : List
        DESCRIPTION. list of document contents
    Parameters
    ----------
    docList : TYPE
        DESCRIPTION.

    Returns
    -------
    newDocList : TYPE
        DESCRIPTION.

    """
    newDocList=[]

    # Remove punctuation
    docList['paper_text_processed'] = docList['paper_text'].map(lambda x: re.sub('[,\.!?]', '', x))
    
    # Convert the titles to lowercase
    docList['paper_text_processed'] = docList['paper_text_processed'].map(lambda x: x.lower())

    return newDocList

#%% smallDict utility
def smallDict(d, sample=10):
    keys = random.sample(list(d), sample)
    values = [d[k] for k in keys]
    return dict(zip(keys, values))

def doStandardInitialize():
    allDict=loadAllFeedsFromFile()
    docl=getDocList(smallDict(allDict,500), reloaddocs=False,stop_list=getCustomStopWords())
    # docl=preProcessDocs(docs)
    topics= deriveTopicMaps(docl, maxNum=30, ngram_range=(3,4)) # Produces recognisable topics but with many repetitions in different constellations
    testFuzz(topics,smallDict(allDict,500), limit=None, threshold=70)
    return
#%%

def getAllFeedTopics(allDocDict):
    """
    Collect the number of topics per feed from given dictionary

    Parameters
    ----------
    allDocDict : TYPE
        DESCRIPTION.
    reload : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """

    feedTopicdict=defaultdict(lambda: 0)
    feedTopicnamesdict=defaultdict(lambda: set())
    # Listitem= allDocDict[0]
    for key, val in allDocDict.items():
        if hasattr(val , "topiclist") and val.topiclist:
            feedTopicdict[val["feed_name"]] += len(val["topiclist"])
            for TopicItem in val["topiclist"]:
                feedTopicnamesdict[val["feed_name"]].add(TopicItem[0])

    for key,val in feedTopicnamesdict.items():
        feedTopicnamesdict[key]=list(val)
        
    return (feedTopicdict,feedTopicnamesdict)
#%%
def makeTopicMatrix(df):
        
    allTopics=[]
    allFeeds=[]
    num=[]
    for fd in df.index:
        for topItem in df.columns:
            if df[topItem][fd] > 0:
                allFeeds.append(fd) 
                allTopics.append(topItem)
                num.append(df[topItem][fd])
                     
    df = pd.DataFrame({"Feeds": allFeeds, "Topics" : allTopics, "Number": num})
    return df
 
#%% getAllTopics
# Has to have testFuzz called beforehand to populate topiclist correctly
def getAllTopics(allDocDict):
    
    topics=[]
    nwith=0
    nwithout=0
    for key, val in allDocDict.items():
        if hasattr(val , "topiclist") and val.topiclist:
            nwith +=1
            for topicItem in val["topiclist"]:
                topics.append(topicItem[0])
        else:
            nwithout +=1
            
    print("="*90,  "\nThere were", nwith, "items with topics and", nwithout, "without topics")
    c_topics=Counter(topics)
    print("="*90, "\nThese are the 20 most frequent topics used:\n","="*90,"\n",c_topics.most_common(20))

    return topics

#%% testDisplayTopicsAndFeeds Scatterplot
def testDisplayTopicsAndFeeds(numArticles=300, dict=None, numTopics=30,
                              ngram_range=(3,3)):
    if not dict:
        dict=loadAllFeedsFromFile()

    docl=getDocList(smallDict(dict,numArticles),
                    reloaddocs=False,
                    stop_list=getCustomStopWords())
    topics= deriveTopicMaps(docl, maxNum=numTopics, ngram_range=ngram_range)
    testFuzz(topics,dict) # populates topiclist in dict entries
    displayTopicsAndFeeds(dict)
    return

#%% Test Code for Topic Maps and Fuzzy

# allDict=loadAllFeedsFromFile()
# # docl=getDocList(allDict, limit=1000, reloaddocs=False)
# docl=getDocList(allDict, reloaddocs=False)
# # topics= deriveTopicMaps(docl)
#     # Play with ngrams to find sensible topics
# # topics= deriveTopicMaps(docl,ngram_range=(1,2)) # Produces nonsense
# # topics= deriveTopicMaps(docl,ngram_range=(2,2)) # Produces recognisable topics but with many repetitions in different constellations
# topics= deriveTopicMaps(docl, maxNum=30, ngram_range=(3,4)) # Produces recognisable topics but with many repetitions in different constellations
# # testFuzz(topics,allDict, limit=10)
# len(testFuzz(topics,allDict, limit=None, threshold=70))
# # topics= deriveTopicMaps(docl,ngram_range=(4,4), maxNum=20) # with 4,4 you find completely differnt tabloid type stories (possibly they are all agency copy+paste stories?)
# allDict1=loadAllFeedsFromFile()
# small=smallDict(allDict1,300)
# dl=getDocList(small, stop_list=getCustomStopWords())

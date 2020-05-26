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
import math

# Plotly imports
import plotly.offline as py
py.init_notebook_mode(connected=True)
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


import importlib
importlib.import_module("reader")
from reader import (loadAllFeedsFromFile,getStringContents,
                    getAllTags,loadPickleArticles,getRssArticleDate)
# importlib.import_module("rssreader.reader")
# importlib.import_module("reader")
# from reader import getDocList,loadAllFeedsFromFile,getStringContents

# Other imports
# from scipy.misc import imread
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from fuzzywuzzy import process # for Levenshtein  Distance calculations
from nltk.tokenize.treebank import TreebankWordDetokenizer

# Suppress annoying deprecation messages from nltk which I'm not going to fix yet
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from tqdm.notebook import trange, tqdm

# TODO something for the readme.txt Do one time only to get wordnet for Lemmatization
# import nltk
# nltk.download() # --> and choose Corpora/wordnet
# For data-structure visualization
# conda install python-graphviz
# pip install lolviz

# Source: https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial#3.-Topic-modelling

#%% getDocList

def getDocList(allEntryDict=None, limit = None, reloaddocs= False, 
               stop_list=None, with_ids=False):
    """
    Returns either a list of RSSEntry contents with stop words removed, limited in length
    to limit (if set) or, if with_ids is True, the article UIDs are zipped together
    with the document contents.

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
    for key, val in tqdm( allEntryDict.items(), desc="Removing Stop Words"):
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

#%% unzipLeftSide
def unzipLeftSide(iterable):
    return zip(*iterable).__next__()

#%% TODO Do same thing for CosineSimilarities
# TODO Do same thing for CosineSimilarities (but multiply them by 100)
# remake the matrix with the full Ids of the Documents, then write them back 
# to the allEntryDict
# need to do deletions (if at all) after F and testSoftCosine have
# been applied

#%% updateDictionaryByFuzzyRelevanceofTopics

def updateDictionaryByFuzzyRelevanceofTopics(topic_list, allEntryDict, limit = None, threshold=75, remove=False):
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
    for key, val in tqdm(allEntryDict.items(), desc='Fuzzy Relevance Testing'):
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
            toBeRemoved.append(key)
            # print("An exception occurred with:", key)
        goodTops = [tupl for tupl in matchedTopics if tupl[1] > threshold]
        if len(goodTops) > 0:
            val["topiclist"]=goodTops
        else:
            val["topiclist"]=None
            toBeRemoved.append(key)
        if limit and i > limit :
            break

    if remove:#remove non topics from dict
        for gone in tqdm(toBeRemoved, desc="Removing documents"):
            try:
                allEntryDict.pop(gone)
            except:
                print ("removal of", key, "not possible")

    return toBeRemoved
#%% simpleTopicDisplay Histogram
def simpleTopicDisplay(ax,topnames,topNumbers):

    topList = pd.DataFrame({"Topics": topnames,
                        "Frequency":topNumbers
                         })
    topList = topList.sort_values('Frequency',ascending=True).reset_index()
    
    # Plot the total crashes
    # sns.set_color_codes("pastel")
    cmap = sns.cubehelix_palette (40, dark = .3, light=.8,start=0.9,
                                  rot=-1.0,gamma=0.8, as_cmap=False)
    sns.barplot(y="Topics", x="Frequency", data=tagList,
                label="Tags", palette=cmap)

    # Add a legend and informative axis label
    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(xlim=(0, max(topNumbers)+5), xlabel="",
            ylabel="Topics designated to articles")
    plt.title("Topic Frequency Overall", fontsize=20)
    return

#%% displayTopicsAndFeeds
def displayTopicsAndFeeds(allItemDict, numTopics=30):
    sns.set()
    # plt.xticks(rotation=60)
    # plt.figure(figsize=(50,100))
    sns.set(rc={'figure.figsize':(14,5+numTopics*0.35)})

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
    # cmap = sns.cubehelix_palette (dark = .3, light=.8, as_cmap=True)
    cmap = sns.cubehelix_palette (10, dark = .3, light=.8,start=0.9, rot=-1.0,gamma=0.8, as_cmap=True)

    ax = sns.scatterplot(data=df2,x="Feeds", y="Topics", size="Number", 
                         hue="Number",sizes=(100,300), markers = False, palette=cmap)
    ax.tick_params(labelsize=12)
    plt.title("Topic Usage in RSS Feeds", fontsize=20)
    plt.tight_layout()
    plt.show()
    return

#%% displayTopicsAndFeeds
def displayTopicsAndFeeds2(allItemDict, numTopics=30):
    sns.set()
    # plt.xticks(rotation=60)
    # plt.figure(figsize=(50,100))
    sns.set(rc={'figure.figsize':(14,5+numTopics*0.75)})

    plt.xticks(rotation=45, horizontalalignment='right')
    feedTuple=getAllFeedTopics(allItemDict)
    
    feeds=[]
    allTopics=getAllTopics(allItemDict)
    c_Topics=Counter(allTopics)
    topnames=[item[0] for item in c_Topics]
    topNumbers=[item[1] for item in c_Topics]

    allTopics=getAllTopics(allItemDict)
    c_Topics=Counter(allTopics)
    topN=c_Topics.most_common(numTopics)
    Topicnames=[item[0] for item in topN]
    for feed,nrTopics in feedTuple[0].items():
        feeds.append(feed)

    fig = plt.figure()
    ax = fig.add_subplot(211)
    widg1=simpleTopicDisplay(ax,topnames,topNumbers)

    matr=np.zeros( (len(feeds),len(Topicnames) ) )
    df = pd.DataFrame(data= matr, columns=Topicnames, index=feeds)
    populateTopicMatrix(allItemDict, df)
    df2=makeTopicMatrix(df)
    sns.set_context("paper", font_scale=1.0)
    # sns.set_style("whitegrid", {'axes.grid' : False})
    # cmap = sns.cubehelix_palette (dark = .3, light=.8, as_cmap=True)
    cmap = sns.cubehelix_palette (10, dark = .3, light=.8,start=0.9, rot=-1.0,gamma=0.8, as_cmap=True)

    ax2 = fig.add_subplot(212)
    ax = sns.scatterplot(data=df2,x="Feeds", y="Topics", size="Number", ax=ax2,
                         hue="Number",sizes=(100,300), markers = False, palette=cmap)
    ax2.tick_params(labelsize=12)
    plt.title("Topic Usage in RSS Feeds", fontsize=20)
    plt.tight_layout()
    plt.show()
    return

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

#%% getAllFeedTopics

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
#%% makeTopicMatrix
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
# Has to have updateDictionaryByFuzzyRelevanceofTopics called beforehand to populate topiclist correctly
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
            
    print("\nThere were", nwith, "items with topics and", nwithout, "without topics")
    # c_topics=Counter(topics)
    # print("="*90, "\nThese are the 20 most frequent topics used:\n","="*90,"\n",c_topics.most_common(20))

    return topics
#%% Sentiment Analysis
def conductSentimentAnalysis(allDict):
    import nltk
    from nltk.sentiment import vader
    nltk.download('vader_lexicon')
    senti=vader.SentimentIntensityAnalyzer()

    for val in tqdm(list(allDict.values()), desc="Analyze Sentiment"):
        val["sentiment"]=senti.polarity_scores(val["collatedContents"])
    return
#%% getSentimentsForTopic
def getSentimentsForTopic(topic, dict):
    feeds=[]
    dates=[]
    types=[]
    value=[]
    id=[]
    for val in tqdm(dict.values(), desc = "Fetching Sentiments for '" + topic + "'"):
        # try:
        if topic in [item[0] for item in val.topiclist]:
            for typ in [("Positive","pos"),("Negative","neg"),("Neutral","neu"),("Overall","compound")]:
                feeds.append(val.get("feed_name"))
                dates.append(getRssArticleDate(val))
                types.append(typ[0])
                value.append(val.get("sentiment").get(typ[1]))
                id.append(val.get("id"))
        # except:
        #     continue
    df = pd.DataFrame({"Source": feeds,
                       "Published":dates,
                       "Sentiment Type":types,
                       "Sentiment Value":value,
                       "UID":id,
                         })
    return df

# def normalizeSentiValue(val):
#     return int(math.ceil(val*100))
#%%
def jointPlotOfSentiment(topic,dict0, sentitype="Positive"):
    df=getSentimentsForTopic(topic,dict0)
    sns.set(style="white", color_codes=True)
    # g = sns.jointplot(x="Source", y="Published", data=df)
    # plt.xticks(rotation=45, horizontalalignment='right')
    # .set_axis_labels("Source", "Published") #, scatter = False)
    # g.ax_joint.cla()
    # plt.sca(g.ax_joint)
    # plt.scatter(x="Source", y="Published", data=df,c=sentitype)

    # g = sns.FacetGrid(data=df, row="Source",col="Sentiment Type", hue="Positive")
    kws = dict(s=50, linewidth=.5, edgecolor="w")
    pal = dict(Negative="red", Positive="green", Neutral="yellow", Overall="blue")
    g = sns.FacetGrid(df, col="Sentiment Type", hue="Sentiment Type", palette=pal,
                  hue_order=["Positive", "Negative", "Neutral", "Overall"])
    g = (g.map(plt.heatmap, "Source", "Sentiment Value", **kws).add_legend())
    return
 
def heatmapOfSentiment(topic,dict0, sentitype="Positive"):
    df=getSentimentsForTopic(topic,dict0)
    #make a correlation matrix and display in heatmap
    # probably use DataFrame multiindex technique to extract the necessary
    correl=pd.DataFrame({"Source":set(df["Source"])})
    for feed in set(df["Source"]):
        vals=[]
        for day in set(df["Published"]):
            vals.append(df[feed,day][sentitype])
        correl[day]=vals


    df.plot(x='Source', y='Published', col='Sentiment Value')
    # [axis.set_aspect('equal') for axis in g.axes.ravel()]
    return

def testJointPlot(allDict, size=100):
    sm=smallDict(allDict,size)
    conductSentimentAnalysis(sm)
    docl=getDocList(sm, reloaddocs=False,stop_list=getCustomStopWords())
    topics= deriveTopicMaps(docl, maxNum=30, ngram_range=(3,3))
    updateDictionaryByFuzzyRelevanceofTopics(topics,sm, limit=None, threshold=20, remove=True)
    tlist=[item[0] for item in topics]
    # getSentimentsForTopic(tlist[0],sm)
    jointPlotOfSentiment(tlist[0],sm, "Positive")
    return

#%% smallDict utility
def smallDict(d, sample=10):
    keys = random.sample(list(d), sample)
    values = [d[k] for k in keys]
    return dict(zip(keys, values))

#%% doStandardInitialize
def doStandardInitialize(remove=False):
    allDict=loadAllFeedsFromFile()
    docl=getDocList(smallDict(allDict,500), reloaddocs=False,stop_list=getCustomStopWords())
    # docl=preProcessDocs(docs)
    topics= deriveTopicMaps(docl, maxNum=30, ngram_range=(3,4)) # Produces recognisable topics but with many repetitions in different constellations
    updateDictionaryByFuzzyRelevanceofTopics(topics,smallDict(allDict,500), limit=None, threshold=70, remove=remove)
    return allDict
#%% doStandardInitialize
def doFullTopicInitialize(remove=False, maxNum=30, ngram_range=(3,3),threshold=70):
    allDict=loadAllFeedsFromFile()
    docl=getDocList(allDict, reloaddocs=False,stop_list=getCustomStopWords())
    # docl=preProcessDocs(docs)
    topics= deriveTopicMaps(docl, maxNum=maxNum, ngram_range=ngram_range)
    updateDictionaryByFuzzyRelevanceofTopics(topics,allDict, limit=None, threshold=threshold, remove=remove)
    return allDict
#%% testDisplayTopicsAndFeeds Scatterplot
def testDisplayTopicsAndFeeds(numArticles=300, dict=None, numTopics=30,
                              ngram_range=(3,3)):
    if not dict:
        dict=loadAllFeedsFromFile()

    docl=getDocList(smallDict(dict,numArticles),
                    reloaddocs=False,
                    stop_list=getCustomStopWords())
    topics= deriveTopicMaps(docl, maxNum=numTopics, ngram_range=ngram_range)
    updateDictionaryByFuzzyRelevanceofTopics(topics,dict) # populates topiclist in dict entries
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
# # updateDictionaryByFuzzyRelevanceofTopics(topics,allDict, limit=10)
# len(updateDictionaryByFuzzyRelevanceofTopics(topics,allDict, limit=None, threshold=70, remove=True))
# # topics= deriveTopicMaps(docl,ngram_range=(4,4), maxNum=20) # with 4,4 you find completely differnt tabloid type stories (possibly they are all agency copy+paste stories?)
# allDict1=loadAllFeedsFromFile()
# small=smallDict(allDict1,300)
# dl=getDocList(small, stop_list=getCustomStopWords())

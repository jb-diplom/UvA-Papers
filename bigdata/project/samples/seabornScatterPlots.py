# -*- coding: utf-8 -*-
"""
Created on Wed May  6 21:31:08 2020

@author: Janice
"""
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter, defaultdict 

import importlib
# importlib.import_module("rssreader.reader")
importlib.import_module("reader")
from reader import loadAllFeedsFromFile,getStringContents, getAllTags
importlib.import_module("topicmap")
from topicmap import getDocList, smallDict, getAllTopics, deriveTopicMaps,updateDictionaryByFuzzyRelevanceofTopics, getCustomStopWords
import seaborn as sns
sns.set()

#%% displayTopics
# https://www.drawingfromdata.com/how-to-rotate-axis-labels-in-seaborn-and-matplotlib (sizing)

def displayTopics(topics):
    sns.set(style="whitegrid")
    
    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(13, 10))
    
    top=[]
    freq=[]
    for a,b in topics:
        top.append(a)
        freq.append(int(b))
        
    topData = pd.DataFrame({"Topic": top, 
                        "Frequency":freq ,
                        " ":"" })
    
    topData = topData.sort_values('Frequency',ascending=True).reset_index()
    
    # Plot the total crashes
    # sns.set_color_codes("pastel")
    cmap = sns.cubehelix_palette (40, dark = .3, light=.8,start=0.9, rot=-1.0,gamma=0.8, as_cmap=False)
    sns.barplot(y="Frequency", x="Topic", data=topData,
                label="Topics", palette=cmap)
    
    # Plot the crashes where alcohol was involved
    # sns.set_color_codes("muted")
    # sns.barplot(x="alcohol", y="abbrev", data=crashes,
    #             label="Alcohol-involved", color="b")
    
    # Add a legend and informative axis label
    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(ylim=(0, max(freq)+5), xlabel="",
           ylabel="Topics derived from all articles")
    plt.xticks(rotation=45, horizontalalignment='right')
    pltout=sns.despine(left=True, bottom=True)
    return pltout

#%% Authors over all Articles
def getAuthors(inDict):
    authors=[]
    nwith=0
    nwithout=0
    for key, val in inDict.items():
        try:
            # if hasattr(val , "authors"):
            #     authors.extend([n['name'] for n in val.authors])
            if hasattr(val , "author"):
                if ',' in val.author:
                    authors.extend(val.author.split (",") for val in authors)
                elif ' and ' in val.author:
                    authors.extend(val.author.split (" and "))
                else:
                    authors.append(val.author)
                nwith +=1
            else:
                nwithout +=1
        except:
            nwithout +=1
    for ix, auth in enumerate(authors):
        if ' and ' in auth:
            authors.extend(auth.split (" and "))
     # for auth in authors:
     #     if (len(auth))
    print("With author:",nwith, "Without author:", nwithout)
    return authors

#%% displayAuthors
def displayAuthors(theAuthors=None, dict=None):
    sns.set(style="whitegrid")

    if not theAuthors:
        theAuthors=getAuthors(dict)

    # Initialize the matplotlib figure
    authorfig, ax = plt.subplots(figsize=(16,20))
    plt.subplots_adjust()
    frequen2 = Counter (theAuthors)
    authFrq=frequen2.most_common(30)
    frequen=[n[1] for n in authFrq]
    auth=[n[0] for n in authFrq]
    
    authList = pd.DataFrame({"Authors": auth, 
                        "Frequency":frequen ,
                         })
    authList = authList.sort_values('Frequency',ascending=True).reset_index()
    
    # Plot the total crashes
    # sns.set_color_codes("pastel")
    cmap = sns.cubehelix_palette (40, dark = .3, light=.8,start=0.9,
                                  rot=-1.0,gamma=0.8, as_cmap=False)
    sns.barplot(y="Authors", x="Frequency", data=authList,
                label="Authors", palette=cmap)

    # Add a legend and informative axis label
    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(xlim=(0, max(frequen)+5), xlabel="",
            ylabel="Authors of all articles")
    plt.yticks(rotation=0, horizontalalignment='right')
    pltout=sns.despine(left=True, bottom=True)
    return pltout

#%% getAllFeedtags

def getAllFeedtags(allDocDict):
    """
    Collect the number of tags per feed from given dictionary

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

    feedtagdict=defaultdict(lambda: 0)
    feedtagnamesdict=defaultdict(lambda: set())
    # Listitem= allDocDict[0]
    for key, val in allDocDict.items():
        if hasattr(val , "tags"):
            feedtagdict[val["feed_name"]] += len(val["tags"])
            for tagItem in val["tags"]:
                feedtagnamesdict[val["feed_name"]].add(tagItem["term"])   
                
    for key,val in feedtagnamesdict.items():
        feedtagnamesdict[key]=list(val)
        
    return (feedtagdict,feedtagnamesdict)

#%% populateTagMatrix

def populateTagMatrix(allDocDict, feedTagMatrix):
    """
    Calculate from allDocDict how many of the specified tags occur for each 
    FeedItem of the named feeds in feedTagMatrix, summing them in the 
    x.y position (Feed, Tagname) in the given matrix

    Parameters
    ----------
    allDocDict : TYPE
        DESCRIPTION.
    feedTagMatrix : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
        
    tags=[]
    for key, val in allDocDict.items():
        if hasattr(val , "tags"):
            for tagItem in val["tags"]:
                 tag = tagItem["term"]
                 if tag in feedTagMatrix.columns and val.feed_name in feedTagMatrix.index:
                     feedTagMatrix[tag][val.feed_name] +=1
    return

#%% makeTagMatrix
def makeTagMatrix(df):
        
    allTags=[]
    allFeeds=[]
    num=[]
    for fd in df.index:
        for tagItem in df.columns:
            if df[tagItem][fd] > 0:
                allFeeds.append(fd) 
                allTags.append(tagItem)
                num.append(df[tagItem][fd])
                     
    df = pd.DataFrame({"Feeds": allFeeds, "Tags" : allTags, "Number": num})
    return df
 

#%% displayTags
def displayTags(allItemDict):
    sns.set()
    sns.set(style="whitegrid")
    # plt.xticks(rotation=60)
    # plt.figure(figsize=(16,20))
    sns.set(rc={'figure.figsize':(13,13)})

    plt.xticks(rotation=45, horizontalalignment='right')
    feedTuple=getAllFeedtags(allItemDict)
    
    feeds=[]
    allTags=getAllTags(allItemDict)
    c_tags=Counter(allTags)
    top30=c_tags.most_common(30)
    tagnames=[item[0] for item in top30]
    for feed,nrTags in feedTuple[0].items():
        feeds.append(feed)
    
    matr=np.zeros( (len(feeds),len(tagnames) ) )
    df = pd.DataFrame(data= matr, columns=tagnames, index=feeds)
    populateTagMatrix(allItemDict, df)
    df2=makeTagMatrix(df)
    # sns.set_context("paper", font_scale=1.0)
    # sns.set_style("whitegrid", {'axes.grid' : False})
    # cmap = sns.cubehelix_palette (5, dark = .2, light=.6,start=2.6, rot=0,gamma=0.8, as_cmap=True)
    # cmap = sns.dark_palette("blue",n_colors=6)
    # sns.set_palette(sns.color_palette("Paired", as_cmp=True))
    # cmap=sns.color_palette("Blues")
    cmap = sns.cubehelix_palette (10, dark = .3, light=.8,start=0.9, rot=-1.0,gamma=0.8, as_cmap=True)

    ax = sns.scatterplot(data=df2,x="Feeds", y="Tags", size="Number",
                         hue="Number",sizes=(50,400), markers = False, palette=cmap) #"Blues_r"
    ax.tick_params(labelsize=12)
    plt.title("Tag Usage in RSS Feeds", fontsize=20)
    plt.show()
    return

#%% testDisplayTopics Histogram
def testDisplayTopics(numArticles=None, numTopics=30, dict=None):
    if not dict:
        dict=loadAllFeedsFromFile()
    if not numArticles:
        small=smallDict(dict,numArticles)
    else:
        small=dict
    docl=getDocList(small, reloaddocs=False, stop_list=getCustomStopWords())
    # docl=getDocList(small, reloaddocs=False)
    topics= deriveTopicMaps(docl, maxNum=numTopics, ngram_range=(3,3))
    updateDictionaryByFuzzyRelevanceofTopics(topics, small, limit=30)
    displayTopics(topics)
    return

#%% testDisplayAuthors Histogram
def testDisplayAuthors(numArticles=300, dict=None):
    if not dict:
        dict=loadAllFeedsFromFile()
    small=smallDict(dict,numArticles)
    displayAuthors(dict = small)
    return

#%% testDisplayTags Scatterplot
def testDisplayTags(numArticles=300, dict=None):
    if not dict:
        dict=loadAllFeedsFromFile()
    small=smallDict(dict,numArticles)
    displayTags(small)
    return

#%%
# allDict1=loadAllFeedsFromFile()
# small=smallDict(allDict1,300)
# testDisplayTopics(dict=small)
# testDisplayAuthors(dict=small)
# testDisplayTags(dict=small)
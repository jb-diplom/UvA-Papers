# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:10:08 2020

@author: Janice
"""
from reader import loadAllFeedsFromFile,getStringContents,getRssArticleDate
from topicmap import (unzipLeftSide,smallDict,getDocList,
                      deriveTopicMaps,updateDictionaryByFuzzyRelevanceofTopics)
from Gensim.gensim_test import getAuthorFromRssEntry,getCustomStopWords

import ipywidgets as widgets
from IPython.display import display, clear_output
from ipywidgets import Layout, Button, Box, VBox, HBox

import nltk
from nltk.sentiment import vader
import matplotlib.pyplot as plt
import pandas as pd
from dateutil.parser import *

import plotly
import plotly.graph_objs as go
from tqdm.notebook import trange, tqdm

import warnings
warnings.filterwarnings("ignore")


#%% Sentiment Analysis
def conductSentimentAnalysis(allDict):
    
    nltk.download('vader_lexicon')
    senti=vader.SentimentIntensityAnalyzer()

    for val in tqdm(list(allDict.values()), desc="Analyze Sentiment"):
        val["sentiment"]=senti.polarity_scores(val["collatedContents"])
    return


#%% getSentimentsForTopic
def getSentimentsForTopic3(topic, dict):
    feeds=[]
    dates=[]
    positive=[]
    negative=[]
    neutral=[]
    combined=[]
    id=[]
    topicality=[]
    for key,val in tqdm(dict.items(), desc = "Fetching Sentiments for '" + topic + "'"):
        # try:
        if topic in [item[0] for item in val.topiclist]:
            for tup in val.topiclist:
                if tup[0]==topic:
                    topicality.append((tup[1]**2)/200)
                    break
            feeds.append(val.get("feed_name"))
            dates.append(getRssArticleDate(val))
            positive.append(val.get("sentiment").get("pos"))
            negative.append(val.get("sentiment").get("neg"))
            neutral.append(val.get("sentiment").get("neu"))
            combined.append(val.get("sentiment").get("compound"))
            id.append(key)
        # except:
        #     continue
    df = pd.DataFrame({"Source": feeds,
                       "Published":dates,
                       "Positive":positive,
                       "Negative":negative,
                       "Neutral":neutral,
                       "Overall":combined,
                       "UID":id,
                       "Topicality":topicality
                         })
    return df

#%% getSentimentsForTopic
def getTopicIdDict(dict):
    # Make a dictionary mapping topics to RSSEntry-Ids
    topicIdDict={}
    for key,val in tqdm(dict.items(), desc = "Mapping Topics"):
        try:
            if hasattr(val,"topiclist"):
                for tup in val.topiclist:
                    if not bool(topicIdDict.get(tup[0])):
                        topicIdDict[tup[0]]=[]
                    topicIdDict[tup[0]].append(key)
        except:
            continue
    return topicIdDict

#%%
def headerSentiText(rssEntry):
    maxlen=40

    src = rssEntry["feed_name"]
    posSent = rssEntry["sentiment"]["pos"]
    negSent = rssEntry["sentiment"]["neg"]
    neuSent = rssEntry["sentiment"]["neu"]
    compSent = rssEntry["sentiment"]["compound"]
    auth = getAuthorFromRssEntry(rssEntry)
    if hasattr(rssEntry , "updated"):
        dt=parse(rssEntry.updated, ignoretz=True)
    else:
        dt=parse(rssEntry.published, ignoretz=True)
    title = rssEntry.title
    if len(title) > maxlen:
        title = title[:maxlen] + "..."

    datestr=dt.strftime("%d/%m/%Y, %H:%M:%S")
    items=[]
    items.append(f"Title:           {title : <10}")
    items.append(f"Feed:          {src: <10}")
    items.append(f"Published:  {datestr: <10}")
    items.append(f"Sentiment:  " +
                 f"pos: {posSent : <8}"+
                 f"neg: {negSent : <8}"+
                 f"neut: {neuSent : <8}"+
                 f"comp: {compSent : <8}")

    str=""
    for token in items:
        str=str+(token)+("\n")

    return str
#%% tooltipText
def tooltipSentiText(rssEntry):
    """
    Pseudo HTML string for displaying the RSS-entry infos, currently 
    title, feedname, date of publication and author

    Parameters
    ----------
    rssEntry : TYPE
        DESCRIPTION.

    Returns
    -------
    str : TYPE
        DESCRIPTION.

    """
    maxlen=40

    src = rssEntry["feed_name"]
    posSent = rssEntry["sentiment"]["pos"]
    negSent = rssEntry["sentiment"]["neg"]
    neuSent = rssEntry["sentiment"]["neu"]
    compSent = rssEntry["sentiment"]["compound"]
    auth = getAuthorFromRssEntry(rssEntry)
    if hasattr(rssEntry , "updated"):
        dt=parse(rssEntry.updated, ignoretz=True)
    else:
        dt=parse(rssEntry.published, ignoretz=True)
    title = rssEntry.title
    if len(title) > maxlen:
        title = title[:maxlen] + "..."

    datestr=dt.strftime("%d/%m/%Y, %H:%M:%S")
    items=[]
    items.append(f"Title:           {title : <10}")
    items.append(f"Feed:         {src: <10}")
    items.append(f"Published: {datestr: <10}")
    items.append(f"<b>Sentiment: <br>" +
                 f"pos: {posSent : <8}"+
                 f"neg: {negSent : <8}"+
                 f"neut: {neuSent : <8}"+
                 f"comp: {compSent : <8}")

    str=""
    for token in items:
        str=str+(token)+("<br>")

    return str

#%%
def plotSentiment3D(df2, allDict, notebook=True, topic=""):
    statTooltips=[]
    for key in list(df2["UID"]):
        try:
            statTooltips.append(tooltipSentiText(allDict[key]))
        except:
            print (key, "not found")

    trace = go.Scatter3d(
        x=df2['Positive'],
        y=df2['Negative'],
        z=df2['Neutral'],
        mode='markers',
        marker=dict(
            size=df2["Topicality"],
            color=df2['Overall'],
            colorscale='Inferno',
            colorbar = dict(title= "Compound<br>Sentiment"),
            # symbol=df["specGroup"], # TODO actually want Feedname
            showscale=True,
            opacity=0.7,
        ),
        # symbol=df["specGroup"],
        text=statTooltips,
        textfont=dict(family="sans serif",size=8,color='crimson'),
        hoverinfo='text'
    )
    # Configure the layout.
    layout = go.Layout(showlegend=False,
                       title="<b>Sentiment Analysis for Topic '"+ topic +"'",
                       margin={'l': 0, 'r': 0, 'b': 50, 't': 30},
                       scene=go.Scene(
                           xaxis=go.XAxis(title='Positive<br>Sentiment'),
                           yaxis=go.YAxis(title='Negative<br>Sentiment'),
                           zaxis=go.ZAxis(title='Neutral<br>Sentiment')))
    data = [trace]
    plot_figure = go.Figure(data=data, layout=layout)
    camera = dict( eye=dict(x=1.5, y=1.5, z=0.1))
    plot_figure.update_layout(scene_camera=camera)

    #plt.tight_layout()
    go.FigureWidget(data=data, layout=layout)
   
    pl=plotly.offline.iplot(plot_figure)
    pl
    if not notebook:
        plotly.offline.plot(plot_figure, filename='file.html')
    return

#%% contentsViewer
def contentsViewer(allDict, topics):

    topicIdMap=getTopicIdDict(allDict)# Initialize one time to map topics to ids
    intro=widgets.Label(value=r'\(\textbf{Pre-select a topic and choose an article to view its contents}\)' )
    topicdd=widgets.Dropdown(options=unzipLeftSide(topics),
                             description='Topics:',
                             layout=Layout(flex='1 1 auto', width='30%'),
                             disabled=False)
    
    docs=widgets.Dropdown(description='RSS Entries:',
                          layout=Layout(flex='1 1 auto',width='70%'),
                          disabled=False)
    
    ta=widgets.Textarea(description='Content:',rows=10,
                        layout=Layout(flex='1 1 100%', width='auto'),
                        disabled=True)
    
    items_auto = [topicdd,docs]
    items_0 = [ta]
    box_layout = Layout(display='flex',
                        flex_flow='row',
                        align_items='stretch',
                        width='80%')
    box_auto = Box(children=items_auto, layout=box_layout)
    box_0 = Box(children=items_0, layout=box_layout)
    
    def updateDoclist(b):
        docs.options=getNewRSSEntryList(allDict,topicIdMap,topicdd.value)
    
    def get_and_plot(b):
        dispTxt=headerSentiText(allDict[docs.value])+"\n" + allDict[docs.value]["collatedContents"]
        ta.value=dispTxt

    # Initialize dropdown
    docs.options=getNewRSSEntryList(allDict,topicIdMap,topicdd.value)

    topicdd.observe(updateDoclist, names='value')
    docs.observe(get_and_plot, names='value')
    display(VBox([intro,box_auto, box_0]))
    updateDoclist(None) # just for initialization
    get_and_plot(None)  # just for initialization
    return

def getNewRSSEntryList(allDict, topicIdMap, topic):
    # return list of Title, Docid tuples
    vallist=[]
    for docid in topicIdMap[topic]:
        vallist.append((allDict[docid]["title"],docid))
    return vallist

#%% displaySentiment3D
def displaySentiment3D(allDict, topics, notebook=True):

    conductSentimentAnalysis(allDict)
    allTopics=unzipLeftSide(topics)
    intro=widgets.Label(value=r'\(\textbf{Select a topic for the 3d sentiment visualization}\)' )
    ddTopics=widgets.Dropdown(
        options=allTopics,
        description='Topic:',
        disabled=False)
    
    def callSentiViewer(b):
        clear_output(wait=True)
        display (ddTopics)
        df2=getSentimentsForTopic3(ddTopics.value,allDict)
        plotSentiment3D(df2, allDict, notebook=notebook, topic=ddTopics.value)

    ddTopics.observe(callSentiViewer, names='value')
    display(VBox([intro,ddTopics]))
    callSentiViewer(None) # init
    return

#%% runSentiment
def runSentiment (allDict, sm):
    allDict=loadAllFeedsFromFile()
    sm=smallDict(allDict,200)
    conductSentimentAnalysis(sm)
    docl=getDocList(sm, reloaddocs=False,stop_list=getCustomStopWords())
    topics= deriveTopicMaps(docl, maxNum=30, ngram_range=(3,3))
    updateDictionaryByFuzzyRelevanceofTopics(topics,sm, limit=None, threshold=20, remove=True)
    gt=getTopicIdDict(sm)
    # tlist=[item[0] for item in topics]
    # top=tlist[4]
    # df2=getSentimentsForTopic3(top,sm)
    # plotSentiment3D(df2, sm, notebook=False, topic=top)
    return

# allDict=loadAllFeedsFromFile()
# sm=smallDict(allDict,200)
# conductSentimentAnalysis(sm)
# docl=getDocList(sm, reloaddocs=False,stop_list=getCustomStopWords())
# topics= deriveTopicMaps(docl, maxNum=30, ngram_range=(3,3))
# updateDictionaryByFuzzyRelevanceofTopics(topics,sm, limit=None, threshold=20, remove=True)
# gt=getTopicIdDict(sm)

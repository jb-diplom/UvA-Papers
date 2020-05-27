# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:10:08 2020

@author: Janice
"""
from reader import loadAllFeedsFromFile,getStringContents
from topicmap import *
from Gensim.gensim_test import *
import nltk
from nltk.sentiment import vader

import plotly
import plotly.graph_objs as go
from tqdm.notebook import trange, tqdm

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
    items.append(f"Sentiment: <br>" +
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

    # colorDropdown = widgets.Dropdown(
    #     description='Topics',
    #     value=df["specGroup"][0],
    #     options=df["specGroup"]
    # )
    trace = go.Scatter3d(
        x=df2['Positive'],
        y=df2['Negative'],
        z=df2['Neutral'],
        mode='markers',
        marker=dict(
            size=df2["Topicality"],
            color=df2['Overall'],
            colorscale='Inferno',
            colorbar = dict(title= "Compound Sentiment"),
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
                       title="Sentiment Analysis for Topic '"+ topic +"'",
                       scene=go.Scene(
                           xaxis=go.XAxis(title='Positive Sentiment'),
                           yaxis=go.YAxis(title='Negative Sentiment'),
                           zaxis=go.ZAxis(title='Neutral Sentiment')))
    data = [trace]
    plot_figure = go.Figure(data=data, layout=layout)

    #plt.title(title, fontsize=16)
    #plt.tight_layout()
    go.FigureWidget(data=data, layout=layout)
   
    pl=plotly.offline.iplot(plot_figure)
    pl
    plotly.offline.plot(plot_figure, filename='file.html')
    return

#%%
# def jointPlotOfSentiment(topic,dict0, sentitype="Positive"):
#     df=getSentimentsForTopic(topic,dict0)
#     sns.set(style="white", color_codes=True)
#     # g = sns.jointplot(x="Source", y="Published", data=df)
#     # plt.xticks(rotation=45, horizontalalignment='right')
#     # .set_axis_labels("Source", "Published") #, scatter = False)
#     # g.ax_joint.cla()
#     # plt.sca(g.ax_joint)
#     # plt.scatter(x="Source", y="Published", data=df,c=sentitype)

#     # g = sns.FacetGrid(data=df, row="Source",col="Sentiment Type", hue="Positive")
#     kws = dict(s=50, linewidth=.5, edgecolor="w")
#     pal = dict(Negative="red", Positive="green", Neutral="yellow", Overall="blue")
#     g = sns.FacetGrid(df, col="Sentiment Type", hue="Sentiment Type", palette=pal,
#                   hue_order=["Positive", "Negative", "Neutral", "Overall"])
#     g = (g.map(plt.heatmap, "Source", "Sentiment Value", **kws).add_legend())
#     return
 
# def heatmapOfSentiment(topic,dict0, sentitype="Positive"):
#     df=getSentimentsForTopic(topic,dict0)
#     #make a correlation matrix and display in heatmap
#     # probably use DataFrame multiindex technique to extract the necessary
#     correl=pd.DataFrame({"Source":set(df["Source"])})
#     for feed in set(df["Source"]):
#         vals=[]
#         for day in set(df["Published"]):
#             vals.append(df[feed,day][sentitype])
#         correl[day]=vals
#         from scipy import stats

# from scipy import stats
# def qqplot(x, y, **kwargs):
#     _, xr = stats.probplot(x, fit=False)
#     _, yr = stats.probplot(y, fit=False)
#     sns.scatterplot(xr, yr, **kwargs)
#     g = sns.FacetGrid(df, col="Sentiment Type", hue="Sentiment Value")
#     g = (g.map(qqplot, "Source", "Published", **kws)
#       .add_legend())

#     df.plot(x='Source', y='Published', col='Sentiment Value')
#     # [axis.set_aspect('equal') for axis in g.axes.ravel()]
#     return

# def testJointPlot(allDict, size=100):
#     sm=smallDict(allDict,size)
#     conductSentimentAnalysis(sm)
#     docl=getDocList(sm, reloaddocs=False,stop_list=getCustomStopWords())
#     topics= deriveTopicMaps(docl, maxNum=30, ngram_range=(3,3))
#     updateDictionaryByFuzzyRelevanceofTopics(topics,sm, limit=None, threshold=20, remove=True)
#     tlist=[item[0] for item in topics]
#     # getSentimentsForTopic(tlist[0],sm)
#     jointPlotOfSentiment(tlist[0],sm, "Positive")
#     return



#%%
def runSentiment (allDict, sm):
    allDict=loadAllFeedsFromFile()
    sm=smallDict(allDict,2000)
    conductSentimentAnalysis(sm)
    docl=getDocList(sm, reloaddocs=False,stop_list=getCustomStopWords())
    topics= deriveTopicMaps(docl, maxNum=30, ngram_range=(3,3))
    updateDictionaryByFuzzyRelevanceofTopics(topics,sm, limit=None, threshold=20, remove=True)
    tlist=[item[0] for item in topics]
    top=tlist[4]
    df2=getSentimentsForTopic3(top,sm)
    plotSentiment3D(df2, sm, notebook=False, topic=top)
    return
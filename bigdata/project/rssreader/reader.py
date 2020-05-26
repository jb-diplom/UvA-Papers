# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:40:42 2020

@author: Janice
"""


#%% Imports

import feedparser 
import ipywidgets as widgets
from IPython import display
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt, mpld3

import datetime
from dateutil.parser import parse
from bs4 import BeautifulSoup
import pickle
import glob
import collections
import time
from collections import Counter, defaultdict
import os

from tqdm.notebook import trange, tqdm
# from tqdm import trange, tqdm
# from tqdm.auto import tqdm, trange

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import importlib
# importlib.import_module("topicmap")
# from topicmap import removeStopWords
# from topicmap import populateTopicMatrix

#%% outputSummary

def outputSummary(theDict):
    """
    Displays a simple Pandas table with the given dictionary

    Parameters
    ----------
    theDict : dict
        A dictionary with keys as the headings and lists as the contents
        of the columns

    Returns
    -------
    the Widget Hbox

    """
    
    pd.set_option('display.max_rows', 40)
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.width', 100)
    pd.set_option('display.max_colwidth', 80)
    # sample data
    df1 = pd.DataFrame(theDict)
    
    # create output widgets
    widget1 = widgets.Output()
    
    # render in output widgets
    with widget1:
        display.display(df1)
    
    # create HBox
    hbox = widgets.HBox([widget1])
    
    # render hbox
    hbox
    return hbox

#%% getHTMLParts
    
def getHTMLParts(content, part):
    """
    Return a list of the specified parts of the given HTML content

    Parameters
    ----------
    content : HTML string
        HTML string.
    part : string
        name of HTML tag to search for e.g. 'a'

    Returns
    -------
    allParts : list 
        List of strings
    """
    soup = BeautifulSoup(content, 'html.parser')
    allParts=soup.find_all(part)
    return allParts
#%% getStringContents
    
def getStringContents(content):
    soup = BeautifulSoup(content, 'html.parser')
    fullStr=""
    for str in soup.stripped_strings:
        fullStr = fullStr +str
    return fullStr

#%% savePickle

def savePickle(dictOut):
    now=datetime.datetime.now()
    runtimeStr=now.strftime("%d%m%Y_%H%M%S")    # for saving datafiles uniquely
    outfileName="rssreader/data/feed" + runtimeStr + ".pickle"
    with open(outfileName, 'wb') as outfile:
        pickle.dump(dictOut, outfile, pickle.HIGHEST_PROTOCOL)
        
    return outfileName

#%% loadPickleArticles

def loadPickleArticles(fileName):
    with open(fileName, 'rb') as infile:
        dictIn= pickle.load(infile)
    remove=[]
    for key, val in dictIn.items():
        if "cnn.com" in key:
            remove.append(key)
    for key in remove:
        dictIn.pop(key, None)
    return dictIn

#%% summarizeItems
def getFeedDict():
    myfeeds= {
    "NY Times" : "https://www.nytimes.com/svc/collections/v1/publish/https://www.nytimes.com/section/world/rss.xml",
    "Buzzfeed" : "https://www.buzzfeed.com/world.xml",
    "Al Jazeera" : "http://www.aljazeera.com/xml/rss/all.xml",
    "Defence Blog" : "http://defence-blog.com/feed",
    "Global Issues" : "http://www.globalissues.org/news/feed",
    "The Cifer Brief" : "https://www.thecipherbrief.com/feed",
    "Yahoo" : "https://www.yahoo.com/news/world/rss",
    "Times of India" : "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms",
    "The Guardian" : "https://www.theguardian.com/world/rss",
    "CNBC" : "https://www.cnbc.com/id/100727362/device/rss/rss.html",
    "RT" : "https://www.rt.com/rss/news/",
    "Reuters" : "http://feeds.reuters.com/Reuters/worldNews",
    "Der Spiegel" : "https://www.spiegel.de/international/index.rss",
    "Vox" : "https://www.vox.com/rss/world/index.xml",
    "Time Magazine" : "https://time.com/feed",
    "UN" : "https://news.un.org/feed/subscribe/en/news/all/rss.xml",
    "BBC News" : "http://feeds.bbci.co.uk/news/rss.xml",
    "The Independent" : "http://www.independent.co.uk/news/world/rss",
    "The Sun" : "https://www.thesun.co.uk/news/worldnews/feed/",
    "South China Morning Post" : "https://www.scmp.com/rss/91/feed",
    "State Department" : "https://www.state.gov/rss-feed/press-releases/feed/",
    "Christian Science Monitor" : "https://rss.csmonitor.com/feeds/world",
    "PRI": "https://www.pri.org/stories/feed/everything",
    "Eastern Herald" : "https://www.easternherald.com/feed/",
    "New Europe": "https://www.neweurope.eu/category/world/feed/",
    "News Blaze": "https://newsblaze.com/feed/",
    "Small Wars":"https://smallwarsjournal.com/rss/blogs",
    "Headline Code" : "https://headlinecode.com/feed/",
    "Article IFY":"https://articleify.com/feed/",
    "Times in Plain English" : "https://www.thetimesinplainenglish.com/feed/",
    "World Weekly News":"https://worldweeklynews.com/feed/",
    "Annals Hub" : "https://annalshub.com/feed/",
    "Z6 Mag" : "https://z6mag.com/feed/",
    "HGS Media Plus" : "https://hgsmediaplus.com.ng/feed/",
    "The Next Hint": "https://www.thenexthint.com/feed/",
    "Africa Launch Pad" : "https://africalaunchpad.com/feed/",
    "International Security Journal" : "https://internationalsecurityjournal.com/feed/",
    "NewsZetu":"https://newszetu.com/feed/",
    "IT World" : "http://itworld.blog/feed/",
    "Baltic World" : "https://balticword.eu/feed/",
    "World Affairs Journal" : "http://www.worldaffairsjournal.org/feed/",
    "Zemo City" : "https://zemocity.com/feed/",
    "Newzit News" : "https://newzitnews.com/feed",
    "Patriot Rising":"https://patriotrising.com/feed/",
    "Daily CN News" : "https://dailycnnews.com/feed/",
    "World United News" : "http://worldunitednews.blogspot.com/feeds/posts/default",
    "Nedu Wealth" : "https://www.neduwealth.com/feeds/posts/default",
    "Vivian Violine's blog" : "https://www.blogger.com/feeds/2886832199291333748/posts/default",
    "Global Vision UK":"https://globalvisionuk.com/feed/",
    "Just World News" : "https://justworldnews.org/feed/"
    }
    return myfeeds
#%% collectArticles

def collectArticles(myfeeds=getFeedDict()):
    
    # Primary source of feeds https://blog.feedspot.com/world_news_rss_feeds/
    # TODO put feed data in a separate configurable dictionary

    
    allFeeds={}
    # The critical collection of all articles
    # TODO tries=collections.defaultdict(lambda : None)
    allEntries={}
    # bar = trange(len(myfeeds.items()))

    # for feedName, feedURL in myfeeds.items():
    for feedName, feedURL in tqdm(myfeeds.items(),desc="Loading feed data"):
        feed = feedparser.parse(feedURL)
        allFeeds[feedURL]=feed
        feed.entries = enhanceEntries(feed.entries, feed.href, feedName)
        addEntries(feed.entries, allEntries)
        # if hasattr(feed , "entries") and hasattr(feed.entries[0] , "content"):
        #     print (f"{feedName: >30}Content Loaded in: {toc - tic:0.4f} seconds")
        # else:
        # tqdm.write(f"{feedName: >30}Summary Loaded in: {toc - tic:0.4f} seconds")
        tqdm.write(feedName)

    # populates collatedContents and removes any RSS-Entries with no contents
    # or summary detail or spurious content
    collateDocContents(allEntries)
    return savePickle(allEntries)
#%%
def summarizeItems(dict1=getFeedDict()):
    """
    Takes dictionary of RSS Items per media outlet and returns a simple 
    panda table of the contents

    Parameters
    ----------
    dict1 : dict
        DESCRIPTION key values of Feed Names and FeedParserDicts

    Returns
    -------
    pa_table : Panda Table
        DESCRIPTION. With three columns Source | Title | Content-Type

    """
    storyTitle=[]
    feedNames=[]
    contentType=[]
    
    for uid, val  in dict1.items():
        # print("processing", uid)
        if hasattr(val , "content"):
            contentType.append("Content")
            storyTitle.append(val.title)
            feedNames.append(val.feed_name)
        elif hasattr(val , "summary_detail"):
            contentType.append("Summary")
            storyTitle.append(val.title)
            feedNames.append(val.feed_name)
            
    # print(len(feedNames), len(storyTitle), len(contentType))
    outDict={"Source":feedNames, "Title":storyTitle, "Content":contentType}
    pa_table=outputSummary(outDict)    
    return pa_table
#%%
def summarizeFeeds(dict1=getFeedDict()):
    """
    Takes dictionary of feeds and URLs, and displays them in a pandas table

    Parameters
    ----------
    dict1 : dict
        DESCRIPTION key values of Feed Names and FeedParserDicts

    Returns
    -------
    pa_table : Panda Table
        DESCRIPTION. With three columns Source | Title | Content-Type

    """
    feedNames=[]
    urls=[]
    
    for key, val  in sorted(dict1.items()):
        feedNames.append(key)
        urls.append(val)

    theDict={"Source":feedNames, "Feed":urls}
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', 40)
    pd.set_option('display.width', 160)
    pd.set_option('display.max_colwidth', -1)
    pd.set_option('display.expand_frame_repr', False)

    # sample data
    df = pd.DataFrame(theDict)
    display.display(df)
    return

#%% summarizeByDate

def summarizeByDate(dict1):
    """
    Takes dictionary of RSS Items per media outlet and returns a Seaborn 
    swarmplot/violinplot grouped by dates

    Parameters
    ----------
    dict1 : dict
        DESCRIPTION key values of Feed Names and FeedParserDicts

    Returns
    -------
    swarm : Swarmplot
        DESCRIPTION. Columns are days, colours are per Source

    """
    articleDate=[]
    articleSize=[]
    feedNames=[]
    sns.set(rc={'figure.figsize':(14,17)})

    for uid, val  in tqdm(dict1.items(), desc="Summarizing by date"):
        aDate=getRssArticleDate(val)
        if aDate!=None:
            articleDate.append(aDate)
            feedNames.append(val.feed_name)
# TODO add actual content etc for tooltip (in val.collatedContents)
# TODO add tags and/or topics to use instead of feedname in swarmplot
        articleSize.append(getRssArticleSize(val))
            
    outDict={"Source":feedNames, "Article Size (words)":articleSize, "Date":articleDate, "labels":feedNames}
    df = pd.DataFrame(outDict)
    df = df.sort_values('Date',ascending=True).reset_index()
    paired50=sns.color_palette("hls", n_colors=50)
    # strip =sns.stripplot (data=df, x="Date", y ="Article Size (words)", hue="Source", jitter = 0.25,  orient = "h" )
    # swarm=sns.violinplot(y="Date", x="Article Size (words)", fontsize= "12", hue="Source", data=df, vert=False, width=20, height=40, aspect= 40)
    cmap = sns.cubehelix_palette (50, dark = .3, light=.8,start=0.9, rot=-1.0,gamma=0.8, as_cmap=False)
    swarm=sns.boxplot(y="Date", x="Article Size (words)", hue="Source", data=df, width=20, palette=paired50)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize= "16",ncol=1)

    #Tooltips
    fig = plt.gcf()
    tooltip = mpld3.plugins.PointLabelTooltip(fig, labels=list(outDict["labels"]))
    mpld3.plugins.connect(fig, tooltip)


    #swarm=sns.catplot(x="Date", y="Article Size (words)", hue="Source", orient = "h", kind="swarm", data=df, height=4, aspect= 1.5);
   
    #sns.set_yticklabels(sns.get_yticklabels(), fontsize=7)
    # swarm=sns.catplot(x="Date", y="Article Size (words)", hue="Source", kind="swarm", data=df);
    plt.xticks(rotation = 45, horizontalalignment="right" )
    plt.tight_layout()

    return
#%%
def appendRssArticleDateAndSize(rssentry):
    if not (bool(rssentry.get("articleDate"))):
        if hasattr(rssentry , "published"):
            dt=parse(rssentry.published, ignoretz=True)
        else:
            dt=parse(rssentry.updated, ignoretz=True)
        
        dt=dt.strftime('%d %m %Y')
        rssentry["articleDate"]=dt

    if not (bool(rssentry.get("articleSize"))):
        rssentry["articleSize"]=len(rssentry.collatedContents.split())
    return

def getRssArticleDate(rssentry):
    appendRssArticleDateAndSize(rssentry)
    return rssentry["articleDate"]

def getRssArticleSize(rssentry):
    appendRssArticleDateAndSize(rssentry)
    return rssentry["articleSize"]

#%% getNoPub

def getNoPub(dict1):

    for uid, val  in dict1.items():
        # print("processing", uid)
        if not hasattr(val , "published") and not hasattr(val , "updated") :
            outval=val
            break
    return outval

#%% dateOlderThan

def dateOlderThan(date_from, num_days=0):
    start_of_project=parse("03 May 2020 00:00:00", ignoretz=True)
    time_between_insertion = start_of_project - date_from

    return time_between_insertion.days>num_days 
#%% enhanceEntries

def enhanceEntries(entriesList, feedId, feedName):
    """
    Add Id of feed to each entry so that we only need the item, which then  
    contains all information that we need

    Parameters
    ----------
    entriesList : list
        A List of RSSEntries (FeedParserDicts)
    feedId : string
        The URL of the source feed
    feedName : string
        The clear text name of the source

    Returns
    -------
    entriesList : dict
        The enhanced entriesList

    """
    for entry in entriesList:
        entry["source"]=feedId
        entry["feed_name"]=feedName
        
    return entriesList

#%% addEntries

def addEntries(entriesList, allEntries):
    for entry in entriesList:
        allEntries[articleId(entry)]=entry
        
    return 

 
#%% articleId
# return the unique Id of the given feed item in FeedParserDict form
        
def articleId(feedParserDict):
    if feedParserDict.has_key("id"):
        return feedParserDict["id"]
    else:
        return feedParserDict["link"]   # Just for the NY Times :-()

#%% loadAllFeedsFromFile
    # Load all pickle files found in given relative directrory and 
    # merge to one dictionary of unique items

def loadAllFeedsFromFile(path = "rssreader/data" ): #this is probably a stupid place for the data long term
    # allDict=collections.defaultdict(lambda : None)
    allDict={}
    for file in glob.glob(path + "/*.pickle"):
        print ("loading file: ", file)
        dict1=loadPickleArticles(file)
        allDict.update(dict1)       # Merge all loaded items

    # summarizeItems(allDict)
    collateDocContents(allDict)
    return allDict
    
#%% getSampleDocs
# TODO getSampleDocs should be removed everywhere - it's just a test function'
# presumably replace with getDocList
def getSampleDocs(num = 40):
    allEntryDict=loadAllFeedsFromFile()
    docs=[]
    i=0 # use to break out at num
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
        docs.append(val.title +" " + getStringContents(html))
        if i > num :
            break
            
    return docs 

#%% collateDocContents

def collateDocContents(allEntryDict, deleteBadEntries=True):
    """
    Content or summary_detail are extracted from the entry, cleaned from
    HTML-tags and saved to item with the key collatedContents in each entry.
    If deleteBadEntries is specified, then additionally all entries without 
    Content or summary_detail are removed from the supplied allEntryDict, as 
    are legacy articles older than the start of project (see: dateOlderThan 
    function). 
    Remove NYT daily briefings, which skew topics/clustering like crazy whilst
    having almost no content

    Parameters
    ----------
    allEntryDict : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    toBeRemoved=[]
    items=allEntryDict.items()# help tqdm
    for key, val in tqdm(items, desc="Collating Contents"):
        html=""
        if hasattr(val , "content"):
            for line in val.content:
                html = html + " " + line.value
        elif hasattr(val , "summary_detail"):
            html = val.summary_detail.value
        else:
            toBeRemoved.append(key)
            continue
        
        val["collatedContents"]=val.title +" " + getStringContents(html)

#       Remove legacy articles    
        dt=None
        try:
            if hasattr(val , "updated"):
                    dt=parse(val.updated, ignoretz=True)
            else:
                dt=parse(val.published, ignoretz=True)
            if dateOlderThan(dt):
                toBeRemoved.append(key)  
        except:
            toBeRemoved.append(key)

        # Remove NYT daily briefings, which skew topics/clustering like crazy
        if val.feed_name == "NY Times" and "you need to know" in val.collatedContents:
            toBeRemoved.append(key)

    if deleteBadEntries:    # remove all old entries or those without contents
        for gone in tqdm(toBeRemoved, desc="Deleting bad entries"):
            allEntryDict.pop(gone)   
         
    return  

#%% getAllTags

def getAllTags(allDocDict, reload=False, displayAmount=30):
    
    if reload :
        allDocDict = loadAllFeedsFromFile()
    
    tags=[]
    nwith=0
    nwithout=0
    for key, val in allDocDict.items():
        if hasattr(val , "tags"):
            nwith +=1
            for tagItem in val["tags"]:
                tags.append(tagItem["term"])
        else:
            nwithout +=1
            
    print("\nThere were", nwith, "articles with tags and", nwithout, "without tags")
    c_tags=Counter(tags)
    # print("="*90, "\nThese are the",displayAmount,"most frequent tags used:\n","="*90,"\n",c_tags.most_common(displayAmount))

    return tags

#%% Test code for collecting and loading RSS Feed Data
    
# collectArticles()
# allDict=loadAllFeedsFromFile()
# summarizeItems(allDict) # Output panda Table summarizing all articles
# swarm=summarizeByDate(allDict) # TODO do we need all the feeds shown on the swarm plot?
    
# displayTopicsAndFeeds(allDict)
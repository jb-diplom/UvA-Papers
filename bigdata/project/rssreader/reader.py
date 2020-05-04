# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:40:42 2020

@author: Janice
"""


#%%
import feedparser 
import ipywidgets as widgets
from IPython import display
import pandas as pd
import datetime
from bs4 import BeautifulSoup
import pickle
import glob


#%%

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
    # pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 40)
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

#%%
    
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
#%%
    
def getStringContents(content):
    soup = BeautifulSoup(content, 'html.parser')
    fullStr=""
    for str in soup.stripped_strings:
        fullStr = fullStr +str
    return fullStr

#%%
def savePickle(dictOut):
    now=datetime.datetime.now()
    runtimeStr=now.strftime("%d%m%Y_%H%M%S")    # for saving datafiles uniquely
    outfileName="./data/feed" + runtimeStr + ".pickle"
    with open(outfileName, 'wb') as outfile:
        pickle.dump(dictOut, outfile, pickle.HIGHEST_PROTOCOL)
        
    return outfileName

#%%

def loadPickleArticles(fileName):
    with open(fileName, 'rb') as infile:
        dictIn= pickle.load(infile)
        
    return dictIn

#%%
def collectArticles():
    
    # Primary source of feeds https://blog.feedspot.com/world_news_rss_feeds/
    
    myfeeds= {
    "NY Times" : "https://www.nytimes.com/svc/collections/v1/publish/https://www.nytimes.com/section/world/rss.xml",
    "Buzzfeed" : "https://www.buzzfeed.com/world.xml",
    "Al Jazeera" : "http://www.aljazeera.com/xml/rss/all.xml",
    "Defence Blog" : "http://defence-blog.com/feed",
    "Global Issues" : "http://www.globalissues.org/news/feed",
    "The Cifer Brief" : "https://www.thecipherbrief.com/feed",
    "Yahoo" : "https://www.yahoo.com/news/world/rss",
    "CNN" : "http://rss.cnn.com/rss/edition_world.rss",
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
    "Digital.Alive Worlkd": "https://digitalive.world/feed/",
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
    
    allFeeds={}
    allEntries={}    # The critical collection of all articles
    
    for feedName, feedURL in myfeeds.items():
        feed = feedparser.parse(feedURL)
        allFeeds[feedURL]=feed
        feed.entries = enhanceEntries(feed.entries, feed.href, feedName)
        feed.entries = addEntries(feed.entries, allEntries)
        if hasattr(feed, "entries") and hasattr(feed.entries[0] , "content"):
            print (feedName, "has Content")
    
    return savePickle(allEntries)
 
#%%

def summarizeItems(dict1):
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
    outputSummary(outDict)    
    
    # Example of HTML parsing
    first = next(iter(dict1.values()))
    
    if hasattr(first , "content"):
        htm=first.content[0]["value"]
    else:
        htm=first.summary_detail.value
        
    return htm

#%%

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

#%%

def addEntries(entriesList, allEntries):
    for entry in entriesList:
        allEntries[articleId(entry)]=entry
        
    return entriesList

 
#%%
# return the unique Id of the given feed item in FeedParserDict form
        
def articleId(feedParserDict):
    if feedParserDict.has_key("id"):
        return feedParserDict["id"]
    else:
        return feedParserDict["link"]   # Just for the NY Times :-()

#%%
    # Load all pickle files found in given relative directrory and 
    # merge to one dictionary of unique items

def loadAllFeedsFromFile(path = "./data" ):
    # os.chdir(path)
    allDict={}
    for file in glob.glob("./data/*.pickle"):
        print ("loading file: ", file)
        dict1=loadPickleArticles(file)
        allDict.update(dict1)       # Merge all loaded items

    summarizeItems(allDict)
    return allDict
    
#%% Test code for collecting and loading RSS Feed Data
    
# collectArticles()
# loadAllFeedsFromFile()
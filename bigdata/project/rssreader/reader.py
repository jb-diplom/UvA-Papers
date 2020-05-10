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
from collections import Counter
import re
import time

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
    pd.set_option('display.width', 80)
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
    return hbox

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
    # TODO put feed data in a separate configurable dictionary
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
    # "Digital.Alive World": "https://digitalive.world/feed/",
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
        tic = time.perf_counter()   # start timing
        feed = feedparser.parse(feedURL)
        allFeeds[feedURL]=feed
        feed.entries = enhanceEntries(feed.entries, feed.href, feedName)
        addEntries(feed.entries, allEntries)
        toc = time.perf_counter()   # end timing
        if hasattr(feed.entries[0] , "content"):
            print (f"{feedName: >30}Content Loaded in: {toc - tic:0.4f} seconds")
        else:
            print (f"{feedName: >30}Summary Loaded in: {toc - tic:0.4f} seconds")
                   
    return savePickle(allEntries)
 
#%%

def summarizeItems(dict1):
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
    
    # Example of HTML parsing
    # first = next(iter(dict1.values()))
    
    # if hasattr(first , "content"):
    #     htm=first.content[0]["value"]
    # else:
    #     htm=first.summary_detail.value
        
    return pa_table

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
        
    return 

 
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

def loadAllFeedsFromFile(path = "../rssreader/data" ): #this is probably a stupid place for the data long term
    # os.chdir(path)
    allDict={}
    for file in glob.glob(path + "/*.pickle"):
        print ("loading file: ", file)
        dict1=loadPickleArticles(file)
        allDict.update(dict1)       # Merge all loaded items

    # summarizeItems(allDict)
    return allDict
    
#%%
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

#%%
def getDocList(allEntryDict=None, limit = None, reloaddocs= False, stop_list=None):
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

    Returns
    -------
    docs : TYPE
        DESCRIPTION.

    """
    
    if reloaddocs or not allEntryDict:
        allEntryDict=loadAllFeedsFromFile()
        
    docs=[]
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
        if stop_list :   #substitute all phrases for '' case insensitive
            for remove in stop_list: 
                redata = re.compile(re.escape(remove), re.IGNORECASE)
                finalVal = redata.sub('', finalVal)
        docs.append(finalVal)
        if limit and i > limit :
            break
            
    return docs 
            
#%%

def getAllTags(allDocDict, reload=False):
    
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
            
    print("="*90,  "\nThere were", nwith, "items with tags and", nwithout, "without tags")
    c_tags=Counter(tags)
    print("="*90, "\nThese are the 10 most frequent tags used:\n","="*90,"\n",c_tags.most_common(20))

# ========================================================================================== 
# There were 1856 items with tags and 3912 without tags)
# ========================================================================================== 
# These are the 20 most frequent tags used:
#  ========================================================================================== 
#  [('News', 343), ('Coronavirus', 212), ('World News', 206), ('worldNews', 190), ('World', 145), 
# ('World news', 120), ('Coronavirus outbreak', 119), ('Uncategorized', 112), ('Business', 91), 
# ('Sports', 88), ('COVID-19', 87), ('Health', 81), ('National News', 78), ('Covid-19 Pandemic', 72), 
# ('China', 71), ('Europe', 67), ('USA', 57), ('Society', 57), ('Baltic states', 57), ('US news', 54)]
# len(set(tags)) --> 2797 unique tags

    return tags
            
#%% Test code for collecting and loading RSS Feed Data
    
# collectArticles()
# allDict=loadAllFeedsFromFile()
# summarizeItems(allDict) # Output panda Table summarizing all articles
    

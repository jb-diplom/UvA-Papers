# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:44:15 2020

@author: Janice
"""

#%% open and read in data and take a look at the first 10 lines
import json
with open ("takehome60.json", mode="r", encoding= "utf-8") as file:
    data = json.load(file)
 
data_short= data[:10]

separator="\n------------------------------------------------------------------------------------------------"
#%% print titles and teasers 
titles = []
teasers = []
for element in data:
    titles.append(element["_source"]["title_rss"])
    teasers.append (element["_source"]["teaser_rss"])

print(separator, "\nA sample of the titles\n", separator)
for each_title in  titles[:10]:
        print (each_title)

print("\n",separator, "\nA sample of the teasers\n", separator)
for each_teaser in  teasers[:10]:
        print (each_teaser)
        
#%% frequency of words (excluding stop words) in titles and teasers

import spacy
sp = spacy.load('en_core_web_sm')
all_stopwords = sp.Defaults.stop_words

from collections import Counter

allwords_titles=[]

#TODO: Should really remove superfluous punctuation marks from values
for value in titles:
    tokenized=[val.lower() for val in value.split()]
    allwords_titles.extend([word for word in tokenized if not word in all_stopwords])
    
c_titles=Counter(allwords_titles)
print(separator,"\nThese are the 10 most frequent words used in titles:\n", 
      separator,"\n",c_titles.most_common(10))

allwords_teasers = []
for value in teasers:
    tokenized=[val.lower() for val in value.split()]
    allwords_teasers.extend([word for word in tokenized if not word in all_stopwords])
    
c_teaser=Counter(allwords_teasers)
print(separator,"\nThese are the most frequent words used in teasers:", 
      separator, c_teaser.most_common(10))



#%% Visualization of Frequent Words in Titles in a Histogram
import pandas as pd
import pandas as pd2
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20,7))

# Divide the figure into a 2x1 grid, and give me the first section
ax1 = fig.add_subplot(211)


top10_titles = c_titles.most_common(10)
x_titles=[]
y_titles=[]
for bars in top10_titles:
    x_titles.append(bars[0])
    y_titles.append(bars[1])

pd.Series(y_titles,x_titles).plot(kind="barh", 
                                  ylim=tuple([0,max(y_titles)*1.1]), 
                                  title = "Frequency Counter of Words in Titles")

top10_teasers = c_teaser.most_common(10)
# Divide the figure into a 2x1 grid, and give me the second section

ax2 = fig.add_subplot(212)
x_teaser=[]
y_teaser=[]
for bars_teaser in top10_teasers:
    x_teaser.append(bars_teaser[0])
    y_teaser.append(bars_teaser[1])

pd2.Series(y_teaser,x_teaser).plot(kind="barh",
                                   ylim=tuple([0,max(y_teaser)*1.1]),
                                   title = "Frequency Counter of Words in Teasers")

fig.subplots_adjust(hspace=0.5, left=0.5) # do NOT ask how I chose these spacings :-)
fig

#%% Average lengths of the titles and teasers

#Titles
print(separator, "\nSome Statistics on the titles and teasers", separator)

length_title=[]
for characters in titles:
    length_title.append(len(characters))

print("The number of titles is\t\t", len (length_title))
print ("Average length of titles is\t", sum(length_title)/len(length_title))

#Teasers
length_teaser=[]
for words in teasers:
    length_teaser.append(len(words))
print("The number of teasers is\t", len (length_teaser))
print ("Average length of teasers is\t", sum(length_teaser)/len(length_teaser))

  
#%% publication dates (by month and year) --> histogram on number of publications per month
    
dates = []
for publicationdate in data:
    dates.append((publicationdate["_source"]["publication_date"])[:10])
    
dates_short = dates[:10]
print (separator, "\nThese are the publication dates of the articles:", separator)
dateCounter = Counter(dates)
topDates = dateCounter.most_common(10)

for dates in range(10):
    print ( "\t", 
           dates+1,"\t",  
           topDates[dates][0], 
           "Number of Dates:\t",  
           topDates[dates][1])

    
#%% URL's taken apart to get main and sub-subject e.g sport, news and hockey.
    

url_total = []
main_topics = []
sub_topics = []
for url in data:
    thisURL=url["_id"]
    url_total.append(url["_id"])
    sections=thisURL.split("/")
    main_topics.append(sections[3])
    sub_topics.extend(sections[4:])
    
topicCounter=Counter(main_topics)
topTopics=topicCounter.most_common(10)
print ( separator,"\nThese are the most common topics:",separator)
for main_topics in range(2):
    print ( "\t", 
           main_topics+1,  
           topTopics[main_topics][0], 
           "Number of different Topics:\t\t",  
           topTopics[main_topics][1])

print ( separator,"\nThese are the most common subtopics:",separator)
subtopicCounter=Counter(sub_topics)
topSubtopics=subtopicCounter.most_common(10)
for sub_topics in range(10):
    print ( "\t", 
           sub_topics+1,  
           topSubtopics[sub_topics][0], 
           "Number of different Subtopics:\t\t",  
           topSubtopics[sub_topics][1])    
    
#%% Authors and the Frequency of them
author = []
authorDict={}

for element in data:
    try:        #since not all elements have a byline, try for a byline and if not there then jump over
        bline=element["_source"]["byline"]
        auth=bline[3:]           
        authorDict[element["_id"]]=auth
    except:
        pass

authCounter=Counter(authorDict.values())
topAuthors=authCounter.most_common(10)

print ( separator,"\nThese are the most prolific authors:",separator)
for journalist in range(10):
    print ( "\t", 
           journalist+1,  
           topAuthors[journalist][0], 
           "Number of Articles:",  
           topAuthors[journalist][1])

#%% type of data
data_type = {}
keys=[]
types=[]

for element in data:
    for key,value in element.items():
            if not(key in keys):
                keys.append(key)
                types.append(type(value).__name__)        
                if isinstance(value,dict):
                    for key1, val1 in value.items():
                        if not(key1 in keys):
                            keys.append(str(key)+"."+ str(key1))
                            types.append(type(val1).__name__)        
                        if isinstance(val1,dict):
                            for key2, val2 in val1.items():
                                if not(key2 in keys):
                                    keys.append(str(key)+"."+ str(key1)+"."+ str(key2))
                                    types.append (type(val2).__name__)
                            
import ipywidgets as widgets
from IPython import display
import pandas as pd

# sample data
df1 = pd.DataFrame({"Item": keys, "Type":types  })
widget1 = widgets.Output()

# render in output widgets
with widget1:
    display.display(df1)

# create HBox
hbox = widgets.HBox([widget1])

# render hbox
hbox
#%%
#potential part: list out number of potential topics
    
   
    
    #TODO:add the type of the data and put all into jupyter with nice visualisation
    
 #%% use pandas to create styled table with all important info
    # e.g. https://towardsdatascience.com/style-pandas-dataframe-like-a-master-6b02bf6468b0
# =============================================================================
#     
#     collect data in dictionaries using the primary (_id) key to crossreference any data that is there.
#     e.g which authors produce how many articles for the different subtopcis and topics
#     collect data for different news sites
#     could do a sentiment analysis on the content body of the text or just the titles/leads
#     could do a pattern analysis of the titles to group using unsupervised learning to derive which groupings of articles
#     there are and their relation to each other
#     
#     might also be an idea to crossreference the content (and it's metedata) with 
#     statistics from a web-tracking tool (e.g. Google Analytics)
#     
#     create a clear research question and write text for the third analysis plan part
# 
# =============================================================================

#%%
# import numpy as np
# import pandas as pd
# from IPython.core.display import display, HTML

# data1 = np.array([['','Col1','Col2'],    ['Row1',1,2],   ['Row2',3,4]])
                
# df1=pd.DataFrame(data=data1[1:,1:], index=data1[1:,0],  columns=data1[0,1:])
# display(HTML(df1.to_html()))
# print(df1.to_html())

import ipywidgets as widgets
from IPython import display
import pandas as pd
import numpy as np

# sample data
df1 = pd.DataFrame(np.random.randn(8, 3))
df2 = pd.DataFrame(np.random.randn(8, 3))

# create output widgets
widget1 = widgets.Output()
widget2 = widgets.Output()

# render in output widgets
with widget1:
    display.display(df1)
with widget2:
    display.display(df2)

# create HBox
hbox = widgets.HBox([widget1, widget2])

# render hbox
hbox
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 18:24:51 2020

@author: Daddy
"""
numFreqTags = 6
numTitles = 6
seper="\n----------------------------------------------------------------"

import json
from collections import Counter
import statistics

#dataset="C:/Users/Janice/Documents/Big_Data_and_Content_Analysis/pornexercise/testdata2.json"
dataset="C:/Users/Janice/Documents/Big_Data_and_Content_Analysis/pornexercise/xhamster.json"

with open(dataset, mode="r", encoding="utf-8") as fi:
    mydict = json.load(fi)

allchan = []
alldesc_words= []
alldesc_len = []
tagdict={}
print ( seper, "\n","Titles were:", seper)
i=0
for key, value in (mydict.items()):
    if i < numTitles:
        print (value['title'])          # print some titles as demo
    allchan.extend(value['channels'])   # collect all channels
    alldesc_len.append(len(value['description']))
    alldesc_words.extend(value['description'].split())
    i +=1
    for chan in value['channels']:      # count comments/votes/views per channel
        if chan in tagdict:
            tagdict[chan]["views"]=tagdict[chan]["views"] + value['nb_views']
            tagdict[chan]["comments"]=tagdict[chan]["comments"] + value['nb_comments']
            tagdict[chan]["votes"]=tagdict[chan]["votes"] + value['nb_votes']
        else:
            tagdict[chan]={"views": value['nb_views'], 
                           "comments":value['nb_comments'], 
                           "votes":value['nb_votes'] 
                           }
            
c = Counter(allchan)
maxChannels = c.most_common(numFreqTags)

c2 = Counter(alldesc_words)
freqDescWords = c2.most_common(numFreqTags)

print ( seper,"\nFrequent channels were:",seper)
for commomTags in range(numFreqTags-1):
    print ( "\t", commomTags+1,  maxChannels[commomTags][0], "\tFrequency:",  maxChannels[commomTags][1])

print ( seper,"\nFrequent description words were:",seper)
for commomWords in range(numFreqTags-1):
    print ( "\t", commomWords+1,  freqDescWords[commomWords][0], "\tFrequency:",  freqDescWords[commomWords][1])

print (seper, "\nAverage nr. of tags is:",statistics.mean (c.values()))
print ("Average description length is:",statistics.mean(alldesc_len),seper)


allkeys=list(tagdict.keys())
most_comments=allkeys[0]
most_votes=allkeys[0]
most_views=allkeys[0]

num_views=tagdict[most_views]["views"]
num_comments=tagdict[most_comments]["comments"]
num_votes=tagdict[most_votes]["votes"]

for atag in allkeys:
    if num_views > tagdict[atag]["views"]:
       num_views =  tagdict[atag]["views"] 
       most_views = atag
    if num_comments > tagdict[atag]["comments"]:
       num_comments =  tagdict[atag]["comments"] 
       most_comments= atag
    if num_votes > tagdict[atag]["votes"]:
       num_votes =  tagdict[atag]["votes"] 
       most_votes = atag
       
print(seper,"\n","Most Important Tags",seper)
print("\tViews:\t\t",most_views)
print("\tComments:\t",most_comments)
print("\tVotes:\t\t",most_votes)
        
        
        
    
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:40:03 2020

@author: -
"""


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

with open(file="C:/Users/Janice/Documents/Big_Data_and_Content_Analysis/pornexercise/testdata2.json", 
          mode="r", 
          encoding="utf-8") as fi:
    mydict = json.load(fi)

allchan = []

print ( seper, "\n","Titles were:", seper)
for key, value in (mydict.items()):
    allchan.extend(value['channels'])   # collect all channels
            
c = Counter(allchan)
maxChannels = c.most_common(numFreqTags)


print ( seper,"\nFrequent channels were:",seper)
for commomTags in range(numFreqTags-1):
    print ( "\t", commomTags+1,  maxChannels[commomTags][0], "\tFrequency:",  maxChannels[commomTags][1])





        
        
        
    
    
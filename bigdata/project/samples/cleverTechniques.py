# -*- coding: utf-8 -*-
"""
Created on Tue May 12 22:11:43 2020

@author: Janice
"""


# how to remove roms from a pandas Series on a condition
# the series is 

col0=df.iloc[2]
col0=col0.drop(col0[col0 < 0.8].index)

or for a range combine with &
colnew=col0.drop(col0[(1 < col0) & (col0 < 0.8) ].index)

# how to get the label at a certain  position
col0.index[0] --> 'https://www.nytimes.com/2020/05/03/world/americas/venezuela-coup.html'

last element of a list
mylist[-1]


# declaring defaultdict 
# sets default value 'Key Not found' to absent keys 
defd = collections.defaultdict(lambda : 'Key Not found')
defd = collections.defaultdict(lambda : int)
defd[1]
Out[43]: int

# TODO Iterate (ix) through the columns from the SoftCosine matrix, delete stuff outside threshold (drop)
# add what's left to allDict using the key from col0.index[ix]

the indian pythonista
dependency üarsing?? Doing it already??
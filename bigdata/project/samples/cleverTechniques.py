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


#%% testFuzzy
def testFuzzy():
    from fuzzywuzzy import process # for Levenshtein  Distance calculations

    str2Match = "apple inc"
    strOptions = ["Apple Inc.","apple park","apple incorporated","iphone"]
    Ratios = process.extract(str2Match,strOptions)
    print(Ratios)
    # You can also select the string with the highest matching percentage
    highest = process.extractOne(str2Match,strOptions)
    print(highest)
    return

#%% smallDict utility
def smallDict(d, sample=50):
    import random
    keys = random.sample(list(d), sample)
    values = [d[k] for k in keys]
    return dict(zip(keys, values))

    #%% Using Notebook
from bokeh.io import  show, output_notebook #output_file,
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
output_notebook()
#output_file("tools_hover_tooltip_formatting.html")
    """
    do your stuff ...
    p = figure(plot_height=250, x_axis_type="datetime", tools="", toolbar_location=None,
           title="Hover Tooltip Formatting", sizing_mode="scale_width")
    """
show(p)

    #%% Using HTML
from bokeh.io import  show, ,output_file
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
output_file("myfile.html")
    """
    do your stuff ...
    p = figure(plot_height=250, x_axis_type="datetime", tools="", toolbar_location=None,
           title="Hover Tooltip Formatting", sizing_mode="scale_width")
    """
show(p)

# using DataFrames

df.index # gets the index column of a DataFrame
df.index[0] # gets the value of the first index

# Add a column
df['pca-three'] = pca_result[:,2] # in this case the second column of the

#%% Choose colours for seaborn in a notebook
import seaborn as sns
sns.choose_cubehelix_palette()
#%% Dictionaries: check if there's a value at key withjout getting a KeyError
dd={}
(bool(dd.get('key')))
Out[64]: False
dd["key"]=98
(bool(dd.get('key')))
Out[66]: True

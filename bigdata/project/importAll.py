# -*- coding: utf-8 -*-
"""
Created on Tue May 19 18:27:52 2020

@author: Janice
"""


import importlib
importlib.import_module("rssreader.reader")
importlib.import_module("topicmaps.topicmap")
importlib.import_module("Gensim.gensim_test")
importlib.import_module("samples.seabornScatterPlots")

from rssreader.reader import loadAllFeedsFromFile
from samples.seabornScatterPlots import displayTags

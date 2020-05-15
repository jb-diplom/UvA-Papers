# -*- coding: utf-8 -*-
"""
Created on Mon May 11 17:29:30 2020

@author: Janice
"""

# in Command shell do conda install -c conda-forge pyldavis 
# see also https://github.com/bmabey/pyLDAvis
import json
import numpy as np
import pyLDAvis
from bokeh.io import  show, output_notebook, output_file



def load_R_model(filename):
    with open(filename, 'r') as j:
        data_input = json.load(j)
    data = {'topic_term_dists': data_input['phi'], 
            'doc_topic_dists': data_input['theta'],
            'doc_lengths': data_input['doc.length'],
            'vocab': data_input['vocab'],
            'term_frequency': data_input['term.frequency']}
    return data

output_file("pyDAVis.html")
# output_notebook() # TODO for use in notebook
    # pyLDAvis.enable_notebook()
movies_model_data = load_R_model('data/movie_reviews_input.json')


movies_vis_data = pyLDAvis.prepare(**movies_model_data)
# p=pyLDAvis.display(movies_vis_data) # should use this in notebook
p=pyLDAvis.show(movies_vis_data) # displays in own window combined with output_file
show(p)


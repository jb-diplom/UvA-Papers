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


def load_R_model(filename):
    with open(filename, 'r') as j:
        data_input = json.load(j)
    data = {'topic_term_dists': data_input['phi'], 
            'doc_topic_dists': data_input['theta'],
            'doc_lengths': data_input['doc.length'],
            'vocab': data_input['vocab'],
            'term_frequency': data_input['term.frequency']}
    return data

movies_model_data = load_R_model('data/movie_reviews_input.json')

#%%

movies_vis_data = pyLDAvis.prepare(**movies_model_data)
pyLDAvis.display(movies_vis_data)

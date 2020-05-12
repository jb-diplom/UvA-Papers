# -*- coding: utf-8 -*-
"""
Created on Wed May  6 23:11:37 2020

@author: Janice
"""

# https://ipyvolume.readthedocs.io/en/latest/install.html

# Conda installation on the console
# conda install -c conda-forge ipyvolume
# conda install -c conda-forge nodejs  # or some other way to have a recent node
# jupyter labextension install @jupyter-widgets/jupyterlab-manager
# jupyter labextension install ipyvolume
# jupyter labextension install jupyter-threejs

# Can also activate in jupyterlab --> Settings -> Enable Extension Manager (I think)

import ipywidgets as widgets
import numpy as np
import ipyvolume as ipv


x, y, z, u, v, w = np.random.random((6, 1000))*2-1
selected = np.random.randint(0, 1000, 100)
ipv.figure()
quiver = ipv.quiver(x, y, z, u, v, w, size=5, size_selected=8, selected=selected)

from ipywidgets import FloatSlider, ColorPicker, VBox, jslink
size = FloatSlider(min=0, max=30, step=0.1)
size_selected = FloatSlider(min=0, max=30, step=0.1)
color = ColorPicker()
color_selected = ColorPicker()
jslink((quiver, 'size'), (size, 'value'))
jslink((quiver, 'size_selected'), (size_selected, 'value'))
jslink((quiver, 'color'), (color, 'value'))
jslink((quiver, 'color_selected'), (color_selected, 'value'))
VBox([ipv.gcc(), size, size_selected, color, color_selected])

#%%
# Import dependencies
import plotly
import plotly.graph_objs as go

# Configure Plotly to be rendered inline in the notebook.
plotly.offline.init_notebook_mode()

# Configure the trace.
trace = go.Scatter3d(
    x=[1, 2, 3],  # <-- Put your data instead
    y=[4, 5, 6],  # <-- Put your data instead
    z=[7, 8, 9],  # <-- Put your data instead
    mode='markers',
    marker={
        'size': 10,
        'opacity': 0.8,
    }
)

# Configure the layout.
layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
)

data = [trace]

plot_figure = go.Figure(data=data, layout=layout)

# Render the plot.
pl=plotly.offline.iplot(plot_figure)
pl
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 22:04:56 2020

@author: Janice
"""
#%% TODO if you need Bokeh test data do this
def init():
    import bokeh
    bokeh.sampledata.download()
    return
init()
#%% bokeh tooltip demo
import numpy as np

from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure

from bokeh.sampledata.stocks import AAPL

output_file("tools_hover_tooltip_formatting.html")

def datetime(x):
    return np.array(x, dtype=np.datetime64)

source = ColumnDataSource(data={
    'date'      : datetime(AAPL['date'][::10]),
    'adj close' : AAPL['adj_close'][::10],
    'volume'    : AAPL['volume'][::10],
})

p = figure(plot_height=250, x_axis_type="datetime", tools="", toolbar_location=None,
           title="Hover Tooltip Formatting", sizing_mode="scale_width")
p.background_fill_color="#f5f5f5"
p.grid.grid_line_color="white"
p.xaxis.axis_label = 'Date'
p.yaxis.axis_label = 'Price'
p.axis.axis_line_color = None

p.line(x='date', y='adj close', line_width=2, color='#ebbd5b', source=source)

p.add_tools(HoverTool(
    tooltips=[
        ( 'date',   '@date{%F}'            ),
        ( 'close',  '$@{adj close}{%0.2f}' ), # use @{ } for field names with spaces
        ( 'volume', '@volume{0.00 a}'      ),
    ],

    formatters={
        'date'      : 'datetime', # use 'datetime' formatter for 'date' field
        'adj close' : 'printf',   # use 'printf' formatter for 'adj close' field
                                  # use default 'numeral' formatter for other fields
    },

    # display a tooltip whenever the cursor is vertically in line with a glyph
    mode='vline'
))

show(p)

#%% Matplot Tooltip demo

import matplotlib.pyplot as plt,mpld3
import numpy as np; np.random.seed(1)
import seaborn as sns; sns.set()

x = np.random.rand(15)
y = np.random.rand(15)
names = np.array(list("ABCDEFGHIJKLMNO"))
c = np.random.randint(1,5,size=15)

norm = plt.Normalize(1,4)
cmap = plt.cm.RdYlGn

fig,ax = plt.subplots()
# sc = plt.scatter(x,y,c=c, s=100, cmap=cmap, norm=norm)
sc=sns.boxplot(y=x, x=y, hue=c, width=20, palette="Blues")

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):

    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
                           " ".join([names[n] for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()
#%% Matplot tooltips hack
import numpy as np
import matplotlib.pyplot as plt, mpld3
import seaborn as sns
import pandas as pd

N=10
data = pd.DataFrame({"x": np.random.randn(N),
                     "y": np.random.randn(N), 
                     "size": np.random.randint(20,200, size=N),
                     "label": np.arange(N)
                     })


scatter_sns = sns.lmplot("x", "y", 
           scatter_kws={"s": data["size"]},
           robust=False, # slow if true
           data=data, size=8)
fig = plt.gcf()

tooltip = mpld3.plugins.PointLabelTooltip(fig, labels=list(data.label))
mpld3.plugins.connect(fig, tooltip)

mpld3.display(fig)

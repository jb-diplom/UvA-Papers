# -*- coding: utf-8 -*-
"""
Created on Wed May  6 21:31:08 2020

@author: Janice
"""


import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
# tips = sns.load_dataset("tips")
# ax = sns.scatterplot(x="total_bill", y="tip", data=tips)


# # ax = sns.scatterplot(x="total_bill", y="tip", hue="time",
# #                      data=tips)
# ax = sns.scatterplot(x="total_bill", y="tip",
#                       hue="day", style="time", data=tips)


import seaborn as sns
sns.set()

# Load the example planets dataset
planets = sns.load_dataset("planets")

cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
ax = sns.scatterplot(x="distance", y="orbital_period",
                     hue="year", size="mass",
                     palette=cmap, sizes=(10, 200),
                     data=planets)

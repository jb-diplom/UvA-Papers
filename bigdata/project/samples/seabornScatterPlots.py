# -*- coding: utf-8 -*-
"""
Created on Wed May  6 21:31:08 2020

@author: Janice
"""


import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

# https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html

#%%
tips = sns.load_dataset("tips")
tips = tips.sort_values('day',ascending=False).reset_index()
ax = sns.scatterplot(x="total_bill", y="tip", data=tips)


# ax = sns.scatterplot(x="total_bill", y="tip", hue="time",
#                      data=tips)

ax = sns.scatterplot(x="total_bill", y="tip",
                      hue="day", style="time", data=tips)

sns.catplot(x="day", y="total_bill", hue="sex", kind="swarm", data=tips);
plt.xticks(rotation = 45, horizontalalignment="right" )

#%%
import seaborn as sns
sns.set()

# Load the example planets dataset
planets = sns.load_dataset("planets")

cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
ax = sns.scatterplot(x="distance", y="orbital_period",
                     hue="year", size="mass",
                     palette=cmap, sizes=(10, 200),
                     data=planets)
#%%
import seaborn as sns
#%matplotlib inline # To show embedded plots in the notebook

tips = sns.load_dataset("tips")

fig, ax = plt.subplots()
ax = sns.boxplot(tips["total_bill"])
#%%
tips = sns.load_dataset("tips")
sns.catplot(x="day", y="total_bill", data=tips);
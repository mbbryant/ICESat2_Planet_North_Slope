#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:44:38 2024

@author: m1bryant
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import cm as cmm
from cmcrameri import cm
import pdb
plt.rcParams["font.family"] = "arial"

#%%
file_path = '/Users/m1bryant/Documents/Data/IS2_case_study/retreat_rates.csv'

retreat = pd.read_csv(file_path)
retreat.sort_values(['year_1'], inplace = True)
colors_o= cm.batlow((np.linspace(0, 1, 10)))
colors = [colors_o[0], colors_o[4], colors_o[6], colors_o[8]]
data_source = list(set(retreat['source']))
data_source.sort()

plt.figure()


for i in range(0, len(data_source)):
    s = data_source[i]
    r = retreat[retreat['source'] == s]
    r.reset_index(inplace = True)
    c_i = colors[i]
    for j in range(0, len(r)):
       # pdb.set_trace()
        r_j = r.loc[j]
        if j == 0:
            plt.plot([r_j['year_1'], r_j['year_2']], [r_j['rate'], r_j['rate']], \
                     lw = 4, c = c_i, label = s)
            #print(j)
        else:
            plt.plot([r_j['year_1'], r_j['year_2']], [r_j['rate'], r_j['rate']], lw = 4,\
                     c = c_i, mec = 'k', label = None)
        
plt.xlabel('Year', fontsize = 25)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.ylabel(r'Shoreline change rate (m a$^{-1}$)', fontsize = 25)
plt.legend(title = 'Study', fontsize = 20,title_fontproperties={'weight':'bold', 'size':20})
plt.subplots_adjust(left=0.146, bottom = .108, right = 0.98, top = .99)
   
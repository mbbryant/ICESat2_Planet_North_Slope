#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 23:06:13 2023

Last updated Jan 30 2025

@author: Marnie Bryant (m1bryant@ucsd.edu)

"""

import numpy as np
import matplotlib.pyplot as plt
import pyproj
from matplotlib import cm as cmm
from cmcrameri import cm
import os
import geopandas as gpd
import pandas as pd
from shapely import wkt
import datetime as dt
import matplotlib.patheffects as pe
import scipy as sp
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from shapely.geometry import Point
plt.rcParams["font.family"] = "arial"
from PIL import Image
#%%
def intersection(m1, b1, m2, b2):
    x = -(b1-b2)/(m1-m2)
    y = m1*x + b1
    return x, y
def intersect_line_contour(line, cont):
   # 
    x_l = line[:, 0]
    y_l = line[:, 1]
    
    p = np.polyfit(x_l, y_l, 1)
    m_l = p[0]
    b_l = p[1]
    x_c = cont[:, 1]
    y_c = cont[:, 2]
    
    #subset to range of line segment

    cont_s = cont[np.where(x_c >= np.min(x_l))[0][0] - 1: np.where(x_c <= np.max(x_l))[0][-1] + 2]
    x_s = cont_s[:, 1]
    y_s = cont_s[:, 2]
    x_i = np.nan
    y_i = np.nan
    for i in range(0, len(x_s)):
        x = x_s[i:i+2]
        y = y_s[i:i+2]
        
        p2 = np.polyfit(x, y, 1)
        m_s = p2[0]
        b_s = p2[1]
        
        x_i, y_i = intersection(m_l, b_l, m_s, b_s)
        
        if ((x_i >= np.min(x)) and (x_i <= np.max(x)) and (y_i >= np.min(y)) and (y_i <= np.max(y))):
            break 
        else:
            x_i = np.nan
            y_i = np.nan
    return x_i, y_i

def project_on_t(track, t):
    xs = track.geometry.x.values
    ys = track.geometry.y.values
    x_0 = xs[-1]
    y_0 = ys[-1]
    final_dist = []
    t_vec = coords = [g.coords.xy for g in t.geometry][0]
    # pdb.set_trace()
    t_vec = [coords[0][-1] - coords[0][0], coords[1][-1]-coords[1][0]]
    angles = []
    for i in range(0, len(xs)-1):
        vec = [xs[i]-x_0, ys[i] - y_0]
        final_dist.append(np.dot(vec, t_vec)/np.linalg.norm(t_vec))
        angles.append(np.arccos(final_dist[i]/np.linalg.norm(vec)))
    mean_angle = np.nanmean(angles)
    final_dist.append(0)
    return final_dist, mean_angle
#%%


bsc_lake = pd.read_csv('data/ICESat-2/atl03_137_gt2r.csv')

bsc_lake_06_0 = pd.read_csv('data/ICESat-2/atl06_sr_137_gt2r.csv')
bsc_lake_06_p = gpd.GeoDataFrame(bsc_lake_06_0, \
                                   geometry = bsc_lake_06_0['geometry'].apply(wkt.loads),\
                                       crs = "EPSG:32605")



    
atl03_sr_lake_p = gpd.GeoDataFrame(bsc_lake, geometry = bsc_lake['geometry'].apply(wkt.loads), crs = "EPSG:4326")


bsc_bluff = pd.read_csv('data/ICESat-2/atl03_137_gt3r.csv')
bsc_bluff_06 = pd.read_csv('data/ICESat-2/atl06_sr_137_gt3r.csv')
bsc_bluff_06_p = gpd.GeoDataFrame(bsc_bluff_06, \
                                   geometry = bsc_bluff_06['geometry'].apply(wkt.loads),\
                                       crs = "EPSG:32605")


atl03_sr_bluff_p = gpd.GeoDataFrame(bsc_bluff, geometry = bsc_bluff['geometry'].apply(wkt.loads), crs = "EPSG:4326")



bsc_beach = pd.read_csv('data/ICESat-2/atl03_137_gt1r.csv')
bsc_beach_06 = pd.read_csv('data/ICESat-2/atl06_sr_137_gt1r.csv')
bsc_beach_06_p = gpd.GeoDataFrame(bsc_beach_06, \
                                   geometry = bsc_beach_06['geometry'].apply(wkt.loads),\
                                       crs = "EPSG:32605")

atl03_sr_beach_p = gpd.GeoDataFrame(bsc_beach, geometry = bsc_beach['geometry'].apply(wkt.loads), crs = "EPSG:4326")
atl03_sr_beach_p.drop(columns = ['Unnamed: 0', 'landcover', 'yapc_score', 'snowcover', \
                                'relief'], inplace = True)
#%%
image_beach = Image.open('figures/beach_google_earth_zoom_no_labels.png')

image_bluff = Image.open('figures/bluff_google_earth_zoom_no_labels.png')
image_lake = Image.open('figures/lake_google_earth_zoom_no_labels.png')
#%%coastline data from Planet Imagery
cont_2019r = np.genfromtxt("data/annual_coastlines/annual_coastline_2019_6_25_1006_resampled_r.csv", \
                           delimiter=",")
cont_2020r = np.genfromtxt("data/annual_coastlines/annual_coastline_2020_07_25_1065_resampled_r.csv",\
                           delimiter=",")
cont_2021ar = np.genfromtxt("data/annual_coastlines/annual_coastline_2021_07_02_2442_resampled_r.csv", delimiter=",")
cont_2022r = np.genfromtxt("data/annual_coastlines/annual_coastline_2022_7_01_2439_resampled_r.csv", delimiter = ",")
#%%
transects = gpd.read_file('data/transects_2021_50m.json')
is2_transect_list = []

#%%
lake_lon_labels = [r'153.747', r'153.744']
lake_lon_ticks = [235, 765]
lake_lat_labels = [r'70.885', r'70.886', r'70.887']
lake_lat_ticks = [1150, 600, 50]

bluff_lon_labels = [r'153.836', r'153.833']
bluff_lon_ticks = [170, 700]
bluff_lat_labels = [r'70.883', r'70.884', r'70.885']
bluff_lat_ticks = [1050, 550, 50]

beach_lon_labels = [r'153.657', r'153.654']
beach_lon_ticks = [280, 800]
beach_lat_labels = [r'70.882', r'70.883']
beach_lat_ticks = [840, 320]
#%%

atl03_sr = atl03_sr_lake_p
atl06 = bsc_lake_06_p
ylims = [-3, 4]
overlay = image_lake
lon_ticks = lake_lon_ticks
lat_ticks = lake_lat_ticks
lon_labels = lake_lon_labels
lat_labels = lake_lat_labels


#%%
def plots_and_boundaries(name):

    if name == 'bluff':
        atl03_sr = atl03_sr_bluff_p
        atl06 = bsc_bluff_06_p
        ylims = [-3, 4]
        overlay = image_bluff
        lon_ticks = bluff_lon_ticks
        lat_ticks = bluff_lat_ticks
        lon_labels = bluff_lon_labels
        lat_labels = bluff_lat_labels
    elif name == 'lake':
        atl03_sr = atl03_sr_lake_p
        atl06 = bsc_lake_06_p
        ylims = [-3, 4]
        overlay = image_lake
        lon_ticks = lake_lon_ticks
        lat_ticks = lake_lat_ticks
        lon_labels = lake_lon_labels
        lat_labels = lake_lat_labels
    elif name == 'beach':
        atl03_sr = atl03_sr_beach_p
        atl06 = bsc_beach_06_p
        ylims = [-3, 4]
        overlay = image_beach
        lon_ticks = beach_lon_ticks
        lat_ticks = beach_lat_ticks
        lon_labels = beach_lon_labels
        lat_labels = beach_lat_labels
    else:
        print('Invalid name')
    
    date_list = ['2019-04-07', '2020-04-07', '2021-07-02', '2021-12-31'] 
    
    track_04_19 = atl03_sr[(atl03_sr['date'] == '2019-04-07')& (atl03_sr['atl03_cnf'] >= 3)]
    track_04_19_06 = atl06[(atl06['date']== '2019-04-07')]
    track_01_20 = atl03_sr[(atl03_sr['date'] == '2020-01-04')& (atl03_sr['atl03_cnf'] >= 3)]
    track_01_20_06 = atl06[(atl06['date']== '2020-01-04')]
    track_07_21 = atl03_sr[(atl03_sr['date'] == '2021-07-02')& (atl03_sr['atl03_cnf'] >= 3)]
    track_07_21_06 = atl06[(atl06['date']== '2021-07-02')]
    track_12_21 = atl03_sr[(atl03_sr['date'] == '2021-12-31')& (atl03_sr['atl03_cnf'] >= 3)]
    track_12_21_06 = atl06[(atl06['date']== '2021-12-31')]
    
    all_x = pd.concat([track_04_19_06.geometry.x, track_01_20_06.geometry.x, track_07_21_06.geometry.x, track_12_21_06.geometry.x])
    all_y = pd.concat([track_04_19_06.geometry.y, track_01_20_06.geometry.y, track_07_21_06.geometry.y, track_12_21_06.geometry.y])
    mid_x = np.mean([np.max(all_x), np.min(all_x)])
    mid_y = np.mean([np.max(all_y), np.min(all_y)])
    
    mean_point = Point(mid_x, mid_y)
    
    t_distance = transects.geometry.distance(mean_point)
    t = transects[t_distance == np.min(t_distance)]
    coords = [g.coords.xy for g in t.geometry][0]
    t_vec = [coords[0][-1] - coords[0][0], coords[1][-1]-coords[1][0]]
    n_s_vec = [0, 1]
    angle = np.arccos((np.dot(t_vec, n_s_vec))/np.linalg.norm(t_vec))
    
    colors = cm.lajolla((np.linspace(0, 1, 8)))

    legend_colors = [Patch(facecolor='k', edgecolor = None, label = '2019'),\
                     Patch(facecolor=colors[2], edgecolor = None, label = '2020'),\
                     Patch(facecolor=colors[4], edgecolor = None, label = '2021'),\
                     Patch(facecolor=colors[6], edgecolor = None, label = '2022')    ]    
    
    y_min = np.min(track_04_19.geometry.y)
    
    fig = plt.figure(figsize  = (16.8,  7))
    
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(overlay)
    
    plt.xticks(lon_ticks, lon_labels, fontsize = 20)
    plt.xlabel('Longitude ($^\circ$ W)', fontsize = 20)
    plt.ylabel('Latitude ($^\circ$ N)', fontsize = 20)
    plt.yticks(lat_ticks, lat_labels, fontsize = 20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.plot([40, 290], [1160, 1160], color = 'white', lw = 3)
    plt.annotate('50 m', (125,1130), fontsize = 15, color = 'white')
    plt.annotate(r"""Google Earth
    $\copyright$ 2024 CNES/Airbus""",(580, 1180), fontsize = 13, color = 'white')
    plt.annotate('(e) Image date:18 September 2018 ', (0.01, .95), xycoords = 'axes fraction', \
                 bbox=dict(facecolor='w', edgecolor='none'), fontsize = 17)
    
    ax2 = fig.add_subplot(1,2,2)
    ax.legend(handles = legend_colors, bbox_to_anchor=(.5, 1), loc = 'lower center', \
              fontsize = 20, title = 'Year', ncol = 4,\
                  title_fontproperties={'weight':'bold', 'size':20},handletextpad= .2, columnspacing=0.3)
    n_s_vec = track_04_19.geometry.y - y_min
    
    dist_19_03, angle_19 = project_on_t(track_04_19, t)
    dist_19_06, angle_19 = project_on_t(track_04_19_06, t)
    
    
    
    offset_19 = track_04_19.geometry.iloc[0].distance(track_04_19_06.geometry.iloc[0])*np.cos(angle_19)
    
    track_04_19['cross_track_distance'] = dist_19_03
    track_04_19_06['cross_track_distance'] = [d+offset_19 for d in dist_19_06]
    
    atl03_19 = ax2.scatter(dist_19_03, track_04_19['height'], alpha = 0.4, color = 'k', linewidth=0, s = 40, label = '07 April 2019')#'(ATL03)'
    
    
    sr_item = ax2.plot([d+offset_19 for d in dist_19_06],track_04_19_06['h_mean'], '-o', markersize = 2, color = 'k',path_effects=[pe.Stroke(linewidth=3, foreground='gray'), pe.Normal()], label = None)#'2019-04-07 (SR)'
    
    
    
    line = np.vstack((track_04_19.geometry.x.values, track_04_19.geometry.y.values)).T
    x_6_19, y_6_19 = intersect_line_contour(line, cont_2019r)
    
    line_item = ax2.vlines((y_6_19 - y_min)*np.cos(angle), -3, 5, linestyle = '--',linewidth = 2, color = 'k', label = '2019-06-25 Coastline (Planet)')
    
    
    dist_20_03, angle_20 = project_on_t(track_01_20, t)
    dist_20_06, angle_20 = project_on_t(track_01_20_06, t)
    
    offset_20 = track_01_20.geometry.iloc[0].distance(track_01_20_06.geometry.iloc[0])*np.cos(angle_20)
    
    track_01_20['cross_track_distance'] = dist_20_03
    track_01_20_06['cross_track_distance'] = [d+offset_20 for d in dist_20_06]
    
    atl03_20 = ax2.scatter(dist_20_03, track_01_20['height'], alpha = 0.4, color = colors[2], linewidth=0,  s = 40, label = '04 January 2020')#(15.5 m)
    
    ax2.plot([d + offset_20 for d in dist_20_06],track_01_20_06['h_mean'],'-o', markersize = 2,color = colors[2],path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()], label = '2020-01-04 (SR)')
    line = np.vstack((track_01_20.geometry.x.values, track_01_20.geometry.y.values)).T
    
    x_7_20, y_7_20 = intersect_line_contour(line, cont_2020r)
    
    ax2.vlines((y_7_20 - y_min)*np.cos(angle), -3, 5, linestyle = '--',linewidth = 2, color = colors[2], label = '2020-07-25 Coastline (Planet)')
    
    
    dist_21_03, angle_21 = project_on_t(track_07_21, t)
    dist_21_06, angle_21 = project_on_t(track_07_21_06, t)
    
    offset_21 = track_07_21.geometry.iloc[0].distance(track_07_21_06.geometry.iloc[0])*np.cos(angle_21)
    
    track_07_21['cross_track_distance'] = dist_21_03
    track_07_21_06['cross_track_distance'] = [d+offset_21 for d in dist_21_06]
    
    atl03_21 = ax2.scatter(dist_21_03, track_07_21['height'], alpha = 0.4, color = colors[4], linewidth=0, s = 40, label = '02 July 2021') #(0.3 m)
    
    ax2.plot([d + offset_21 for d in dist_21_06],track_07_21_06['h_mean'],'-o', markersize = 2, color = colors[4],path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()], label = '2021-07-02 (SR)')
    
    line = np.vstack((track_07_21.geometry.x.values, track_07_21.geometry.y.values)).T
    
    x_7_21, y_7_21 = intersect_line_contour(line, cont_2021ar)
    
    ax2.vlines((y_7_21 - y_min)*np.cos(angle), -3, 5, linestyle = '--',linewidth = 2, color = colors[4], label = '2021-07-02 Coastline (Planet)')
    
    
    x_12_21, y_12_21 = intersect_line_contour(line, cont_2022r)
    
    dist_22_03, angle_22 = project_on_t(track_12_21, t)
    dist_22_06, angle_22 = project_on_t(track_12_21_06, t)
    
    offset_22 = track_12_21.geometry.iloc[0].distance(track_12_21_06.geometry.iloc[0])*np.cos(angle_22)
    
    track_12_21['cross_track_distance'] = dist_22_03
    track_12_21_06['cross_track_distance'] = [d+offset_22 for d in dist_22_06]
    
    atl03_22 = ax2.scatter(dist_22_03, track_12_21['height'], alpha = 0.4, color = colors[6], linewidth=0,  s = 40,label = '01 December 2021')#(3.6 m)
    ax2.plot([d + offset_22 for d in dist_22_06],track_12_21_06['h_mean'],'-o', markersize = 2, color = colors[6],path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()], label = '2021-12-31 (SR)')
    ax2.vlines((y_12_21 - y_min)*np.cos(angle), -3, 5, linestyle = '--', linewidth = 2, color = colors[6], label = '2021-10-04 Coastline (Planet)')
    
     
    
    
    plt.xlabel('Cross-Shore Distance (m)', fontsize = 20)
    plt.ylabel('Ellipsoid Height (m)', fontsize = 20)
    plt.xticks([0, 25, 50, 75, 100, 125, 150, 175],fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.xlim([0, 200])
    plt.ylim(ylims)
    plt.subplots_adjust(left=0, bottom=0.13, right=.98, top=0.82, wspace=0, hspace=0.2)
    fig.set_size_inches((16.8,  7.7))
    plt.annotate('(f)', (0.008, .94), xycoords = 'axes fraction', \
                 bbox=dict(facecolor='w', edgecolor='none'), fontsize = 17)    
    plt.gcf().set_size_inches(16.8,  6)
    
    if name == 'bluff':
        #picked locations for bluff
        track_04_19_top = track_04_19_06[(track_04_19_06.geometry.y - y_min >= 93) & (track_04_19_06.geometry.y - y_min  <= 94)]
        track_04_19_bot = track_04_19_06[(track_04_19_06.geometry.y - y_min >= 121) & (track_04_19_06.geometry.y - y_min  <= 122)]
        

        track_01_20_top = track_01_20_06[(track_01_20_06.geometry.y - y_min >= 77) & (track_01_20_06.geometry.y - y_min  <= 79)]
        track_01_20_bot = track_01_20_06[(track_01_20_06.geometry.y - y_min >= 93) & (track_01_20_06.geometry.y - y_min  <= 95)]
        

        track_07_21_top = track_07_21_06[(track_07_21_06.geometry.y - y_min >= 55) & (track_07_21_06.geometry.y - y_min  <= 56)]
        track_07_21_bot = track_07_21_06[(track_07_21_06.geometry.y - y_min >= 73) & (track_07_21_06.geometry.y - y_min  <= 74)]
        

        
        track_12_21_top = track_12_21_06[(track_12_21_06.geometry.y - y_min >= 48) & (track_12_21_06.geometry.y - y_min  <= 49)]
        track_12_21_bot = track_12_21_06[(track_12_21_06.geometry.y - y_min >=93) & (track_12_21_06.geometry.y - y_min  <= 95)]
        
    elif name == 'lake':
        track_04_19_top = track_04_19_06[(track_04_19_06.geometry.y - y_min >= 123) & (track_04_19_06.geometry.y - y_min  <= 125)]
        track_04_19_bot = track_04_19_06[(track_04_19_06.geometry.y - y_min >= 165) & (track_04_19_06.geometry.y - y_min  <= 167)]
        
        track_01_20_top = track_01_20_06[(track_01_20_06.geometry.y - y_min >= 78) & (track_01_20_06.geometry.y - y_min  <= 80)]
        track_01_20_bot = track_01_20_06[(track_01_20_06.geometry.y - y_min >= 97) & (track_01_20_06.geometry.y - y_min  <= 99)]
        
        track_07_21_top = track_07_21_06[(track_07_21_06.geometry.y - y_min >= 76) & (track_07_21_06.geometry.y - y_min  <= 78)]
        track_07_21_bot = track_07_21_06[(track_07_21_06.geometry.y - y_min >= 86) & (track_07_21_06.geometry.y - y_min  <= 87)]
    
        track_12_21_top = track_12_21_06[(track_12_21_06.geometry.y - y_min >= 14) & (track_12_21_06.geometry.y - y_min  <= 16)]
        
        track_12_21_bot = track_12_21_06[(track_12_21_06.geometry.y - y_min >=66) & (track_12_21_06.geometry.y - y_min  <= 67)]
     
    elif name == 'beach':
        track_04_19_top = track_04_19_06[(track_04_19_06.geometry.y - y_min >= 122) & (track_04_19_06.geometry.y - y_min  <= 124)]
        track_04_19_bot = track_04_19_06[(track_04_19_06.geometry.y - y_min >= 176) & (track_04_19_06.geometry.y - y_min  <= 178)]
        
        track_01_20_top = track_01_20_06[(track_01_20_06.geometry.y - y_min >= 133) & (track_01_20_06.geometry.y - y_min  <= 134)]
        track_01_20_bot = track_01_20_06[(track_01_20_06.geometry.y - y_min >= 169) & (track_01_20_06.geometry.y - y_min  <= 170)]
        
        track_07_21_top = track_07_21_06[(track_07_21_06.geometry.y - y_min >= 135) & (track_07_21_06.geometry.y - y_min  <= 136)]
        track_07_21_bot = track_07_21_06[(track_07_21_06.geometry.y - y_min >= 159) & (track_07_21_06.geometry.y - y_min  <= 160)]
        
        track_12_21_top = track_12_21_06[(track_12_21_06.geometry.y - y_min >= 137) & (track_12_21_06.geometry.y - y_min  <= 138)]
        track_12_21_bot = track_12_21_06[(track_12_21_06.geometry.y - y_min >=174) & (track_12_21_06.geometry.y - y_min  <= 176)]
        
    ax2.plot(track_01_20_top['cross_track_distance'], \
             track_01_20_top.h_mean, marker = '^', c = colors[2], markersize = 25,mew = 2, markeredgecolor = 'k', label = None)
    ax2.plot(track_01_20_bot['cross_track_distance'], track_01_20_bot.h_mean, marker = 'v', c = colors[2], markersize = 25, markeredgecolor = 'k',mew = 2, label = None)
    
    ax2.plot(track_07_21_top['cross_track_distance'], \
             track_07_21_top.h_mean, marker = '^', c = colors[4], markersize = 25,mew = 2, markeredgecolor = 'k', label = None)
    ax2.plot(track_07_21_bot['cross_track_distance'], track_07_21_bot.h_mean, marker = 'v', c = colors[4], markersize = 25, markeredgecolor = 'k',mew = 2, label = None)
    
    ax2.plot(track_12_21_top['cross_track_distance'], \
             track_12_21_top.h_mean, marker = '^', c = colors[6], markersize =25, markeredgecolor = 'k', mew = 2, label = None)
    ax2.plot(track_12_21_bot['cross_track_distance'], track_12_21_bot.h_mean, marker = 'v', c = colors[6], markersize = 25, markeredgecolor = 'k', mew = 2, label = None)
    
    legend_symbols = [Line2D([0], [0], color = 'gray', marker = 'o', lw = 0, label = 'ATL03'),\
                      Line2D([0], [0],  color = 'gray',markeredgecolor = 'k', marker = 'o', linestyle = '-', label = 'ATL06-SR'),\
                      Line2D([0], [0],  color = 'gray', linestyle = '--', label = 'Planet shoreline'),\
                      Line2D([0], [0], color = 'gray',markeredgecolor = 'k', marker = '^',markersize = 15, lw = 0, label = 'Upper shoreline'),\
                      Line2D([0], [0], color = 'gray',markeredgecolor = 'k', marker = 'v', markersize= 15, lw = 0, label = 'Lower shoreline') ]

    legend_1 = plt.legend(handles = legend_symbols, \
               bbox_to_anchor=(.5, 1), loc = 'lower center', ncol = 3, fontsize = 20,\
                   handletextpad= .2, columnspacing=0.3)
    ax2.add_artist(legend_1)
    
    legend_2 = plt.legend(handles = [atl03_19, atl03_20, atl03_21, atl03_22], \
                          bbox_to_anchor = (1.015, 1.02), loc = 'upper right', \
                              title = 'Observation Date', fontsize = 20 ,handletextpad=0.05, \
                                  title_fontproperties={'weight':'bold', 'size':20},\
                                      frameon = True, framealpha = 1)
    
    
    top_item = ax2.plot(track_04_19_top['cross_track_distance'], \
             track_04_19_top.h_mean, marker = '^', c = 'k', markersize = 25, markeredgecolor = 'w',mew = 2, label = 'Shoreline upper boundary')
    bot_item = ax2.plot(track_04_19_bot['cross_track_distance'], track_04_19_bot.h_mean, marker = 'v', c = 'k', markersize = 25, markeredgecolor = 'w',mew = 2, label = 'Shoreline Lower  Boundary')
    
    for lh in legend_2.legendHandles: 
        lh.set_alpha(1)
    ax2.add_artist(legend_2)   


    
    tops_full = [track_04_19_top, track_01_20_top, \
                  track_07_21_top, \
                      track_12_21_top]

    bots_full = [track_04_19_bot, track_01_20_bot, \
                  track_07_21_bot, \
                      track_12_21_bot]
        
    planet_dates_list = ['2019-06-25', '2020-07-25', '2021-07-02',  '2022-07-01']
    planet_points =  [y_6_19-y_min,y_7_20-y_min,y_7_21-y_min, y_12_21 - y_min]
    planet_df = gpd.GeoDataFrame({'date': planet_dates_list, 'Boundary': ['Planet']*len(planet_dates_list), \
                            'error': 2.2*np.ones(len(planet_dates_list))}, geometry = [Point(x_6_19, y_6_19), Point(x_7_20, y_7_20),\
                                           Point(x_7_21, y_7_21), Point(x_12_21, y_12_21)])
                                                     
        
    return t, tops_full, bots_full, planet_df
#%%
bluff_t, bluff_tops_full, bluff_bots_full, bluff_planet_df =  plots_and_boundaries('bluff')
lake_t, lake_tops_full, lake_bots_full, lake_planet_df =  plots_and_boundaries('lake')
beach_t, beach_tops_full, beach_bots_full, beach_planet_df =  plots_and_boundaries('beach')


#%%
lake_t.to_file('lake_transect.json', driver = 'GeoJSON')
#%%
beach_t.to_file('beach_transect.json', driver = 'GeoJSON')
#%%
bluff_t.to_file('bluff_transect.json', driver = 'GeoJSON')


#%%
date_list = ['2019-04-07', '2020-01-04', '2021-07-02','2021-12-31']

lake_tops_df = pd.concat(lake_tops_full)
lake_tops_df['Boundary'] = ['upper']*len(lake_tops_df)
lake_bots_df = pd.concat(lake_bots_full)
lake_bots_df['Boundary'] = ['lower']*len(lake_bots_df)
lake_boundaries_full = pd.concat([lake_tops_df, lake_bots_df])
lake_boundaries_full_s = lake_boundaries_full[['date', 'Boundary', 'geometry', 'geo_error']]
lake_boundaries_full_s.rename(columns = {'geo_error': 'error'}, inplace = True) 
lake_boundaries_all = pd.concat([lake_boundaries_full_s, lake_planet_df])
lake_boundaries_all.sort_values(by = ['date'], inplace = True)
lake_boundaries_all.reset_index(inplace = True)
lake_boundaries_all.drop(columns = ['index'], inplace = True)

lake_boundaries_all.to_csv('output/gt2r_boundaries_n.csv')

  

#%%

beach_tops_df = pd.concat(beach_tops_full)
beach_tops_df['Boundary'] = ['upper']*len(beach_tops_df)
beach_bots_df = pd.concat(beach_bots_full)
beach_bots_df['Boundary'] = ['lower']*len(beach_bots_df)
beach_boundaries_full = pd.concat([beach_tops_df, beach_bots_df])
beach_boundaries_full_s = beach_boundaries_full[['date', 'Boundary', 'geometry', 'geo_error']]
beach_boundaries_full_s.rename(columns = {'geo_error': 'error'}, inplace = True) 
beach_boundaries_all = pd.concat([beach_boundaries_full_s, beach_planet_df])
beach_boundaries_all.sort_values(by = ['date'], inplace = True)
beach_boundaries_all.reset_index(inplace = True)
beach_boundaries_all.drop(columns = ['index'], inplace = True)

beach_boundaries_all.to_csv('output/gt1r_boundaries.csv')


#%%

bluff_tops_df = pd.concat(bluff_tops_full)
bluff_tops_df['Boundary'] = ['upper']*len(bluff_tops_df)
bluff_bots_df = pd.concat(bluff_bots_full)
bluff_bots_df['Boundary'] = ['lower']*len(bluff_bots_df)
bluff_boundaries_full = pd.concat([bluff_tops_df, bluff_bots_df])
bluff_boundaries_full_s = bluff_boundaries_full[['date', 'Boundary', 'geometry', 'geo_error']]
bluff_boundaries_full_s.rename(columns = {'geo_error': 'error'}, inplace = True) 
bluff_boundaries_all = pd.concat([bluff_boundaries_full_s, bluff_planet_df])
bluff_boundaries_all.sort_values(by = ['date'], inplace = True)
bluff_boundaries_all.reset_index(inplace = True)
bluff_boundaries_all.drop(columns = ['index'], inplace = True)

bluff_boundaries_all.to_csv('output/gt3r_boundaries.csv')
#%%get relative sea level
offshore_bluff = bsc_bluff_06_p[(bsc_bluff_06_p['date']== '2021-07-02') & (bsc_bluff_06_p.geometry.y > bluff_bots_full[2].geometry.y.values[0])]
offshore_beach = bsc_beach_06_p[(bsc_beach_06_p['date']== '2021-07-02') & \
                                (bsc_beach_06_p.geometry.y > beach_bots_full[2].geometry.y.values[0])]
offshore_lake = bsc_lake_06_p[(bsc_lake_06_p['date']== '2021-07-02') & \
                                    (bsc_lake_06_p.geometry.y > lake_bots_full[2].geometry.y.values[0])]

ref_elev = np.mean(np.concatenate((offshore_bluff['h_mean'].values, offshore_bluff['h_mean'].values, offshore_lake['h_mean'].values)))
def onshore_height(top):
    return top['h_mean'].values[0] - ref_elev
#%%
def backshore_slope(track, top, bot):
    backshore = track[(track.geometry.y >= top.geometry.y.values[0]) & \
                      (track.geometry.y <= bot.geometry.y.values[0])]
    p2 = np.polyfit(track['cross_track_distance'], backshore.h_mean, 1)
    m = p2[0]
    return m
 
#%%
bluff_heights = []
bluff_slopes = []
transect_bluff = gpd.read_file('data/gt3r_transect.json')
for i in range(0, len(date_list)):
    track = bsc_bluff_06_p[(bsc_bluff_06_p['date']== date_list[i])]
    dist, angle = project_on_t(track, transect_bluff)
    top = bluff_tops_full[i]
    bot = bluff_bots_full[i]
    bluff_slopes.append((top['h_mean'].values[0] - bot['h_mean'].values[0])/(top['cross_track_distance'].values[0] - bot['cross_track_distance'].values[0]))
    bluff_heights.append(onshore_height(top))
#%%
lake_heights = []
lake_slopes = []
transect_lake = gpd.read_file('data/gt2r_transect.json')

for i in range(0, len(date_list)):
    track = bsc_lake_06_p[(bsc_lake_06_p['date']== date_list[i])]
    dist, angle = project_on_t(track, transect_lake)
    top = lake_tops_full[i]
    bot = lake_bots_full[i]
    lake_slopes.append((top['h_mean'].values[0] - bot['h_mean'].values[0])/(top['cross_track_distance'].values[0] - bot['cross_track_distance'].values[0]))
    lake_heights.append(onshore_height(top))
#%%    
beach_heights = []
beach_slopes = []
transect_beach = gpd.read_file('data/gt1r_transect.json')

for i in range(0, len(date_list)):
    track = bsc_beach_06_p[(bsc_beach_06_p['date']== date_list[i])]
    dist, angle = project_on_t(track, transect_beach)
    top = beach_tops_full[i]
    bot = beach_bots_full[i]
    beach_slopes.append((top['h_mean'].values[0] - bot['h_mean'].values[0])/(top['cross_track_distance'].values[0] - bot['cross_track_distance'].values[0]))
    beach_heights.append(onshore_height(top))

#%%plot specific ATl03 tracks
colors = cm.lajolla((np.linspace(0, 1, 8)))

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)

t_bluff = gpd.read_file('data/gt3r_transect.json')

t_lake = gpd.read_file('data/gt3r_transect.json')

bluff_block = atl03_sr_bluff_p[(atl03_sr_bluff_p['date'] == '2021-07-02') & (atl03_sr_bluff_p['atl03_cnf'] >= 3)]
bluff_block_2 = atl03_sr_bluff_p[(atl03_sr_bluff_p['date'] == '2021-12-31') & (atl03_sr_bluff_p['atl03_cnf'] >= 3)]
y_min_bluff = np.min(bsc_bluff_06_p[bsc_bluff_06_p['date']=='2019-04-07'].geometry.y)
lake_water = atl03_sr_lake_p[(atl03_sr_lake_p['date'] == '2021-07-02') & (atl03_sr_lake_p['atl03_cnf'] >= 3)]
y_min_lake = np.min(bsc_lake_06_p[bsc_lake_06_p['date']=='2019-04-07'].geometry.y)

dist_b1, angle = project_on_t(bluff_block, t_bluff)

dist_b2, angle = project_on_t(bluff_block_2, t_bluff)

dist_l, angle = project_on_t(lake_water, t_lake)


ax1.scatter(dist_b2, bluff_block_2['height'], c = colors[4], alpha = 0.5, s = 20)
ax1.scatter(dist_b1, bluff_block['height'], c = colors[6], alpha = 0.5, s = 20)
plt.title('Ground track 3r', fontsize = 20)
plt.annotate('(a)', (0, .925), fontsize=15, xycoords = 'axes fraction')
l1 = plt.legend(['2 July 2021','31 December 2021'], fontsize = 20)
for lh in l1.legendHandles: 
    lh.set_alpha(1)
    
plt.xlabel('Cross-Shore Distance (m)', fontsize = 20)
plt.ylabel('Ellipsoid Height (m)', fontsize = 20)
plt.xticks([0, 25, 50, 75, 100, 125, 150, 175],fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlim([0, 200])
plt.ylim(-3, 4)

ax2 = fig.add_subplot(2, 1, 2)
ax2.scatter(lake_water.geometry.y - y_min_lake, lake_water['height'], c = colors[4], alpha = 0.5, s = 20)
plt.title('Ground track 2r', fontsize = 20)

plt.xlabel('Cross-Shore Distance (m)', fontsize = 20)
plt.ylabel('Ellipsoid Height (m)', fontsize = 20)
plt.xticks([0, 25, 50, 75, 100, 125, 150, 175],fontsize = 20)
plt.yticks(fontsize = 20)
plt.xlim([0, 200])
plt.ylim(-3, 4)
l2 = plt.legend(['2 July 2021'], fontsize = 20)
for lh in l2.legendHandles: 
    lh.set_alpha(1)
plt.annotate('(b)', (0, .925), fontsize=15, xycoords = 'axes fraction')
plt.subplots_adjust(left = 0.18, bottom = 0.101, right = 0.925, top = 0.948, wspace = 0.2, \
                    hspace = 0.45)

plt.gcf().set_size_inches(9,  8)

def plot_rms(track, title, transect):
    y_min = np.min(track[track['date']=='2019-04-07'].geometry.y)
    
    date_list = ['2019-04-07', '2020-01-04', '2021-07-02','2021-12-31']
    date_labels = ['7 April 2019', '4 January 2020', '2 July 2021','31 December 2021']
    fig = plt.figure()
    
    for i in range(0, 4):
        ax = fig.add_subplot(2, 2, i+1)
        date_i = date_list[i]
        track_i = track[track['date'] == date_i] 
        dist, angle = project_on_t(track_i, transect)
        subplot_label = ['(a)', '(b)', '(c)', '(d)']
        plt.scatter(dist, track_i['h_mean'], c = track_i['rms_misfit'], vmin = 0, vmax = .5)
        if i >1:
            plt.xlabel('Cross-Shore Distance (m)', fontsize = 25)
        plt.ylabel('Ellipsoid Height (m)', fontsize = 25)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize =20)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=20) 
        cbar.set_label(label = 'RMS error (m)',size = 20)
        plt.xlim([0, 200])
        plt.ylim(-3, 4)
        plt.annotate(subplot_label[i], (0, .93), xycoords = 'axes fraction', fontsize = 25)
        plt.subplots_adjust(left = .1, bottom = .09, right = 0.964, top = 0.88,\
                        wspace = 0.221, hspace = 0.267)
        
    fig.suptitle(title, fontsize = 30)
    plt.gcf().set_size_inches(14.4 ,  10)
#%%
plot_rms(bsc_beach_06_p, 'Ground Track 1r (Region 3)', transect_beach)
plot_rms(bsc_lake_06_p, 'Ground Track 2r (Region 2)', transect_lake)
plot_rms(bsc_bluff_06_p, 'Ground Track 1r (Region 3)', transect_bluff)


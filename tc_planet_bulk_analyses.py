#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 22:16:00 2023

@author: m1bryant
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import glob
import pdb
import datetime as dt
import pandas as pd
import geopandas as gpd
#%%functions
def distance_along_track(xs, ys):
    #calculates the along-track distance as the sum of distances between consecutive points
    #xs, us should both both 1-D arrays or lists
    x_0 = xs[0]
    y_0 = ys[0]
    distance = [0]
    for i in range(1, len(xs)):
        x_1 = xs[i-1] - x_0
        x_2 = xs[i] - x_0
        y_1 = ys[i-1] - y_0
        y_2 = ys[i] - y_0
        distance.append(distance[i-1]+ np.sqrt((x_2-x_1)**2 + (y_2-y_1)**2))
    return distance

def smooth_contour(contour, l):
    #calculates an along-track running mean of x/y positions
    #contour: 2-D array with x and y coordinates
    #l - length of the smoothing window in contour units (i.e meters)
     xs = contour[:, 0]
     ys = contour[:, 1]
     #get along-track distance
     dist = np.array(distance_along_track(contour[:, 0], contour[:, 1]))
     #trim edges of contour according to the window length
     first = l/2
     last = np.max(dist) - l/2
     dist_s = dist[(dist >=first) & (dist <=last)]
     xs_s = xs[(dist >=first) & (dist <=last)]
     ys_s = ys[(dist >= first) & (dist <=last)]
     
     x_smooth = []
     y_smooth = []
     #running mean
     for i in range(0, len(xs_s)):
         dist_i = dist_s[i]
         xis = xs[(dist >= dist_i - l/2) & (dist <= dist_i + l/2)]
         yis = ys[(dist >= dist_i - l/2) & (dist <= dist_i + l/2)]
         x_smooth.append(np.mean(xis))
         y_smooth.append(np.mean(yis))
     return np.transpose(np.array([x_smooth, y_smooth]))
 
def resample_along_track(contour, l):
    #resamples provided contour to have points sampled every l meters along-track]
    #contour: 2D array with x and y coordinates
    #l: along-track interval to resample contour to
    
    xs = contour[:, 0]
    ys = contour[:, 1]
    dist = distance_along_track(contour[:, 0], contour[:, 1])
    n_segs = int(np.floor((np.max(dist))/l)) #number of points after resampling
    segs = (np.linspace(0, n_segs*l, n_segs+1))
    #get ends points of segements using a nearest-neighbor interplolation
    x_e = np.interp(segs, dist, xs)
    y_e = np.interp(segs, dist, ys)
    x_n = [xs[0]]
    y_n = [ys[0]]
    #define final value for each segment based on linear interpoloation of all points within the segment
    for s in range(0, len(segs)-1):
        
        x = xs[(dist >= segs[s]) & (dist <=segs[s+1])]
        y = ys[(dist >= segs[s]) & (dist <=segs[s+1])]
        
        if len(x) == 0:
            pdb.set_trace()
        p = np.polyfit(x, y, 1)
        x_p = np.linspace(x_e[s], x_e[s+1], len(x))
        y_p = np.polyval(p, x_p)
        x_n.append(np.mean(x_p))
        y_n.append(np.mean(y_p))
   
    return np.transpose(np.array([x_e, y_e]))

def resample_coordinates(contour, xs_i):
    #re-sampled transect at provided x-coordinates
    #contour: 2D array with x and y coordinates
    #xs_i: list of x coordinates
    xs = contour[:, 0]
    ys = contour[:, 1]
    dist = distance_along_track(contour[:, 0], contour[:, 1])
    y_n = np.interp(xs_i, xs, ys)
    
    return np.transpose(np.array([xs_i, y_n])) 

def intersection(m1, b1, m2, b2):
    x = -(b1-b2)/(m1-m2)
    y = m1*x + b1
    return x, y
def intersect_line_contour(line, cont):
    x_l = line[:, 0]
    y_l = line[:, 1]
    #pdb.set_trace()
    p = np.polyfit(x_l, y_l, 1)
    m_l = p[0]
    b_l = p[1]
    x_c = cont[:, 0]
    y_c = cont[:, 1]
    
    #subset around range of line segment
    cont_s = cont[(x_c > np.min(x_l)-10) & (x_c < np.max(x_l)+10)]
    x_s = cont_s[:, 0]
    y_s = cont_s[:, 1]
    x_i = np.nan
    y_i = np.nan
    for i in range(0, len(x_s)-1):
        x = x_s[i:i+2]
        y = y_s[i:i+2]
        
        p2 = np.polyfit(x, y, 1)
        m_s = p2[0]
        b_s = p2[1]
        
        x_i, y_i = intersection(m_l, b_l, m_s, b_s)
        
        if ((x_i >= np.min(x)) and (x_i <= np.max(x)) and (y_i >= np.min(y)) and (y_i <= np.max(y))):
            break 
    if np.isnan(x_i):
        pdb.set_trace()
    return x_i, y_i

def along_transect_change(cont1, cont2, transects):
    #pdb.set_trace()
    coords_1 = np.array([intersect_line_contour(t, cont1) for t in transects])
    xs1 = coords_1[:, 0]
    ys1 = coords_1[:,1]
    coords_2 = np.array([intersect_line_contour(t, cont2) for t in transects])
    xs2 = coords_2[:, 0]
    ys2 = coords_2[:, 1]
    #pdb.set_trace()
    change = np.sqrt((xs2-xs1)**2 + (ys2-ys1)**2)
    sign = (ys2-ys1)/np.abs(ys2-ys1)
    return  change*sign, xs2, ys2



def date_from_fname(fname):
    #returns dates (in datetime format) for coastline files generated from planet_bulk_processing
    datestring = fname[10:18]
    year = int(datestring[0:4])
    month = int(datestring[4:6])
    day = int(datestring[6:8])
    date = dt.date(year, month, day)
    return date

def id_from_fname(fname):
    #returns satellite id for coastline files generated from planet_bulk_processing
    return fname[19:23]
#%%read in data
data_dir = '/Users/m1bryant/Documents/Planet Images/IS2_paper_bulk_coastlines/coastlines_2020_2021'

os.chdir(data_dir)

files = glob.glob('*.csv')
files.sort()

coastlines = [np.genfromtxt(f, delimiter = ',') for f in files]

dates = [date_from_fname(f) for f in files]
f_ids = [id_from_fname(f) for f in files]
#%% plot all raw coastlines   
plt.figure()
for i in range(0, len(coastlines)):
    c = coastlines[i]
    d = dates[i]
    plt.plot(c[:, 0], c[:, 1], label = d)
plt.legend()
plt.xlabel('Northings')
plt.ylabel('Eastings')
plt.title('All coastlines (raw)')
#%%
#smooth and re-sample contours
coastlines_rs = []
ys = []
plt.figure()
c_0 = coastlines[0]# reference coastline
#get reference coordinates for re-sampling:
c_0_s = smooth_contour(c_0, 30)
c_0_rs = resample_along_track(c_0_s, 10)
x_0 = c_0_rs[:, 0] 
colors = cm.viridis(((np.linspace(0, 1, len(coastlines) +1))))

labels = []
for i in range(0, len(coastlines)):
    c = coastlines[i]
    contour_s = smooth_contour(c, 30)
    contour_rs = resample_coordinates(contour_s, x_0)
    coastlines_rs.append(contour_rs)
    ys.append(contour_rs[:,1])
    plt.plot(contour_rs[:,0], contour_rs[:, 1], c =colors[i])
    labels.append(str(dates[i]) + ' (' + f_ids[i] + ')')
#plt.plot(x_0, np.mean(ys, axis = 0), 'k', lw = 3)

#labels.append('average')
plt.legend(labels)

plt.xlabel('Eastings', fontsize = 15)

plt.ylabel('Northings', fontsize = 15)

plt.title('All coastlines (post-processed)', fontsize = 20)

#%%time series to identify clusters that are close in space and time
y_mean = np.mean(ys, axis = 0)
ts = [ np.median(c[:,1] - y_mean) for c in coastlines_rs ]


plt.figure()
plt.plot(dates,ts, '.-')

plt.xlabel('Date', fontsize = 15)
plt.ylabel('Distance from reference (m)', fontsize = 15)
plt.title('Time series', fontsize = 20)
#%%load in transects
transect_df = gpd.read_file('/Users/m1bryant/Documents/Data/IS2_case_study/v2_data/planet/transects_2021_10m.json')
transect_list = []
for i in range(0, len(transect_df)):
    t = transect_df.iloc[i]
    line = np.array(t.geometry.xy).T
    transect_list.append(line)
#%%cluster analyses
def cluster_analysis(cluster, cluster_dates, title,transects, mask = None):
    #takes a cluster of shorelines, calcaultes the mean shoreline position, and plots the 
    #distribution of residuals from the mean
    cluster_ys = []
    colors = cm.viridis(((np.linspace(0, 1, len(cluster)+1))))
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    i = 0
    labels = []

    #plot and compile all cluster coastlines
    for c in cluster:
        cluster_ys.append(c[:,1]) #compile y values for mean compilation
        #plot each coastline
        ax1.plot(c[:,0], c[:, 1], c =colors[i])
        labels.append(str(cluster_dates[i]) + ' (' + f_ids[i] + ')')
        i = i+1
    
    coastline_mean = np.array([x_0, np.mean(cluster_ys, axis = 0)]).T
    plt.plot(coastline_mean[:,0], coastline_mean[:,1], 'k') #calculate and plot mean coastline
    labels.append('average')
    plt.legend(labels)
    plt.xlabel('Eastings', fontsize = 15)
    plt.ylabel('Northings', fontsize = 15)
    plt.suptitle(title + ': ' + str(cluster_dates[0]) + ' to ' + str(cluster_dates[len(cluster_dates)-1]),\
                 fontsize = 15)
    ax2 = fig.add_subplot(2, 1, 2)
    
    resids_list = []
    x_list = []
    for c in cluster:
        change, x, y = along_transect_change(coastline_mean, c, transects)
        resids_list.append(change)
        x_list.append(x) #store x-coordinates corresponding to change estimates
    resids = np.array(resids_list)
    mean_0 = np.nanmean(resids)
    sigma_0 = np.nanstd(resids)
    
    if mask is not None:
        if mask == 'sigma':
            resids[((resids > mean_0 + 3 * sigma_0) | (resids < mean_0 - 3* sigma_0))] = np.nan
        else:
            for m in mask:
                resids[:,(x_0 >= m[0]) & (x_0<= m[1])] = np.nan

    for i in range(0, resids.shape[0]):
        plt.plot(x_list[i], resids[i, :], c = colors[i])
    plt.legend(labels)
    plt.xlabel('Eastings', fontsize = 15)
    plt.ylabel('Residuals (m)', fontsize = 15)
    plt.figure()
    plt.hist(resids.flatten(), bins = 100)
    #calculate residuals from mean
    
    
    mean_resid = np.nanmean(resids)
    sigma = np.nanstd(resids)
    
    #plt.annotate('mean = ' + "{:.2f}".format((mean_resid)), (.15, .8), xycoords = 'figure fraction')
    plt.annotate('sigma = ' + "{:.2f}".format(sigma) + ' m', (.3, .75), xycoords = 'figure fraction',\
                 fontsize = 12)
    
    plt.xlabel('Residual (m)', fontsize = 15)
    plt.ylabel('Frequency', fontsize = 15)
    
    plt.title(title + ': ' + str(cluster_dates[0]) + ' to ' + str(cluster_dates[len(cluster_dates)-1]),
              fontsize = 15)
    return resids
#%%initiliaze lists for bulk statistcs
all_resids = []
all_coastlines = []
all_dates = []
all_ids = []
#%% First cluster  
cluster_20a = coastlines_rs[0:4] 
cluster_dates_20a = dates[0:4]
cluster_ids_20a = f_ids[0:4]
resids_20a = cluster_analysis(cluster_20a, cluster_dates_20a, '2020 Cluster A', mask = 'sigma', transects = transect_list)
#%%
all_resids.extend(resids_20a)
all_coastlines.extend(cluster_20a)
all_dates.extend(cluster_dates_20a)
all_ids.extend(cluster_ids_20a)
#%%
cluster_20b = coastlines_rs[4:10]
cluster_dates_20b = dates[4:10]
cluster_ids_20b = f_ids[4:10]
resids_20b = cluster_analysis(cluster_20b, cluster_dates_20b, '2020 Cluster B', mask = 'sigma', transects = transect_list)
#%%
all_resids.extend(resids_20b)
all_coastlines.extend(cluster_20b)
all_dates.extend(cluster_dates_20b)
all_ids.extend(cluster_ids_20b)
#%%
cluster_20c = coastlines_rs[10:13] 
cluster_dates_20c = dates[10:13]
cluster_ids_20c = f_ids[10:13]

resids_20c = cluster_analysis(cluster_20c, cluster_dates_20c, '2020 Cluster C', mask = 'sigma', transects = transect_list)
#%%
all_resids.extend(resids_20c)
all_coastlines.extend(cluster_20c)
all_dates.extend(cluster_dates_20c)
all_ids.extend(cluster_ids_20c)

#%%
cluster_21a = coastlines_rs[13:16] 
cluster_dates_21a = dates[13:16]
cluster_ids_21a = f_ids[13:16]

resids_21a = cluster_analysis(cluster_21a, cluster_dates_21a, '2021 Cluster A', mask = 'sigma', transects = transect_list) 
#%%
all_resids.extend(resids_21a)
all_coastlines.extend(cluster_21a)
all_dates.extend(cluster_dates_21a)
all_ids.extend(cluster_ids_21a)

#%%
cluster_21b = coastlines_rs[16:19]
cluster_dates_21b = dates[16:19]
cluster_ids_21b = f_ids[16:19]
resids_21b = cluster_analysis(cluster_21b, cluster_dates_21b, '2021 Cluster B', mask = 'sigma',  transects = transect_list)
#%%
all_resids.extend(resids_21b)
all_coastlines.extend(cluster_21b)
all_dates.extend(cluster_dates_21b)
all_ids.extend(cluster_ids_21b)
#%%
cluster_21c = coastlines_rs[19:23]
cluster_dates_21c = dates[19:23]
cluster_ids_21c = f_ids[19:23]


resids_21c = cluster_analysis(cluster_21c, cluster_dates_21c, '2021 Cluster C', mask = 'sigma' , transects = transect_list)
#%%
all_resids.extend(resids_21c)
all_coastlines.extend(cluster_21c)
all_dates.extend(cluster_dates_21c)
all_ids.extend(cluster_ids_21c)

#%%
resids_array = np.concatenate(tuple(all_resids))
plt.figure()
plt.hist(resids_array.flatten(), bins = 100)
mean_resid = np.nanmean(resids_array)
sigma = np.nanstd(resids_array)

#plt.annotate('mean = ' + "{:.2f}".format((mean_resid)), (.15, .8), xycoords = 'figure fraction')
plt.annotate('sigma = ' + "{:.2f}".format(sigma) + ' m', (.3, .75), xycoords = 'figure fraction',\
             fontsize = 12)

plt.xlabel('Residual (m)', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)
#%%save coastlines used for residual calculations
savedir = '/Users/m1bryant/Documents/Data/IS2_case_study/Planet/uncertainty_coastlines'
os.chdir(savedir)
for i in range(0, len(all_coastlines)):
    coast = all_coastlines[i]
    df  = pd.DataFrame({'X': coast[:, 0], 'Y': coast[:,1]})
    date = all_dates[i]
    c_id = all_ids[i]
    fname = 'coastline_' + str(date.year) + '_' + str(date.month) + '_' + str(date.year) + '_'+ str(c_id)+ '_resampled.csv'
    df.to_csv(fname)
#%%
coastlines_rs = []
ys = []
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
c_0 = coastlines[0]# reference coastline
#get reference coordinates for re-sampling:
c_0_s = smooth_contour(c_0, 30)
c_0_rs = resample_along_track(c_0_s, 10)
x_0 = c_0_rs[:, 0] 
colors = cm.viridis(np.linspace(0, 1, len(coastlines) +1))

labels = []
for i in range(0, len(coastlines)):
    c = coastlines[i]
    contour_s = smooth_contour(c, 30)
    contour_rs = resample_coordinates(contour_s, x_0)
    coastlines_rs.append(contour_rs)
    ys.append(contour_rs[:,1])
    ax1.plot(contour_rs[:,0], contour_rs[:, 1], c =colors[i])
    labels.append(str(dates[i]) + ' (' + f_ids[i] + ')')
#plt.plot(x_0, np.mean(ys, axis = 0), 'k', lw = 3)

#labels.append('average')
#plt.legend(labels)

plt.xlabel('Eastings', fontsize = 15)

plt.ylabel('Northings', fontsize = 15)
#%%
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ts_1 = [ np.median(c[:,1] - c_0_rs[:,1]) for c in coastlines_rs ]


colors = cm.viridis(np.linspace(0, 1, len(coastlines) +1))
ax1.plot(dates,ts_1,  'k', zorder = 0)
ax1.scatter(dates,ts_1,  color = colors[:(len(colors)-1)])
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.set_xlabel('Date', fontsize = 20)

ax2.plot(dates,ts_1,  'k', zorder = 0)
ax2.scatter(dates,ts_1, color = colors[:(len(colors)-1)])
ax2.tick_params(axis='both', which='major', labelsize=15)
ax1.set_xlim([dt.date(2020, 6, 1), dt.date(2020, 11, 1)])
ax2.set_xlim([dt.date(2021, 6, 1), dt.date(2021, 11, 1)])
#ax1.spines.right.set_visible(False)
#ax2.spines.left.set_visible(False)
#ax1.yaxis.tick_left()
#ax1.tick_params(labelright=False)  # don't put tick labels at the top
ax2.yaxis.tick_left()
ax2.yaxis.tick_right()
d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)

plt.xlabel('Date', fontsize = 20)
ax1.set_ylabel('Distance from reference (m)', fontsize = 20)
plt.suptitle('Spatially averaged shoreline change since 2020-06-26', fontsize = 20)

 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 17:53:19 2023

Last modified Jan 30 2025

@author: Marnie Bryant m1bryant@ucsd.edu
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import geopandas as gpd
import pandas as pd
from shapely import wkt
import datetime as dt
import matplotlib.patheffects as pe
import scipy as sp
import matplotlib.patches as patches
from shapely import Point
plt.rcParams["font.family"] = "arial"

   
#%%
lake_boundaries = pd.read_csv('data/ICESat_2/gt2r_boundaries.csv')
lake_boundaries['geometry'] = lake_boundaries['geometry'].apply(wkt.loads)
lake_boundaries_gdf = gpd.GeoDataFrame(lake_boundaries, crs='EPSG:32605')

lake_tops = lake_boundaries_gdf[lake_boundaries['Boundary'] =='upper'].geometry
lake_bots = lake_boundaries_gdf[lake_boundaries['Boundary'] =='lower'].geometry
lake_planet = lake_boundaries_gdf[lake_boundaries['Boundary'] =='Planet'].geometry

bluff_boundaries = pd.read_csv('data/ICESat_2/gt3r_boundaries.csv')
bluff_boundaries['geometry'] = bluff_boundaries['geometry'].apply(wkt.loads)
bluff_boundaries_gdf = gpd.GeoDataFrame(bluff_boundaries, crs='EPSG:32605')

bluff_tops = bluff_boundaries_gdf[bluff_boundaries['Boundary'] =='upper'].geometry
bluff_bots = bluff_boundaries_gdf[bluff_boundaries['Boundary'] =='lower'].geometry
bluff_planet = bluff_boundaries_gdf[bluff_boundaries['Boundary'] =='Planet'].geometry

beach_boundaries = pd.read_csv('data/ICESat_2/gt1r_boundaries.csv')
beach_boundaries['geometry'] = beach_boundaries['geometry'].apply(wkt.loads)
beach_boundaries_gdf = gpd.GeoDataFrame(beach_boundaries, crs='EPSG:32605')

beach_tops = beach_boundaries_gdf[beach_boundaries['Boundary'] =='upper'].geometry
beach_bots = beach_boundaries_gdf[beach_boundaries['Boundary'] =='lower'].geometry
beach_planet = beach_boundaries_gdf[beach_boundaries['Boundary'] =='Planet'].geometry

#%%reproject IS2 data to local perpindicular

transects = gpd.read_file('data/transects_2021_50m.json')
transect_bluff = gpd.read_file('data/bluff_transect.json')
transect_beach = gpd.read_file('data/beach_transect.json')
transect_lake = gpd.read_file('data/lake_transect.json')
#%%
def cross_shore_change(point1, point2, t):
    x1 = point1.x
    x2 = point2.x
    y1 = point1.y
    y2 = point2.y
    mean_point = Point(np.mean([x1, x2]), np.mean([y1, y2]))
    
    change_vec = [x2- x1, y2-y1]
    coords = [g.coords.xy for g in t.geometry][0]
    t_vec = [coords[0][-1] - coords[0][0], coords[1][-1]-coords[1][0]]
    change_proj = np.dot(change_vec, t_vec)/np.linalg.norm(t_vec)
    return change_proj

def cross_shore_time_series(points, t):
    change_ts = []
    for i in range(0, len(points)-1):
        p1 = points.iloc[i]
        p2 = points.iloc[i+1]
        change = cross_shore_change(p1, p2, t) 
        change_ts.append(change)
    return change_ts
#%%
dates_is2 = [dt.date(2019, 4, 7), dt.date(2020, 1, 4), \
             dt.date(2021, 7, 2), dt.date(2021, 10, 1), dt.date(2021, 12, 31)]
dates_is2_s = [dt.date(2019, 4, 7), dt.date(2020, 1, 4), \
                 dt.date(2021, 7, 2), dt.date(2021, 12, 31)]
dates_planet = [dt.date(2019, 6, 25), dt.date(2020, 7, 25), dt.date(2021, 7, 1),dt.date(2021, 10, 4), dt.date(2022, 7, 1)]

dates_planet_s = [dt.date(2019, 6, 25), dt.date(2020, 7, 25), dt.date(2021, 7, 1), dt.date(2022, 7, 1)]

bluff_change_planet = cross_shore_time_series(bluff_planet, transect_bluff)

lake_change_planet = cross_shore_time_series(lake_planet, transect_lake)

beach_change_planet = cross_shore_time_series(beach_planet, transect_beach)

bluff_change_is2_top = cross_shore_time_series(bluff_tops, transect_bluff)

bluff_change_is2_bots = cross_shore_time_series(bluff_bots, transect_bluff)

lake_change_is2_top = cross_shore_time_series(lake_tops, transect_lake)

lake_change_is2_bots = cross_shore_time_series(lake_bots, transect_lake)

beach_change_is2_top = cross_shore_time_series(beach_tops, transect_beach)
beach_change_is2_bots = cross_shore_time_series(beach_bots, transect_beach)

planet_change = np.concatenate((bluff_change_planet, lake_change_planet, beach_change_planet))

is2_top_change = np.concatenate((bluff_change_is2_top, lake_change_is2_top, beach_change_is2_top))

is2_bot_change = np.concatenate((bluff_change_is2_bots, lake_change_is2_bots, beach_change_is2_bots))
#%%is2 uncertainties
date_list = ['2019-04-07','2020-01-04', '2021-07-02', '2021-12-31']
total_error = [4.2, 4.8, 2.8, 3.3, 2.8, 2.5] #positional uncertainites from Lutchke et al
def get_is2_errors(boundaries):
    error = []
    for d in date_list:
        b = boundaries[boundaries['date'] == d]
        error.append(b.iloc[0]['error'])
    return error
#%%

bluff_errors = get_is2_errors(bluff_boundaries)

bluff_change_errors = [np.sqrt(bluff_errors[i]**2 + bluff_errors[i+1]**2) for i in range(0, len(bluff_errors)-1)]

lake_errors = get_is2_errors(lake_boundaries)
lake_change_errors = [np.sqrt(lake_errors[i]**2 + lake_errors[i+1]**2) for i in range(0, len(lake_errors)-1)]


beach_errors = get_is2_errors(beach_boundaries)
beach_change_errors = [np.sqrt(beach_errors[i]**2 + beach_errors[i+1]**2) for i in range(0, len(beach_errors)-1)]

is2_errors = np.concatenate((bluff_change_errors, lake_change_errors, beach_change_errors))


#%%
#get number of open water days corresponding to each IS2 and Planet interval
def daily_list(date_1, date_2):
    n_days = (date_2-date_1).days
    date_list = [date_1 + dt.timedelta(n) for n in range(0, n_days+1)]
    return np.array(date_list)

#
def dates_from_file(file):
    date_array = np.loadtxt(file, dtype = str,delimiter = " ")
    dates = np.array([dt.datetime.strptime(d,'%Y-%m-%d').date() for d in date_array])
    return dates

ows_19 = dates_from_file('data/era_5/owd_2019.csv')
ows_20 = dates_from_file('data/era_5/owd_2020.csv')
ows_21 = dates_from_file('data/era_5/owd_2021.csv')
ows_22 = dates_from_file('data/era_5/owd_2022.csv')


ows_all = np.concatenate((ows_19, ows_20, ows_21, ows_22))


is2_intervals = [daily_list(dates_is2_s[0], dates_is2_s[1]), daily_list(dates_is2_s[1], dates_is2_s[2]),\
                 daily_list(dates_is2_s[2], dates_is2_s[3])]

    
planet_intervals = [daily_list(dates_planet_s[0], dates_planet_s[1]), daily_list(dates_planet_s[1], dates_planet_s[2]),\
                 daily_list(dates_planet_s[2], dates_planet_s[3])]

    
    
is2_owd = np.array([len(np.intersect1d(i, ows_all)) for i in is2_intervals])


planet_owd = np.array([len(np.intersect1d(i, ows_all)) for i in planet_intervals])



#%%
def linear(B, x):
    return B[0]*x + B[1]

def r_squared(data, model):
    res = model - data
    ss_res =np.sum([r**2 for r in res])
    ss = np.sum([(d - np.mean(data))**2 for d in data])
    return (1 - ss_res/ss)

def demming_regression_analysis(x, y, sx, sy):
    linear_model = sp.odr.Model(linear)
    data = sp.odr.RealData(x,y, sx = sx, sy = sy)
    linear_fit = sp.odr.ODR(data, linear_model, beta0 = [1, 0])
    output = linear_fit.run()
    slope = output.beta[0]
    intercept = output.beta[1]
    predicted = x*slope + intercept
    r_2 = r_squared(y, predicted) 
    df = len(y) - 2
    t_score = (slope/(output.sd_beta[0])) #test null hypothesis that slope is 0
    p_value = sp.stats.t.sf(np.abs(t_score), df) * 2
    return slope, intercept, r_2, p_value




#%%
sigma = 3.1
fig = plt.figure(figsize = [16.8 ,  9.61])
ax1 = fig.add_subplot(1, 2, 2)
ax1.scatter(bluff_change_is2_bots, bluff_change_is2_top,c= 'k', s = 150, marker = 'o', label = 'gt3r')
ax1.scatter(lake_change_is2_bots, lake_change_is2_top,c= 'k', s = 150,marker = 'd', label = 'gt2r')
ax1.scatter(beach_change_is2_bots, beach_change_is2_top,c= 'k', s = 300,marker = '*', label = 'gt1r')
ax1.scatter(lake_change_is2_bots[2], lake_change_is2_top[2], c= 'grey', s = 150,marker = 'd', label = 'Outlier')


plt.legend()
plt.xlim([-80, 30])
plt.ylim([-80, 30])

#drop the outlier

is2_top_change_no = np.delete(is2_top_change, 5)
is2_bot_change_no = np.delete(is2_bot_change, 5)

plt.xlabel('Shoreline change from ICESat-2 (lower) (m)', fontsize =25)
plt.ylabel('Shoreline change from ICESat-2 (upper) (m)', fontsize =25)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)

slope_is2, intercept, r_2, p_value_is2 = demming_regression_analysis(is2_bot_change_no, is2_top_change_no, is2_errors, is2_errors)
line_interp = np.linspace(-80, 30, 20)*slope_is2 + intercept
ax1.plot(np.linspace(-80, 30, 20), line_interp, 'k', dashes = [3,3], label = 'Trend')
ax1.errorbar(is2_bot_change, is2_top_change, xerr = is2_errors,yerr = is2_errors, ecolor ='k' , \
             zorder = 0, fmt = 'none', elinewidth=1)
ax1.plot([-80,30], [-80, 30], color = 'lightgray', zorder = 0,label ='1-1 Line')
plt.annotate('$r^2$ = ' + '%2.2f' %r_2, (0, -55), fontsize = 20)

plt.annotate('(b)', (0, .97), xycoords = 'axes fraction', fontsize = 20)
plt.legend(loc = 'upper left', bbox_to_anchor = (.04, 1.015), fontsize = 20)

#add planet vs is2
ax2 = fig.add_subplot(1,2, 1)
ax2.scatter(bluff_change_planet, bluff_change_is2_top, c= 'orange',edgecolor = 'k', s = 150,marker = 'o', label = 'gt3r (upper)')
    
ax2.scatter(lake_change_planet, lake_change_is2_top, c= 'orange', edgecolor = 'k', s = 150,marker = 'd', label = 'gt2r (upper)')
ax2.scatter(beach_change_planet, beach_change_is2_top, c= 'orange',edgecolor = 'k', s = 300,marker = '*', label = 'gt1r (upper)')
st, it, rt, pt = demming_regression_analysis(planet_change, is2_top_change, sigma, is2_errors)
top_interp = np.linspace(-80, 30, 20)*st + it

plt.plot(np.linspace(-80, 30, 20), top_interp,c = 'orange', dashes = [3,3], label = 'Trend (upper)')
ax2.scatter(bluff_change_planet, bluff_change_is2_bots, c= 'b',s = 150, marker = 'o', edgecolor = 'k',label = 'gt3r (lower)')
    
ax2.scatter(lake_change_planet, lake_change_is2_bots, c= 'b',edgecolor = 'k', s = 150,marker = 'd', label = 'gt2r(lower)')
ax2.scatter(beach_change_planet, beach_change_is2_bots, c= 'b',edgecolor = 'k', s = 300, marker = '*', label = 'gt1r (lower)')
sb, ib, rb, pb = demming_regression_analysis(planet_change, is2_bot_change, sigma, is2_errors)
bot_interp = np.linspace(-80, 30, 20)*sb + ib
ax2.plot(np.linspace(-80, 30, 20), bot_interp,c = 'b', dashes = [3,3], label = 'Trend (lower)')

plt.annotate('$r^2$ = ' + '%2.2f' %rt, (0, -55), color = 'orange', fontsize = 20)
plt.annotate('$r^2$ = ' + '%2.2f' %rb, (0, -60), color = 'b', fontsize = 20)
ax2.errorbar(planet_change, is2_top_change, xerr = sigma,yerr = is2_errors, ecolor ='k' , \
             zorder = 0, fmt = 'none', elinewidth=1)
ax2.errorbar(planet_change, is2_bot_change, xerr = sigma,yerr = is2_errors, ecolor ='k' , \
             zorder = 0, fmt = 'none', elinewidth=1)
ax2.plot([-80,30], [-80, 30], color = 'lightgray', zorder = 0,label ='1-1 Line')
plt.xlabel('Shoreline change from Planet (m)', fontsize = 25)
plt.ylabel('Shoreline change from ICESat-2 (m)', fontsize = 25)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.annotate('(a)', (0, .97), xycoords = 'axes fraction', fontsize = 20)
plt.legend(loc = 'upper left', bbox_to_anchor = (.04, 1.015), fontsize = 20 )
plt.xlim([-80, 30])
plt.ylim([-80, 30])
plt.subplots_adjust(left=0.086, bottom=0.11, right=0.995, top=0.88, wspace=0.246, hspace=0.2)
#plt.gcf().set_size_inches(18.57, 9.91)

#%%
start_date = dt.date(2019, 2, 1)
end_date = dt.date(2022, 8, 1)
fig = plt.figure(figsize = [16.8 ,  9.61])

bluff_tops.reset_index(drop = True,inplace = True)
bluff_bots.reset_index(drop = True,inplace = True)
bluff_planet.reset_index(drop = True,inplace = True)

lake_tops.reset_index(drop = True,inplace = True)
lake_bots.reset_index(drop = True,inplace = True)
lake_planet.reset_index(drop = True,inplace = True)

beach_tops.reset_index(drop = True,inplace = True)
beach_bots.reset_index(drop = True,inplace = True)
beach_planet.reset_index(drop = True,inplace = True)

#beach_tops

dates_is2_all = [dates_is2[0], ows_19[0], ows_19[-1], dates_is2[1],ows_20[0],ows_20[-1],\
                 ows_21[0], dates_is2[2], ows_21[-1], dates_is2[4]]
dates_planet_all = [dates_planet[0], ows_19[-1],dates_planet[1],ows_20[-1],\
                  dates_planet[2], ows_21[-1], dates_planet[4]]



bluff_top_dist = [cross_shore_change(bluff_planet[0], b, transect_bluff) for b in bluff_tops]
is2_bluff_t_all = [bluff_top_dist[0],bluff_top_dist[0], bluff_top_dist[1], bluff_top_dist[1],\
                   bluff_top_dist[1], bluff_top_dist[2], bluff_top_dist[2],bluff_top_dist[2],\
                       bluff_top_dist[3], bluff_top_dist[3]]

bluff_bot_dist = [cross_shore_change(bluff_planet[0], b, transect_bluff) for b in bluff_bots]    
is2_bluff_b_all = [bluff_bot_dist[0],bluff_bot_dist[0], bluff_bot_dist[1], bluff_bot_dist[1],\
                   bluff_bot_dist[1], bluff_bot_dist[2], bluff_bot_dist[2],bluff_bot_dist[2],\
                       bluff_bot_dist[3], bluff_bot_dist[3]]

bluff_planet_dist = [cross_shore_change(bluff_planet[0], b, transect_bluff) for b in bluff_planet]    

bluff_planet_all = [bluff_planet_dist[0], bluff_planet_dist[1], bluff_planet_dist[1],bluff_planet_dist[2],\
                    bluff_planet_dist[2], bluff_planet_dist[3], bluff_planet_dist[3]]

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(dates_is2_all, is2_bluff_t_all, '--', color = 'gray', lw = 3, label = None)

ax1.plot(dates_is2_all, is2_bluff_b_all, '--', color = 'gray', lw = 3,label = None)
ax1.plot(dates_planet_all, bluff_planet_all, '--',color = 'k', lw = 3,label = None)
ax1.plot(dates_is2_s, bluff_top_dist, marker = '^', markersize = 15, lw = 0, color = 'gray')
ax1.plot(dates_is2_s, bluff_bot_dist, marker = 'v', markersize  = 15,lw = 0, color = 'gray')
ax1.plot(dates_planet_s, bluff_planet_dist, marker = 'o', markersize  = 15,lw = 0, color = 'k')
plt.yticks(fontsize = 25)
plt.xticks([dt.date(2019, 6, 1), dt.date(2019, 12, 1), dt.date(2020, 6, 1),
            dt.date(2020, 12, 1), dt.date(2021, 6, 1), dt.date(2021, 12, 1), \
                dt.date(2022, 6, 1)], ['June\n2019', 'Dec\n2019', 'June\n2020',\
                                       'Dec\n 2020','June\n2021', 'Dec\n2021',\
                                       'June\n2022'],fontsize = 23, horizontalalignment = 'center')
plt.ylabel('Distance from June 2019 \n shoreline (m)', fontsize = 23)
plt.title('(a) Ground track 3r', fontsize = 20, loc= 'left', x= 0.01,y = 1.0, pad = -20.5, backgroundcolor = 'white')

gray_11 = patches.Rectangle((start_date, -170), ows_19[0]-start_date, 250, color = 'lightgray', zorder = 0)
gray_12 = patches.Rectangle((ows_19[-1], -170), ows_20[0]-ows_19[-1], 250, color = 'lightgray', zorder = 0)
gray_13 = patches.Rectangle((ows_20[-1], -170), ows_21[0]-ows_20[-1], 250, color = 'lightgray', zorder = 0)
gray_14 = patches.Rectangle((ows_21[-1], -170), ows_22[0]-ows_21[-1], 250, color = 'lightgray', zorder = 0)

ax1.add_patch(gray_11)
ax1.add_patch(gray_12)
ax1.add_patch(gray_13)
ax1.add_patch(gray_14)

plt.xlim([start_date, end_date])
plt.ylim([-120, 80])

lake_top_dist = [cross_shore_change(lake_planet[0], b, transect_lake) for b in lake_tops]    

is2_lake_t_all = [lake_top_dist[0],lake_top_dist[0], lake_top_dist[1], lake_top_dist[1],\
                   lake_top_dist[1], lake_top_dist[2], lake_top_dist[2],lake_top_dist[2],\
                       lake_top_dist[3], lake_top_dist[3]]
    

lake_bot_dist = [cross_shore_change(lake_planet[0], b, transect_lake) for b in lake_bots]    

is2_lake_b_all = [lake_bot_dist[0],lake_bot_dist[0], lake_bot_dist[1], lake_bot_dist[1],\
                   lake_bot_dist[1], lake_bot_dist[2], lake_bot_dist[2],lake_bot_dist[2],\
                       lake_bot_dist[3], lake_bot_dist[3]]
    
lake_planet_dist = [cross_shore_change(lake_planet[0], b, transect_lake) for b in lake_planet] 

lake_planet_all = [lake_planet_dist[0], lake_planet_dist[1], lake_planet_dist[1],lake_planet_dist[2],\
                    lake_planet_dist[2], lake_planet_dist[3], lake_planet_dist[3]]

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(dates_is2_all, is2_lake_t_all, '--', color = 'gray', lw = 3,label = None)
ax2.plot(dates_is2_all, is2_lake_b_all, '--', color = 'gray', lw = 3,label = None)
ax2.plot(dates_planet_all, lake_planet_all, '--',color = 'k', lw = 3,label = None)
ax2.plot(dates_is2_s, lake_top_dist, marker = '^', markersize = 15,lw =0, color = 'gray')
ax2.plot(dates_is2_s, lake_bot_dist, marker = 'v', markersize = 15,lw=0, color = 'gray')
ax2.plot(dates_planet_s, lake_planet_dist, marker = 'o', markersize = 15,lw=0, color ='k')
plt.yticks(fontsize = 23)
plt.xticks([dt.date(2019, 6, 1), dt.date(2019, 12, 1), dt.date(2020, 6, 1),
            dt.date(2020, 12, 1), dt.date(2021, 6, 1), dt.date(2021, 12, 1), \
                dt.date(2022, 6, 1)], ['June\n2019', 'Dec\n2019', 'June\n2020',\
                                       'Dec\n 2020','June\n2021', 'Dec\n2021',\
                                       'June\n2022'],fontsize = 23, horizontalalignment = 'center')
plt.ylabel('Distance from June 2019 \n shoreline (m)', fontsize = 23)
plt.title('(b) Ground track 2r',fontsize = 20, loc= 'left', x= 0.01,y = 1.0, pad = -20.5, backgroundcolor = 'white')

gray_21 = patches.Rectangle((start_date, -170), ows_19[0]-start_date, 250, color = 'lightgray', zorder = 0)
gray_22 = patches.Rectangle((ows_19[-1], -170), ows_20[0]-ows_19[-1], 250, color = 'lightgray', zorder = 0)
gray_23 = patches.Rectangle((ows_20[-1], -170), ows_21[0]-ows_20[-1], 250, color = 'lightgray', zorder = 0)
gray_24 = patches.Rectangle((ows_21[-1], -170), ows_22[0]-ows_21[-1], 250, color = 'lightgray', zorder = 0)

ax2.add_patch(gray_21)
ax2.add_patch(gray_22)
ax2.add_patch(gray_23)
ax2.add_patch(gray_24)

plt.xlim([start_date, end_date])

plt.ylim([-120, 80])

    
beach_top_dist = [cross_shore_change(beach_planet[0], b, transect_beach) for b in beach_tops]

is2_beach_t_all = [beach_top_dist[0],beach_top_dist[0], beach_top_dist[1], beach_top_dist[1],\
                   beach_top_dist[1], beach_top_dist[2], beach_top_dist[2],beach_top_dist[2],\
                       beach_top_dist[3], beach_top_dist[3]]
    
beach_bot_dist = [cross_shore_change(beach_planet[0], b, transect_beach) for b in beach_bots]

    
is2_beach_b_all = [beach_bot_dist[0],beach_bot_dist[0], beach_bot_dist[1], beach_bot_dist[1],\
                   beach_bot_dist[1], beach_bot_dist[2], beach_bot_dist[2],beach_bot_dist[2],\
                       beach_bot_dist[3], beach_bot_dist[3]]

beach_planet_dist = [cross_shore_change(beach_planet[0], b, transect_beach) for b in beach_planet]

beach_planet_all = [beach_planet_dist[0], beach_planet_dist[1], beach_planet_dist[1],beach_planet_dist[2],\
                    beach_planet_dist[2], beach_planet_dist[3], beach_planet_dist[3]]
    


ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(dates_is2_all, is2_beach_t_all, '--', color = 'gray', lw = 3,label = None)
ax3.plot(dates_is2_all, is2_beach_b_all, '--', color = 'gray', lw = 3,label = None)
ax3.plot(dates_planet_all, beach_planet_all, '--',color = 'k', lw = 3,label = None)
ax3.plot(dates_is2_s, beach_top_dist, '^', markersize = 15,lw = 0, color = 'gray',label = 'ICESat-2 (upper boundary)')
ax3.plot(dates_is2_s, beach_bot_dist, 'v', markersize = 15,lw=0, color = 'gray', label = 'ICESat-2 (lower boundary)')
ax3.plot(dates_planet_s, beach_planet_dist, 'o', markersize = 15,lw = 0, color = 'k', label = 'Planet')

plt.ylabel('Distance from June 2019 \n shoreline (m)', fontsize = 23)
plt.title('(c) Ground track 1r', fontsize = 20, loc= 'left', x= 0.01,y = 1.0, pad = -20.5, backgroundcolor = 'white')
plt.yticks(fontsize = 23)
plt.xticks([dt.date(2019, 6, 1), dt.date(2019, 12, 1), dt.date(2020, 6, 1),
            dt.date(2020, 12, 1), dt.date(2021, 6, 1), dt.date(2021, 12, 1), \
                dt.date(2022, 6, 1)], ['June\n2019', 'Dec\n2019', 'June\n2020',\
                                       'Dec\n 2020','June\n2021', 'Dec\n2021',\
                                       'June\n2022'],fontsize = 23, horizontalalignment = 'center')
gray_31 = patches.Rectangle((start_date, -170), ows_19[0]-start_date, 250, color = 'lightgray', zorder = 0)
gray_32 = patches.Rectangle((ows_19[-1], -170), ows_20[0]-ows_19[-1], 250, color = 'lightgray', zorder = 0)
gray_33 = patches.Rectangle((ows_20[-1], -170), ows_21[0]-ows_20[-1], 250, color = 'lightgray', zorder = 0)
gray_34 = patches.Rectangle((ows_21[-1], -170), ows_22[0]-ows_21[-1], 250, color = 'lightgray', zorder = 0, label = 'Ice-on interval')

ax3.add_patch(gray_31)
ax3.add_patch(gray_32)
ax3.add_patch(gray_33)
ax3.add_patch(gray_34)

plt.xlim([start_date, end_date])


plt.legend(fontsize = 25,loc = 'lower left', bbox_to_anchor=(1.25, 0))
plt.ylim([-120, 80])

plt.subplots_adjust(left=.105, right=.982, top = .996, bottom=.07, wspace = 0.264, hspace = 0.315)

#%%difference between is2 lower boundary and planet
bluff_bot_diff  = np.array(bluff_planet_dist) - np.array(bluff_bot_dist)
lake_bot_diff = np.array(lake_planet_dist) - np.array(lake_bot_dist)
beach_bot_diff = np.array(beach_planet_dist) - np.array(beach_bot_dist)

bluff_diff_err = [np.sqrt(b**2 + sigma**2) for b in bluff_errors]
lake_diff_err = [np.sqrt(b**2 + sigma**2) for b in lake_errors]
beach_diff_err = [np.sqrt(b**2 + sigma**2) for b in beach_errors]

bluff_top_diff  = np.array(bluff_planet_dist) - np.array(bluff_top_dist)
lake_top_diff = np.array(lake_planet_dist) - np.array(lake_top_dist)
beach_top_diff = np.array(beach_planet_dist) - np.array(beach_top_dist)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 21:19:59 2022

Last updated Jan 28 2025

@author: Marnie Bryant (m1bryant@ucsd.edu)

This script is for the post-processing of planet-derived shorelines and calculating 
year-to-year shoreline change, as done for the following publication: 
https://doi.org/10.5194/egusphere-2024-1656 . This script is used to generate manuscript
Figures 2 and the base of Figure 3, as well as the estimates reported in Table 3.
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import pyproj
from pyproj import Transformer
from matplotlib import cm as cmm
from cmcrameri import cm
from affine import Affine
import os
import glob
import geopandas as gpd
import pandas as pd
from shapely import wkt
import datetime
import scipy as scp
from scipy import ndimage
import skimage.transform as transform
from shapely import LineString
import SDS_centered_transects
plt.rcParams["font.family"] = "arial"

#%%define functions

def distance_along_track(xs, ys):
    #calculates the along-track distance as the sum of distances between consecutive points
    #xs, ys should both both 1-D arrays or lists of x and y coordinates
    #defeine zero-point of transect
    #returns:
    #ditsance: along-track distance
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

def distance_from_point(x, y, xs, ys):
    return np.array(np.sqrt((xs-x)**2+ (ys - y)**2))

def intersection(m1, b1, m2, b2):
    #get the intersection between two lines given their slopes and intercepts
    #m1: slope of first line (unitless)
    #b1: y-intercept of first lines
    #m2: slope of first line (unitless)
    #b2: y-intercept of first lines
    #returns:
    #x, y (doubles): x and y coordinates of intersection point
    x = -(b1-b2)/(m1-m2)
    y = m1*x + b1
    return x, y

def intersect_line_contour(line, cont):
    #Finds the intersection between a non-linear contour and  a line. If no intersection
    #point is found, the returned coordinates with be NaN
    #line: 2D array with x and y coordinates of line
    #cont: 2D array with x and y coordinates of contour
    #returns:
    #x_i, y_i (doubles): x and y coordinates of intersection point
    
    #extract coordinates from line
    x_l = line[:, 0]
    y_l = line[:, 1]

    #get coefficients(slope and interceot) of line 
    p = np.polyfit(x_l, y_l, 1) 
    m_l = p[0] #slope
    b_l = p[1] #intercept
    #extract coordinates from contour
    x_c = cont[:, 0]
    y_c = cont[:, 1]
    
    #subset around range of line segment with a 10-m buffer
    cont_s = cont[(x_c > np.min(x_l)-10) & (x_c < np.max(x_l)+10)]
    x_s = cont_s[:, 0]
    y_s = cont_s[:, 1]
    #initialize intersection points - nan indicated that an intersection point hasn't been found
    x_i = np.nan
    y_i = np.nan
    
    #loop through subsetted contour points and test for intersection
    for i in range(0, len(x_s)-1):
        #re-initliaze each time we iterate
        x_i = np.nan
        y_i = np.nan
        x = x_s[i:i+2]
        y = y_s[i:i+2]
        
        p2 = np.polyfit(x, y, 1) #line fit to selected countour points
        m_s = p2[0]
        b_s = p2[1]
        
        x_i, y_i = intersection(m_l, b_l, m_s, b_s) #calculate intersection
        #check to see if interection points falls within the countour, If so, we're done
        if ((x_i >= np.min(x)) and (x_i <= np.max(x)) and (y_i >= np.min(y)) and (y_i <= np.max(y))):
            break 
    return x_i, y_i

                        

def resample_along_track(contour, l):
    #resamples provided contour to have points sampled every l meters along-track]
    #contour: 2D array with x and y coordinates
    #l: along-track interval to resample contour to
    #returns: 2D array with coorindates of resampled contour
    
    xs = contour[:, 0]
    ys = contour[:, 1]
    dist = distance_along_track(contour[:, 0], contour[:, 1]) #aet along0track distance
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
        
        p = np.polyfit(x, y, 1)
        x_p = np.linspace(x_e[s], x_e[s+1], len(x))
        y_p = np.polyval(p, x_p)
        x_n.append(np.mean(x_p))
        y_n.append(np.mean(y_p))
   
    return np.transpose(np.array([x_n, y_n]))

def resample_coordinates(contour, xs_i):
   #re-sample the countour at provided x-coordinates
   #contour: 2D array with x and y coordinates
   #xs_i: list of x coordinates to sample to
   #returns: 2D array with coorindates of resampled contour
    xs = contour[:, 0]
    ys = contour[:, 1]
    dist = distance_along_track(contour[:, 0], contour[:, 1])
    y_n = np.interp(xs_i, xs, ys)
    
    return np.transpose(np.array([xs_i, y_n]))



def resample_contour(contour, l, smoothing = None):
    xs = contour[:, 0]
    ys = contour[:, 1]
    dist = distance_along_track(contour[:, 0], contour[:, 1])
    #re-sample to l to definae segement ends:
    n_segs = int(np.floor((np.max(dist))/l))
    segs = (np.linspace(0, n_segs*l, n_segs+1))
    x_e = np.interp(segs, dist, xs)
    y_e = np.interp(segs, dist, ys)
    x_n = [xs[0]]
    y_n = [ys[0]]
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
       # pdb.set_trace()
    return np.transpose(np.array([x_n, y_n]))

def smooth_contour(contour, l):
     #calculates an along-track running mean of x/y positions
     #contour: 2-D array with x and y coordinates
     #l - length of the smoothing window in contour units (i.e. meters)
     #returns: a 2-D array with x and y coordinates of smoothed coordinates
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

        
def along_transect_change(cont1, cont2, transects):
   #calculates the change between two contours along a provded list of transects
   #NOTE: This is written for shorelines that are north-south oriented, with 
   #northwards change being positive
   #cont1: 2D array with x and y coorindates
   #cont2: 2D array with x and y coorindates
   #transects: list of transects to use. Each element of the list should be a 2D array 
   #with x and y coordinates
   #returns:
   #change: 1-D array of change estaimes for each transects
   #xs2, ys2: intesection point of transects with 2nd contour - these are the coordinates 
   #that change estimates will be plotted at
   
   #get intersection points between all transects and contours
   coords_1 = np.array([intersect_line_contour(t, cont1) for t in transects])
   xs1 = coords_1[:, 0]
   ys1 = coords_1[:,1]
   coords_2 = np.array([intersect_line_contour(t, cont2) for t in transects])
   xs2 = coords_2[:, 0]
   ys2 = coords_2[:, 1]
   
   #calculate distance between interection points on the two contours
   change = np.sqrt((xs2-xs1)**2 + (ys2-ys1)**2)
   sign = (ys2-ys1)/np.abs(ys2-ys1) #determine sign of change. Northward change is positive
   return  change*sign, xs2, ys2
    
def get_orientation(coastline):
    ys = coastline[:, 1]
    xs = coastline[:, 0]
    local_slope = [(ys[i+1] - ys[i-1])/(xs[i+1] - xs[i-1]) for i in range(1, len(xs)-1)]
    angle = np.arctan(local_slope)
    return angle       
#%% some helper functions for reading in images
def trim_image(A, X, Y, xs, ys):
    #transformer = pyproj.transformer.Transformer.from_crs(4326, p, always_xy = True)
    #xs, ys = transformer.transform(lons, lats)
    x = X[0, :]
    y = Y[:, 0]
    x_t = x[(x >= xs[0]) & (x <= xs[1])] 
    y_t = y[(y >= ys[0]) & (y <= ys[1])]
 
    x1 = np.where(x == x_t[0])[0][0]
    x2 = np.where(x == x_t[len(x_t)-1])[0][0]
    #pdb.set_trace()
    y2 = np.where(y == y_t[len(y_t)-1])[0][0]
    y1 = np.where(y == y_t[0])[0][0]
    A_small = A[:, y1:y2, x1:x2]
    X_small = X[y1:y2, x1:x2]
    Y_small = Y[y1:y2, x1:x2]
    return A_small, X_small, Y_small

def read_file(image_file, lons= None, lats = None):
    with rasterio.open(image_file) as f:
        T0 = f.transform  # upper-left pixel corner affine transform
        p = (f.crs) #projection (UTM in this case)
        A = f.read()
    n_rows = A.shape[1]
    n_cols = A.shape[2]
# Get affine transform for pixel centres
    T1 = T0 * Affine.translation(0.5, 0.5)

# convert row/column index to eastings and norhtings
#eastings, northings = np.vectorize(rc2en, otypes=[float, float])(rows, cols)
#eastings = cols * T1
#northings = rows* T1
    min_coords = T1*(0, 0)
    max_coords = T1*(n_cols, n_rows)
    x_min = min_coords[0]
    x_max = max_coords[0]
    y_min = min_coords[1]
    y_max = max_coords[1]
    x = np.linspace(x_min, x_max, n_cols)
    y = np.linspace(y_min, y_max, n_rows)
    X, Y = np.meshgrid(x, y)
    if lons != None:
        A, X, Y = trim_image(A, X, Y, lons, lats)
    return A,p, X, Y 

def make_rgb(A, lim = 5000):
    blue = np.float64(A[0, :, :])
    green = np.float64(A[1, :, :])
    red = np.float64(A[2, :, :])
    
    
    #re-assign very bringt (snow/ice) pixels
    blue[np.where(blue > lim)] = lim
    red[np.where(red > lim)] = lim
    green[np.where(green > lim)] = lim
    #contrast enhancement
    red_n = ((red - np.min(red[red >0]))*1/(np.max(red) - np.min(red[red >0])))
    blue_n = ((blue - np.min(blue[blue > 0]))*1/(np.max(blue) - np.min(blue[blue>0])))
    green_n = ((green - np.min(green[green > 0]))*1/(np.max(green) - np.min(green[green > 0])))



    rgb_image = np.stack([red_n, green_n, blue_n], axis = 2).reshape((red.shape[0], red.shape[1], 3))
    return rgb_image        

#%%read image
files = glob.glob('/Users/m1bryant/Documents/Planet Images/IS2_paper_full/*SR_clip.tif')
files.sort()

A0, p, X0, Y0 = read_file(files[15])#15
A, X, Y = trim_image(A0, X0, Y0, [469350, 477700], [7864100, 7865200]) # for full inland view change lower bound to 7863200
rgb = make_rgb(A)
#%%
#load in coastline

cont_2019 = np.genfromtxt('data/coastlines/annual_coastlines/coastline_20190625_1006_ndwi.csv', delimiter = ',', skip_header = 1)
cont_2020 = np.genfromtxt('data/coastlines/annual_coastlines/coastline_20200725_1065_ndwi.csv', delimiter=',', skip_header = 1)
cont_2021 = np.genfromtxt('data/coastlines/annual_coastlines/coastline_20210702_2442_ndwi.csv', delimiter = ',', skip_header = 1)
cont_2022 = np.genfromtxt('data/coastlines/annual_coastlines/coastline_20220701_2439_ndwi.csv', delimiter = ',', skip_header = 1)
#%%
IS2 = gpd.read_file('data/IS2_tracks.shp')
IS2 = IS2.to_crs('EPSG:32605') 
beach = IS2[IS2['gt'] == 20]
lake = IS2[IS2['gt'] == 40]
bluff = IS2[IS2['gt'] == 60]

#%%
cont_2019s = smooth_contour(cont_2019, 30)
cont_2020s = smooth_contour(cont_2020, 30)
cont_2020cs = smooth_contour(cont_2020, 30)
cont_2021s = smooth_contour(cont_2021, 30)
cont_2022s = smooth_contour(cont_2022, 30)

cont_2020r = resample_along_track(cont_2020s, 10)
cont_2020cr = resample_along_track(cont_2020cs, 10)
xs = cont_2020r[:, 0]

cont_2019r = resample_along_track(cont_2019s, 10)
cont_2021r = resample_along_track(cont_2021s, 10)
cont_2022r= resample_along_track(cont_2022s, 10)
#%%
cont_2019r_df = pd.DataFrame({'X': cont_2019r[:, 0], 'Y': cont_2019r[:, 1]})
cont_2020r_df = pd.DataFrame({'X': cont_2020r[:, 0], 'Y': cont_2020r[:, 1]})
cont_2021r_df = pd.DataFrame({'X': cont_2021r[:, 0], 'Y': cont_2021r[:, 1]})
cont_2022r_df = pd.DataFrame({'X': cont_2022r[:, 0], 'Y': cont_2022r[:, 1]})

cont_2019r_df.to_csv('output/annual_coastlines/annual_coastline_2019_6_25_1006_resampled_r.csv')
cont_2020r_df.to_csv('output/annual_coastlines/annual_coastline_2020_07_25_1065_resampled_r.csv')
cont_2021r_df.to_csv('output/annual_coastlines/annual_coastline_2021_07_02_2442_resampled_r.csv')
cont_2022r_df.to_csv('/Users/m1bryant/Documents/Data/IS2_case_study/Zenodo/Planet_shorelines/annual_coastlines/annual_coastline_2022_7_01_2439_resampled_r.csv')
#%%plot all images with coastlines
colors = cm.lajolla((np.linspace(0, 1, 8)))
lons = [469400, 477150]
lats = [7864300, 7865200]


A19, p, X19, Y19 = read_file(files[3], lons = lons, lats = lats)
rgb19 = make_rgb(A19, lim= 3000)

A20, p, X20, Y20 = read_file(files[7], lons = lons, lats = lats)
rgb20 = make_rgb(A20, lim= 3000)

A21a, p, X21a, Y21a = read_file(files[14], lons = lons, lats = lats)
rgb21a = make_rgb(A21a, lim= 3000)

A22, p, X22, Y22 = read_file(files[20], lons = lons, lats = lats)
rgb22 = make_rgb(A22, lim= 3000)

#project bounding coordinates into lat/lon for plotting 
transformer = Transformer.from_crs(32605, 4326, always_xy=True) #for converting from utm zone 5 to lat/lon
transformer_inverse = Transformer.from_crs(4326, 32605, always_xy=True)
#2019 has slightly different coordinated than other images since a manual merge was performed
lons_p, lats_p = transformer.transform(lons, lats)
lon_ticks = [-153.8, -153.75, -153.7, -153.65]
lat_ticks = [70.880, 70.8825, 70.885, 70.8875]
ticks_x, ticks_y = transformer_inverse.transform(lon_ticks, lat_ticks)
ticks_y_s = ticks_y[1:4]
lat_ticks_s = lat_ticks[1:4]
lon_labels = [r'153.80', r'153.75', r'153.70',\
              r'153.65']
lat_labels = [r'70.8825', r'70.8850', r'70.8875']

#2021-2022 have same coorindates
lons_20, lats_20 = transformer.transform([np.min(X20), np.max(X20)],\
                                         [np.min(Y20), np.max(Y20)])
    
#reproject coastlines:
cont_2019r_lons, cont_2019r_lats = transformer.transform(cont_2019r[:, 0], cont_2019r[:,1])  

lat = np.mean(Y19)
f = 1.0/np.cos(lat*np.pi/180)
fig = plt.figure()
ax1 = fig.add_subplot(4,1, 1)
ax1.imshow(rgb19, extent = (np.min(X19), np.max(X19), np.min(Y19),np.max(Y19)))
ax1.plot(cont_2019r[:, 0], cont_2019r[:, 1], c = 'k', lw = 3)
ax1.axes.get_xaxis().set_visible(False)
ax1.plot([469600, 469800], [7864400, 7864400], color = 'white')
plt.annotate('200 m', (469600,7864450), fontsize = 20, color = 'white')
ax1.set_title('25 June 2019', y = .95, pad = -14, fontsize = 25, backgroundcolor = 'white')
plt.annotate(r"Imagery $\copyright$ 2024 Planet Labs Inc.",(475450,7864325), fontsize = 15, color = 'white')
plt.ylabel('Latitude ($^\circ$ N)', fontsize = 20)
plt.ticklabel_format(useOffset=False)
plt.xlim(469400, 477150)
plt.ylim(7864300, 7865200)
plt.yticks(ticks_y_s, lat_labels, fontsize = 20)
plt.annotate('(a)', (0.005, .835), xycoords = 'axes fraction', bbox=dict(facecolor='w', edgecolor='none'), fontsize = 20)




ax2 = fig.add_subplot(4,1, 2)
ax2.imshow(rgb20, extent = (np.min(X20), np.max(X20), np.min(Y20), np.max(Y20)), aspect = 'equal')
ax2.plot(cont_2020r[:, 0], cont_2020r[:,1], c = colors[2], lw = 3)
ax2.axes.get_xaxis().set_visible(False)
ax2.set_title('25 July 2020', y = .95, pad = -14, fontsize = 25, backgroundcolor = 'white')
plt.ylabel('Northings (m)', fontsize = 25)
plt.xlim(469400, 477150)
plt.ylim(7864300, 7865200)
plt.yticks(ticks_y_s, lat_labels, fontsize = 20)
plt.ylabel('Latitude ($^\circ$ N)', fontsize = 20)
plt.annotate(r"Imagery $\copyright$ 2024 Planet Labs Inc.",(475450,7864325), fontsize = 15, color = 'white')
plt.annotate('(b)', (0.005, .84), xycoords = 'axes fraction', bbox=dict(facecolor='w', edgecolor='none'), fontsize = 20)


ax3 = fig.add_subplot(4,1, 3)
ax3.imshow(rgb21a, extent = (np.min(X21a), np.max(X21a), np.min(Y21a), np.max(Y21a)), aspect = 'equal')
ax3.plot(cont_2021r[:, 0], cont_2021r[:,1], c = colors[4], lw = 3)
ax3.set_title('02 July 2021', y = .95, pad = -14, fontsize = 25, backgroundcolor = 'white')
ax3.axes.get_xaxis().set_visible(False)
plt.yticks(ticks_y_s, lat_labels, fontsize =20)
plt.ylabel('Latitude ($^\circ$ N)', fontsize = 20)
plt.annotate(r"Imagery $\copyright$ 2024 Planet Labs Inc.",(475450,7864325), fontsize = 15, color = 'white')
plt.annotate('(c)', (0.005, .84), xycoords = 'axes fraction', bbox=dict(facecolor='w', edgecolor='none'), fontsize = 20)


plt.xlim(469400, 477150)
plt.ylim(7864300, 7865200)


ax5 = fig.add_subplot(4,1, 4)
ax5.imshow(rgb22, extent = (np.min(X22), np.max(X22), np.min(Y22), np.max(Y22)), aspect = 'equal')
ax5.plot(cont_2022r[:, 0], cont_2022r[:,1], c = colors[6], lw = 3)
plt.ylabel('Latitude ($^\circ$ N)', fontsize = 20)
plt.xlabel('Longitude ($^\circ$ W)', fontsize = 20)
ax5.set_title('01 July 2022', y = .95, pad = -14, fontsize = 25, backgroundcolor = 'white')
plt.xlim(469400, 477150)
plt.ylim(7864300, 7865200)
plt.xticks(ticks_x, lon_labels, fontsize = 20)
plt.yticks(ticks_y_s, lat_labels, fontsize = 20)
plt.annotate(r"Imagery $\copyright$ 2024 Planet Labs Inc.",(475450,7864325), fontsize = 15, color = 'white')
plt.annotate('(d)', (0.005, .84), xycoords = 'axes fraction', bbox=dict(facecolor='w', edgecolor='none'), fontsize = 20)

plt.subplots_adjust(left = 0.151, bottom = 0.06, right = 0.995, top=1, wspace = 0.2, hspace = 0)


#%%cross-shore shoreline change
baseline = resample_contour(smooth_contour(cont_2020, 60), 10)
angles = get_orientation(baseline)*180/np.pi
baseline_s = baseline[1:len(baseline)-1, :]
transects = [SDS_centered_transects.create_transect((baseline_s[i, 0], baseline_s[i, 1]), -1*angles[i], 200) for i in range(0,len(angles))]
plt.figure()
plt.plot(cont_2019r[:, 0], cont_2019r[:, 1], c = 'k',label = '2019')
plt.plot(cont_2020r[:, 0], cont_2020r[:, 1], c = colors[2], label = '2020')
plt.plot(cont_2021r[:, 0], cont_2021r[:, 1], c = colors[4], label = '2021')
plt.plot(cont_2022r[:, 0], cont_2022r[:, 1],c =colors[6], label = '2022')
plt.plot(baseline[:, 0], baseline[:, 1], c= 'b', label = 'Baseline (2020)')
dots = []
for i in range(0, len(transects)):
    t =transects[i]
    b1 = baseline[i, :]
    b2 = baseline[i+2, :]
    v1 = [t[-1, 0]-t[0, 0], t[-1, 1]-t[0, 1]] 
    v2 = [b2[0]-b1[0], b2[1]-b1[1]]
    dot = np.dot(v1, v2)
    dots.append(dot)

    plt.plot(t[:, 0], t[:, 1], c = 'b', label = None)
plt.legend()
plt.xlabel('Eastings')
plt.ylabel('Northings')
plt.gca().set_aspect('equal')
#%%
change19, x, y = along_transect_change(cont_2019s, cont_2020r, transects)
change20, x, y = along_transect_change(cont_2020r, cont_2021s, transects)
change21, x, y = along_transect_change(cont_2021s, cont_2022s, transects)
#%%
start_list = [t[0] for t in transects]
end_list = [t[-1] for t in transects]
lines = []
for t in transects:
    points = [(x, y) for x, y in t]
    line = LineString(points)
    lines.append(line)
    #%%
transects_dict = {'id': np.arange(1, len(transects) +1), 'Geometry': lines}
transects_df = pd.DataFrame(transects_dict)
transects_gdf = gpd.GeoDataFrame(transects_df, geometry = 'Geometry', crs = 'EPSG:32605')
#%%

transects_gdf.to_file('output/transects_2021_10m.json', driver = 'GeoJSON')
#%%
baseline_df = pd.DataFrame({'X': baseline[:, 0], 'Y': baseline[:, 1]})
baseline_df.to_csv('output/2021_baseline.csv')
#%%
plt.figure()
for t in transects:
    plt.plot(t[:, 0], t[:, 1], c = 'k', lw = 2)
for l in lines:
    plt.plot(l.coords.xy[0], l.coords.xy[1], c = 'r', lw = 1)
#%%compare spatial ditribution of erosion between years - 2 panel version
xs = baseline_s[:,0]
ys = baseline_s[:,1]
colors = cm.lajolla((np.linspace(0, 1, 8)))

A0, p, X0, Y0 = read_file(files[7])
A, X, Y = trim_image(A0, X0, Y0, [469350, 477150], [7864200, 7865200]) # for full inland view change lower bound to 7863200
rgb = make_rgb(A, lim= 3000)

sigma = 3.1

bluff_bounds = (469350, 471450)
headlands_bounds = (471450, 474470)
beach_bounds = (474470,477150)

mask_1 = (471850, 471960)
mask_2 = (474470, 475200)

fig = plt.figure()

ax = fig.add_subplot(2, 1, 1)
ax.imshow(rgb, extent = (np.min(X), np.max(X), np.min(Y), np.max(Y)))
ax.plot(cont_2019r[:, 0], cont_2019r[:, 1], c = 'k')
ax.plot(cont_2020r[:, 0], cont_2020r[:, 1], c = colors[2])
ax.plot(cont_2021r[:, 0], cont_2021r[:, 1], c = colors[4])
ax.plot(cont_2022r[:, 0], cont_2022r[:, 1], c = colors[6])
plt.ylabel('Latitude ($^\circ$ N)', fontsize = 20)

plt.annotate(r"Imagery $\copyright$ 2024 Planet Labs Inc.",(475500,7864325), fontsize = 15, color = 'white')

plt.xticks(ticks_x,visible = False)
plt.yticks(ticks_y_s, lat_labels, fontsize = 20)

lake.plot(ax = ax, color = 'y', zorder = 5, label = 'ICESat-2 ground tracks')
beach.plot(ax = ax, color = 'y', zorder = 5)
bluff.plot(ax = ax, color = 'y', zorder = 5)

ax.vlines(bluff_bounds[1],7864200, 7865200,  linestyle ='--', color = 'b' , linewidth = 2, label = 'Region boundaries') #old color: [(200/250), (113/250), (26/250)]
ax.vlines(headlands_bounds[1],7864200, 7865200,  linestyle ='--',color = 'b', linewidth = 2)
ax.vlines(beach_bounds[1],7864200, 7865200, linestyle ='--', color = 'b', linewidth = 2)

plt.ylim(7864300, 7865200)
plt.xlim([np.min(X), np.max(X)])
ax.plot([469600, 469800], [7864400, 7864400], color = 'white')
plt.annotate('200 m', (469600,7864450), fontsize = 15, color = 'white')
plt.annotate('(a)', (0.005, .85), xycoords = 'axes fraction', bbox=dict(facecolor='w', edgecolor='none'), fontsize = 20, zorder = 6)

ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(xs, change19, c = colors[2])
ax2.plot(xs, change20, c = colors[4])
ax2.plot(xs, change21, c = colors[6])

plt.xlabel('Longitude ($^\circ$ W)', fontsize =20)
plt.ylabel('Shoreline Change (m)', fontsize = 20)

ax2.hlines(0, np.min(X), np.max(X), 'k',linestyle = '-')
ax2.hlines(sigma, np.min(X), np.max(X), 'k',linestyle = '--', linewidth = 1, label = r'Uncertainty (1 $\sigma$)')
ax2.hlines(-1*sigma, np.min(X), np.max(X), 'k',linestyle = '--', linewidth = 1)
plt.xticks(ticks_x, lon_labels, fontsize = 20)
plt.yticks(fontsize = 20)

gray_1 = patches.Rectangle((mask_1[0], -100), mask_1[1]-mask_1[0], 150, color = 'lightgray', zorder = 0, label = 'Masked data')

gray_2 = patches.Rectangle((mask_2[0], -100), mask_2[1]-mask_2[0], 150, color = 'lightgray', zorder = 0)


ax2.add_patch(gray_1)
ax2.add_patch(gray_2)
ax2.vlines(bluff_bounds[1],-100, 50,  linestyle = '--', color = 'b', linewidth = 2, label = 'Region boundaries')


ax2.vlines(headlands_bounds[1],-100, 50,  linestyle ='--', color = 'b', linewidth = 2)


ax2.vlines(beach_bounds[1],-100, 50, linestyle ='--', color = 'b', linewidth = 2)

plt.xlim([np.min(X), np.max(X)])
plt.ylim([-100, 50])
plt.annotate('(b)', (0, .925), xycoords = 'axes fraction',fontsize = 20)

plt.subplots_adjust(left=0.121, bottom = .08, right = 0.995, top = 1, wspace = .2, hspace = 0)



#%%
#mask out invalid areas
change19_m = np.array(change19)
change19_m[((xs >= 471850) & (xs <=471960)) | ((xs >= 474470) & (xs <=475200)) | (xs > 477700)] = np.nan

change20_m = np.array(change20)
change20_m[((xs >= 471850) & (xs <=471960)) | ((xs >= 474470) & (xs <=475200)) | (xs > 477700)] = np.nan

change21_m = np.array(change21)
change21_m[((xs >= 471850) & (xs <=471960)) | ((xs >= 474470) & (xs <=475200)) | (xs > 477700)] = np.nan

change_total = np.concatenate((change19_m, change20_m, change21_m))

#%% calculate schange for 3 regions
region_1_19 = change19_m[(xs >= 469350) & (xs <= 471450)]
region_1_20 = change20_m[(xs >= 469350) & (xs <= 471450)]
region_1_21 = change21_m[(xs >= 469350) & (xs <= 471450)]

region_1_total = np.concatenate((region_1_19, region_1_20, region_1_21))

region_2_19 = change19_m[(xs >= 471450) & (xs <= 474470)]
region_2_20 = change20_m[(xs >= 471450) & (xs <= 474470)]
region_2_21 = change21_m[(xs >= 471450) & (xs <= 474470)]

region_2_total = np.concatenate((region_2_19, region_2_20, region_2_21))

region_3_19 = change19_m[(xs >= 475200) & (xs <= 477150)]
region_3_20 = change20_m[(xs >= 475200) & (xs <= 477150)]
region_3_21 = change21_m[(xs >= 475200) & (xs <= 477150)]

region_3_total = np.concatenate((region_3_19, region_3_20, region_3_21))
#%%save shoreline change rates
data_dir = 'output/shoreline_change/'

change_df = pd.DataFrame({'X': xs, 'change_19': change19_m, 'change_20': change20_m, 'change_21': change21_m})
change_df.to_csv(data_dir + 'shoreline_change_full.csv')
region_1_bounds = [469350, 471450]
region_2_bounds = [471450, 474470]
region_3_bounds = [475200, 477150]

region_bounds = pd.DataFrame({'Region': [1, 2, 3], 'Start': [region_1_bounds[0], region_2_bounds[0], region_3_bounds[0]],\
                              'End': [region_1_bounds[1], region_2_bounds[1], region_3_bounds[1]]})
region_bounds.to_csv(data_dir + 'region_bounds.csv')

#%%compare lake to the reest of the headlands

lake_19 = change19_m[(xs >= 472700) & (xs <= 472850)]
lake_20 = change20_m[(xs >= 472700) & (xs <= 472850)]
lake_21 = change21_m[(xs >= 472700) & (xs <= 472850)]

plt.figure()
plt.hist(region_2_19, bins = 20)
plt.axvline(np.mean(lake_19), 0, 40, c = 'k')
p_19 = np.sum(region_2_19[~np.isnan(region_2_19)]< np.mean(lake_19))/len(region_2_19[~np.isnan(region_2_19)])
plt.annotate('lake mean = ' + "{:.1f}".format((np.mean(lake_19)))+ ' m', (.2, .8), xycoords = 'figure fraction',\
             fontsize = 12)
plt.annotate('percentile = ' + "{:.0f}".format(p_19*100)+ '%', (.2, .75), xycoords = 'figure fraction',\
                 fontsize = 12)
plt.title('2019')

plt.figure()
plt.hist(region_2_20, bins = 20)
plt.axvline(np.mean(lake_20),0, 40,c= 'k')
p_20 = np.sum(region_2_20[~np.isnan(region_2_20)]< np.mean(lake_20))/len(region_2_20[~np.isnan(region_2_20)])

plt.annotate('lake mean = ' + "{:.1f}".format((np.mean(lake_20)))+ ' m', (.2, .8), xycoords = 'figure fraction',\
             fontsize = 12)
plt.annotate('percentile = ' + "{:.0f}".format(p_20*100)+ '%', (.2, .75), xycoords = 'figure fraction',\
                 fontsize = 12)
plt.title('2020')

plt.figure()
plt.hist(region_2_21, bins = 20)
plt.axvline(np.mean(lake_21),0,40,c = 'k')
plt.annotate('lake mean = ' + "{:.1f}".format((np.mean(lake_21)))+ ' m', (.2, .8), xycoords = 'figure fraction',\
             fontsize = 12)
p_21 = np.sum(region_2_21[~np.isnan(region_2_21)]< np.mean(lake_21))/len(region_2_21[~np.isnan(region_2_21)])
plt.annotate('percentile = ' + "{:.0f}".format(p_21*100)+ '%', (.2, .75), xycoords = 'figure fraction',\
                 fontsize = 12)

plt.title('2021')










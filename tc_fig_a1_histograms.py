#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 20:39:17 2025

@author: Marnie Bryant m1bryant(m1bryant@ucsd.edu)

This script reproduces Figure A1 from the following manuscript: https://doi.org/10.5194/egusphere-2024-1656
which shows the distribution of normalized differenced water index (NDWI) for 4 planet images,
along with the threshold used to distinguish between land and water pixels

"""
#%%
import rasterio
from rasterio.transform import from_gcps
from rasterio.control import GroundControlPoint as GCP
import numpy as np
import pyproj
from affine import Affine
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

#%%

def trim_image(A, X, Y, xs, ys):
    #A: 3D array containing image data
    #X, Y: 2D arrays of full image coordinates
   #xs, ys: tuples containing desired bounds for trimming
    x = X[0, :]
    y = Y[:, 0]
    x_t = x[(x >= xs[0]) & (x <= xs[1])] 
    y_t = y[(y >= ys[0]) & (y <= ys[1])]
 
    x1 = np.where(x == x_t[0])[0][0]
    x2 = np.where(x == x_t[len(x_t)-1])[0][0]
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

def get_ndwi(A):
    green = np.float64(A[1, :, :])
    ir = np.float64(A[3, :, :])
    ndwi = (ir-green)/(ir+green)
    return ndwi

def otsu_thresholding(img, n_bins):
    hist_n, bins= np.histogram(img[~np.isnan(img)], n_bins, density = True) #normalized histogram
    hist, bins= np.histogram(img[~np.isnan(img)], n_bins) #histogram with counts
    t = 0 #arbitrary starting threshold
    var = 1e-6 #arbitrary small starting variance
    var_list = []
    for b in np.arange(len(bins)-1):
        t_new = bins[b+1] #right edge of highest bin (our potential threshold value)
        hist_1 = hist_n[0:b]
        bins_1 = bins[1:b+1]
        bins_2 = bins[b+1:len(bins)]
        hist_2 = hist_n[b:len(hist_n)]
        q1 = np.sum(hist_1) #sum of probabilities below threshold
        q2 = np.sum(hist_2) #sum of probabilities above threshold
        #get means and variancesof the two catgories
        mu_1 = np.sum(hist_1*bins_1/q1)
        mu_2 = np.sum(hist_2*bins_2/q2)
    
        #calculate weighted between-class variance (we want to maximize this value)
        var_t = (q1*q2)*(mu_1-mu_2)**2
        var_list.append(var_t)
        if var_t > var:
            t = t_new
            var = var_t 
    return t
#%%
data_dir = '/Users/m1bryant/Documents/Planet Images/IS2_paper_full'
os.chdir(data_dir)
files = glob.glob('*SR_clip.tif')
files.sort()

lons = [469400, 477700]

lats = [7864300, 7865200]

fig_hist = plt.figure(figsize = (16, 9))

ax1 = fig_hist.add_subplot(2, 2, 1)

A,p, X, Y = read_file(files[3], lons, lats) #25 June 2019

image = get_ndwi(A)

t = otsu_thresholding(image, 100)
ax1.hist(image.flatten(), bins = 100, density = True)
plt.xlabel('NDWI', fontsize = 25)
plt.yticks(fontsize = 25)
plt.xlim([-1, 1])
plt.xticks([-.8, -.4, 0, .4, .8],fontsize = 25)
plt.ylabel('Normalized frequency', fontsize = 25)
plt.axvline(t, c = 'k', ls = '--')
plt.annotate('(a)', (0, .93), xycoords = 'axes fraction', fontsize = 20)

ax2 = fig_hist.add_subplot(2, 2, 2)
A,p, X, Y = read_file(files[7], lons, lats)  #25 July 2020,


image = get_ndwi(A)

t = otsu_thresholding(image, 100)
ax2.hist(image.flatten(), bins = 100, density= True)
plt.xlabel('NDWI', fontsize = 25)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
plt.ylabel('Normalized frequency', fontsize = 25)
plt.axvline(t, c = 'k', ls = '--')
plt.annotate('(b)', (0, .93), xycoords = 'axes fraction', fontsize = 20)
plt.xticks([-.8, -.4, 0, .4, .8],fontsize = 25)
plt.yticks(fontsize = 25)
plt.xlim([-1, 1])
plt.xticks([-.8, -.4, 0, .4, .8],fontsize = 25)


ax3 = fig_hist.add_subplot(2, 2, 3)
A,p, X, Y = read_file(files[14], lons, lats) #02 July 2021


image = get_ndwi(A)

t = otsu_thresholding(image, 100)
ax3.hist(image.flatten(), bins = 100, density = True)
plt.xlabel('NDWI', fontsize = 25)
plt.yticks(fontsize = 25)
plt.ylabel('Normalized frequency', fontsize = 25)
plt.axvline(t, c = 'k', ls = '--')
#plt.title('2021-07-02', fontsize  =15)
plt.annotate('(c)', (0, .93), xycoords = 'axes fraction', fontsize = 20)
plt.xlim([-1, 1])
plt.xticks([-.8, -.4, 0, .4, .8],fontsize = 25)


ax4 = fig_hist.add_subplot(2, 2, 4)
A,p, X, Y = read_file(files[20], lons, lats) #01 July 2022

image = get_ndwi(A)

t = otsu_thresholding(image, 100)
ax4.hist(image.flatten(), bins = 100, density = True)
plt.xlabel('NDWI', fontsize = 25)
plt.yticks(fontsize = 25)
plt.xlim([-1, 1])
plt.xticks([-.8, -.4, 0, .4, .8],fontsize = 25)
plt.ylabel('Normalized frequency', fontsize = 25)
plt.axvline(t, c = 'k', ls = '--')
plt.annotate('(d)', (0, .93), xycoords = 'axes fraction', fontsize = 20)

plt.subplots_adjust(left = .12, bottom = .085, right = 0.92, top = 0.964, wspace=.262, hspace=0.22)
    

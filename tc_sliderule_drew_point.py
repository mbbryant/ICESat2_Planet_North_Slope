#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 16:30:11 2023

Last updated Jan 29 2025

@author: Marnie Bryant (m1bryant@ucsd.edu)
"""


import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sliderule import icesat2
import datetime as dt
import time
import numpy as np
from pyproj import Transformer
from matplotlib import cm
import matplotlib.patheffects as pe
#%%
icesat2.init("slideruleearth.io", verbose=True)
#%%
region_lake = [{"lon": -153.85, "lat": 70.885}, #track 2
          {"lon": -153.63, "lat": 70.885},
          {"lon": -153.63, "lat": 70.887},
          {"lon": -153.85, "lat": 70.887},
          {"lon": -153.85, "lat": 70.885}]

region_beach = [{"lon": -153.85, "lat": 70.8815}, #track 1
          {"lon": -153.63, "lat": 70.8815},
          {"lon": -153.63, "lat": 70.8836},
          {"lon": -153.85, "lat": 70.8836},
          {"lon": -153.85, "lat": 70.815}]

region_bluff = [{"lon": -153.85, "lat": 70.883}, #track 3
          {"lon": -153.63, "lat": 70.883},
          {"lon": -153.63, "lat": 70.885},
          {"lon": -153.85, "lat": 70.885},
          {"lon": -153.85, "lat": 70.83}]


#%%
def subset_atl03(region, track):
    parms_ATL03 = {
        "poly": region,
        "srt" : 4,
       "atl03_geo_fields": ["sigma_across", "sigma_along", "sigma_lat", "sigma_lon"]
        }
    atl03_sr = icesat2.atl03sp(parms_ATL03).to_crs('EPSG:32605') 
    times = atl03_sr.index
    dates = [dt.date(t.year, t.month, t.day) for t in times]
    atl03_sr['date'] = dates
    atl03_sr_f = atl03_sr[(atl03_sr['track'] == track) & (atl03_sr['pair'] == 1) & (atl03_sr['rgt']==137)]
    return atl03_sr_f

#%%
atl03_sr_f_bluff = subset_atl03(region_bluff, 3)
atl03_sr_f_lake = subset_atl03(region_lake, 2)
atl03_sr_f_beach = subset_atl03(region_beach, 1)
#%%save data
data_dir = 'output/ICESat_2'
atl03_sr_f_bluff.to_csv(data_dir+'/' + 'atl03_bluff.csv', index = False)
atl03_sr_f_lake.to_csv(data_dir+'/' + 'atl03_lake.csv', index = False)
atl03_sr_f_beach.to_csv(data_dir+'/' + 'atl03_beach.csv', index = False)

#%%
date_list = [dt.date(d) for d in [(2019, 4,7), (2020,1,4), (2021, 7,2), \
                                    (2021, 12, 31)]]


#%%
parms_a = {
    "poly": region_bluff,
    "srt": 4,
    "cnf": 4,
    "ats": 1,
    "cnt": 5,
    "len": 10,
    "res": 2,
    "maxi": 5,
    "H_min_win": 0.8,
    "atl03_geo_fields": ["sigma_across", "sigma_along", "sigma_lat", "sigma_lon"]

}

parms_a["atl03_geo_fields"] = ["ref_azimuth", "ref_elev"]

atl06_sr_a = icesat2.atl06p(parms_a).to_crs('EPSG:32605') 

#%%
atl06_sr_bluff = atl06_sr_a[(atl06_sr_a['gt'] == 60) & (atl06_sr_a['rgt']==137)]# gt = 60 for bluff, 20 for beach

times = atl06_sr_bluff.index
dates = [dt.date(int(t.year), int(t.month), int(t.day)) for t in times]

atl06_sr_bluff['date'] = dates


atl06_sr_lake = atl06_sr_a[(atl06_sr_a['gt'] == 40) & (atl06_sr_a['rgt']==137)]# gt = 60 for bluff, 20 for beach

times = atl06_sr_lake.index
dates = [dt.date(int(t.year), int(t.month), int(t.day)) for t in times]
atl06_sr_lake['date'] = dates

atl06_sr_beach = atl06_sr_a[(atl06_sr_a['gt'] == 20) & (atl06_sr_a['rgt']==137)]# gt = 60 for bluff, 20 for beach

times = atl06_sr_beach.index
dates = [dt.date(int(t.year), int(t.month), int(t.day)) for t in times]
atl06_sr_beach['date'] = dates
#%%add geolocation error
def get_uncertainties(df):
    total_error = [4.2, 4.8, 2.8, 3.3, 2.8, 2.5] #positional uncertainites from Lutchke et al
    error = [total_error[s-1] for s in df['spot'].values]
    return error
#%%
atl06_sr_bluff['geo_error'] = get_uncertainties(atl06_sr_bluff)
atl06_sr_beach['geo_error'] = get_uncertainties(atl06_sr_beach)
atl06_sr_lake['geo_error'] = get_uncertainties(atl06_sr_bluff)

#%%
atl06_sr_bluff.to_csv('ouput/atl06_sr_gt3r.csv', index = False)
atl06_sr_lake.to_csv('ouput/atl06_sr_gt2r.csv', index = False)
atl06_sr_beach.to_csv('ouput/atl06_sr_gt1r.csv', index = False)




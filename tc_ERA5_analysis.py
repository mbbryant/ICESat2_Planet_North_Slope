#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 20:52:48 2023

Last updated Jan 30 2025

@author: Marnie Bryant (m1bryant@ucsd.edu)

"""

import netCDF4 as nc
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from matplotlib import cm
from scipy.stats import lognorm
import scipy as sp
import pdb

#%%

start = dt.datetime(1900, 1, 1)

met_data = nc.Dataset('data/era_5_air_t_ocean_t_waves_sic_2019_2022.nc')
met_t = met_data['time'][:]
met_dates = np.array([start + dt.timedelta(hours = int(ti)) for ti in met_t])
met_months = [d.date() for d in met_dates]
met_days = [d.date() for d in met_dates]
met_hours = [d.hour for d in met_dates]


swh = met_data['swh'][:, 0,0]
temp = met_data['t2m'][:, 0, 0]
sst = met_data['sst'][:, 0, 0]
ice_con = met_data['siconc'][:,0, 0]

met_dataframe = pd.DataFrame({'Date': met_days, 'Hour': met_hours, 'swh': swh, \
                               'air_temp': temp, 'sst': sst, 'sic':ice_con}) #'wave_dir': wave_dir,


met_19 = met_dataframe[met_dataframe['Date'] <= dt.date(2019, 12, 31)]    
met_20 = met_dataframe[(met_dataframe['Date'] > dt.date(2019, 12, 31)) & \
                       (met_dataframe['Date'] <= dt.date(2020, 12, 31))] 
met_21 = met_dataframe[(met_dataframe['Date'] > dt.date(2020, 12, 31)) & \
                       (met_dataframe['Date'] <= dt.date(2021, 12, 31))]  
met_21a = met_dataframe[(met_dataframe['Date'] > dt.date(2020, 12, 31)) & \
                       (met_dataframe['Date'] <= dt.date(2021, 10, 4))]
met_21b = met_dataframe[(met_dataframe['Date'] > dt.date(2021, 10, 4)) & \
                       (met_dataframe['Date'] <= dt.date(2021, 12, 31))]
met_22 = met_dataframe[(met_dataframe['Date'] > dt.date(2021, 12, 31)) & \
                       (met_dataframe['Date'] <= dt.date(2022, 12, 31))]
  
met_daily_19 = met_19.groupby(['Date']).mean()
met_daily_20 = met_20.groupby(['Date']).mean()
met_daily_21 = met_21.groupby(['Date']).mean()
met_daily_22 = met_22.groupby(['Date']).mean()

#%%year-round ice data
ice = nc.Dataset('/Users/m1bryant/Documents/Data/IS2_case_study/era_5/era_5_sic_2019_2022.nc')
ice_t = ice['time'][:]
ice_dates = np.array([start + dt.timedelta(hours = int(ti)) for ti in ice_t])

#%%sea ice year-to-year comparison
colors = cm.viridis(((np.linspace(0, 1, 8))))

fig = plt.figure()



plt.show()
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(met_daily_19.index, met_daily_19['sic'], c = colors[3], label = '2019')
ax1.plot(met_daily_19.index, met_daily_20['sic'],c = colors[5], label = '2020')
ax1.plot(met_daily_19.index, met_daily_21['sic'], c = colors[6], label = '2021')

ax1.axhline(.15, 0, 1, linestyle = '--', color = 'k')
labels = ['May', 'Jun', 'Jul', 'Aug', 'Sep','Oct', 'Nov', 'Dec']
plt.xticks([dt.date(2019, i, 1) for i in range(5, 13)], labels) #for month-labeled plot - comment if you want to read the dates
plt.xlabel('Month', fontsize = 15)
plt.ylabel('Sea Ice Fraction', fontsize=15)
plt.legend(fontsize = 15)
plt.title('Daily Mean Sea Ice Concentration')
#%% 
ows_19_15 = met_daily_19[met_daily_19['sic'] <= 0.15]

ows_20_15 = met_daily_20[met_daily_20['sic'] <= 0.15]


ows_21_15 = met_daily_21[met_daily_21['sic'] <= 0.15]

ows_22_15 = met_daily_22[met_daily_22['sic'] <= 0.15]
#%%export list of open water days
def dates_to_csv(ows, fname):
    date_strings = [str(d) for d in ows.index.values]
    np.savetxt(fname, date_strings, delimiter=" ",fmt="%s")

dates_to_csv(ows_19_15, '/Users/m1bryant/Documents/Data/IS2_case_study/era_5/owd_2019.csv')   
dates_to_csv(ows_20_15, '/Users/m1bryant/Documents/Data/IS2_case_study/era_5/owd_2020.csv')    
dates_to_csv(ows_21_15, '/Users/m1bryant/Documents/Data/IS2_case_study/era_5/owd_2021.csv')    
dates_to_csv(ows_22_15, '/Users/m1bryant/Documents/Data/IS2_case_study/era_5/owd_2022.csv')


#%%
def cumulative_wave_power(dataframe,date_range):
    
   
    date_start = date_range[0]
    date_end = date_range[1]
    dataframe_f = dataframe.loc[(dataframe['Date'] >= date_start) & (dataframe['Date'] <= date_end)]
    dataframe_f.loc[np.isnan(dataframe['swh']), 'swh'] = 0
    
    int_sum = 0
    integral = []
    date_list = set(dataframe_f.Date.values)

    for d in date_list:
        data = dataframe_f[dataframe_f['Date'] == d]        
        daily_int = sp.integrate.trapezoid(data['swh'], dx = 1/24)
        int_sum = int_sum + daily_int
        integral.append(int_sum)
    return integral
        
def cumulative_wave_energy(dataframe,date_range):
    
    date_start = date_range[0]
    date_end = date_range[1]
    dataframe_f = dataframe.loc[(dataframe['Date'] >= date_start) & (dataframe['Date'] <= date_end)]
    dataframe_f.loc[np.isnan(dataframe['swh']), 'swh'] = 0

    
    int_sum = 0
    integral = []
    date_list = set(dataframe_f.Date.values)
    for d in date_list:
        data = dataframe_f[dataframe_f['Date'] == d]

        daily_int = sp.integrate.trapezoid(data['swh']**2, dx = 1/24)
        int_sum = int_sum + daily_int
        integral.append(int_sum)
    return integral

    
#%%
wave_power_19_15 = cumulative_wave_power(met_dataframe, (dt.date(2019, 1, 1), dt.date(2019, 12, 31)), sic_filter = 0.15)
wave_energy_19_15 = cumulative_wave_energy(met_dataframe, (dt.date(2019, 1, 1), dt.date(2019, 12, 31)), sic_filter = 0.15)

wave_power_20_15 = cumulative_wave_power(met_dataframe, (dt.date(2020, 1, 1), dt.date(2020, 12, 31)), sic_filter = 0.15)
wave_energy_20_15 = cumulative_wave_energy(met_dataframe, (dt.date(2020, 1, 1), dt.date(2020, 12, 31)), sic_filter = 0.15)


wave_power_21_15 = cumulative_wave_power(met_dataframe, (dt.date(2021, 1, 1), dt.date(2021, 12, 31)), sic_filter = 0.15)
wave_energy_21_15 = cumulative_wave_power(met_dataframe, (dt.date(2021, 1, 1), dt.date(2021, 12, 31)), sic_filter = 0.15)


#%% extreme event

#%%

plt.figure()
plt.hist(swh, bins  = 50, density = True)
#plt.plot(swh, pdf)
plt.xlabel('Significant Wave Height (m)', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)
plt.vlines(1.4, 0, 600, 'k', '--')
plt.ylim(0, 2)
plt.title('Hourly Significant Wave Height', fontsize = 20)


thresh = 1.4

met_extreme = met_dataframe[met_dataframe['swh'] >= thresh ]
met_extreme_19 = met_19[met_19['swh'] >= thresh]

met_extreme_20 = met_20[met_20['swh'] >= thresh]
met_extreme_21 = met_21[met_21['swh'] >= thresh]


met_extreme_daily_count_19 = met_extreme_19.groupby(['Date']).count()
met_extreme_daily_count_20 = met_extreme_20.groupby(['Date']).count()
met_extreme_daily_count_21 = met_extreme_21.groupby(['Date']).count()


met_extreme_daily_max = met_extreme.groupby(['Date']).max()

#%%aggregate data
daily_mean_2m_t = np.mean(temp.reshape(-1, 24), axis = 1)
daily_mean_sst = np.mean(sst.reshape(-1, 24), axis = 1)
daily_dates = met_dates[0::24]
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(daily_dates, daily_mean_2m_t - 273.15, 'orange', label = '2m Air Temp')
ax1.plot(daily_dates, daily_mean_sst - 273.15, 'r', label = 'Sea Surface Temp')
plt.ylabel('Temperature (C)', fontsize = 15)
plt.xlabel('Date', fontsize = 15)
plt.legend(fontsize = 15)
plt.title('ERA-5 daily mean reanalysis')   
#%%
fig = plt.figure()
plt.plot(met_daily_19.index, met_daily_19['air_temp']- 273.15, c = colors[3], label = '2019 air temp')
plt.plot(met_daily_19.index, met_daily_20['air_temp']- 273.15, c = colors[5], label = '2020 air temp')
plt.plot(met_daily_19.index, met_daily_21['air_temp']- 273.15, c = colors[6], label = '2021 air temp')
plt.hlines(0, dt.date(2019, 5, 1), dt.date(2019, 11, 30), color  = 'k')
plt.xlim([dt.date(2019, 5, 1), dt.date(2019, 11, 30)])
plt.ylabel('Temperature (C)', fontsize = 15)
plt.ylim([-10, 10])
plt.xlabel('Month', fontsize = 15)
plt.legend(fontsize = 15)
plt.title('Daily Mean Air Temperature', fontsize = 15) 
#%%
fig = plt.figure()
plt.plot(met_daily_19.index, met_daily_19['sst']- 273.15, c = colors[3], label = '2019 sea surface temp')
plt.plot(met_daily_19.index, met_daily_20['sst']- 273.15, c = colors[5], label = '2020 sea surface temp')
plt.plot(met_daily_19.index, met_daily_21['sst']- 273.15, c = colors[6], label = '2021 sea surface temp')
plt.hlines(0, dt.date(2019, 5, 1), dt.date(2019, 11, 30), color  = 'k')
plt.xlim([dt.date(2019, 5, 1), dt.date(2019, 11, 30)])
plt.ylabel('Temperature (C)', fontsize = 15)
plt.ylim([-10, 10])
plt.xlabel('Month', fontsize = 15)
plt.legend(fontsize = 15)
plt.title('Daily Mean Sea Surface Temperature', fontsize = 15) 
#%%mean temperarures
#air: full time period
air_mean_19 = np.mean(met_daily_19[(met_daily_19.index <= dt.date(2019, 10,31)) &\
                                   (met_daily_19.index >= dt.date(2019, 6,1))]\
                      ['air_temp'])- 273.15
air_mean_20 = np.mean(met_daily_20[(met_daily_20.index <= dt.date(2020, 10,31)) &\
                                   (met_daily_20.index >= dt.date(2020, 6,1))]\
                      ['air_temp'])- 273.15
air_mean_21 = np.mean(met_daily_21[(met_daily_21.index <= dt.date(2021, 10,31)) &\
                                   (met_daily_21.index >= dt.date(2021, 6,1))]\
                      ['air_temp'])- 273.15


#ocean:open water season
sst_mean_19 = np.mean(ows_19_15['sst'])- 273.15
sst_mean_20 = np.mean(ows_20_15['sst'])- 273.15
sst_mean_21 = np.mean(ows_21_15['sst'])- 273.15

#%%
def get_addt(temp):
    addt = []
    t_sum = 0
    for t in temp:
        if t> 0:
            t_sum = t_sum + t
        addt.append(t_sum)
    return np.array(addt)


#%%
addt_air_19 = get_addt(met_daily_19['air_temp']- 273.15)
addt_air_20 = get_addt(met_daily_20['air_temp']- 273.15)
addt_air_21 = get_addt(met_daily_21['air_temp']- 273.15)

addt_sst_19 = get_addt(ows_19_15['sst']- 273.15)
addt_sst_20 = get_addt(ows_20_15['sst']- 273.15)
addt_sst_21 = get_addt(ows_21_15['sst']- 273.15)

#%%
plt.figure()
plt.plot(met_daily_19.index, addt_air_19, c = colors[3], label = '2019')

plt.plot(met_daily_19.index, addt_air_20, c = colors[5], label = '2020')
plt.plot(met_daily_19.index, addt_air_21, c = colors[6], label = '2021')

plt.xlabel('Month', fontsize = 15)
plt.ylabel('Accumulated degree days of Thaw (C days)', fontsize = 15)
plt.title('Accumulated Degree Days: 2m Air Temperature', fontsize = 20)
plt.legend(fontsize = 15)


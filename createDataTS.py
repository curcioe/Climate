#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports

import matplotlib.pyplot as plt # for plotting
from matplotlib import colors # for color normalization/visualization
import numpy as np
import pandas as pd
from numpy.fft import fft, ifft # for fast fourier transform
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from tslearn.clustering import silhouette_score
from sklearn.decomposition import PCA
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
import seaborn as sns
from yellowbrick.cluster import SilhouetteVisualizer
import timeit
from dtaidistance import dtw
from datetime import datetime
from geopy.distance import geodesic
from sklearn.metrics.pairwise import euclidean_distances
# import netCDF4 to load .nc4 files
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
from sklearn.metrics import pairwise_distances
import os
import sys
import numpy
import PIL #for gifs, later
from skspatial.objects import Plane, Points
from skspatial.plotting import plot_3d

# Step 1A: Download all data files (if daily, one file per day)

# Step 1B: Create placeholder dataframe for time series
#### number of rows is length of file list

# Step 2: Iterate through date files

# Getting the current work directory (cwd)
thisdir = 'C:/Users/Owner/Documents/Postdoc/Climate Project/01012023/' #change this filepath to point toward folder containing nc4 files

# r=root, d=directories, f = files
filelist = []
for r, d, f in os.walk(thisdir):
    for file in f:
        if file.endswith(".nc4"):
            filelist.append(file)
            # print(os.path.join(r, file))

# Step 3: Date (need datetime for this) is in the file name (parse text)
#### Now iterate through filelist
datelist = []
for i in range(0,len(filelist)):
    datelist.append(filelist[i].split('.')[2])


# Step 4: Import contents of that file to grab lat/long/temp/etc.
begin_time = timeit.default_timer()
data_TS = pd.DataFrame(columns=('date', 'lat', 'lon','temp'))
for i in range(0,len(filelist)): # eventually will be range(0,len(filelist)) or test with np.arange(0,1)
    dataFile = os.path.join(thisdir, filelist[i])
    ds = nc.Dataset(dataFile)
    lons = ds.variables['lon'][:]
    lats = ds.variables['lat'][:]
    tempM = ds.variables['T2MMEAN'][:]
    ds = None
    for latnum in range(0,len(lats)): #eventually len(lats) or test with np.arange(0,1) 
        for lonnum in range(0,len(lons)): #eventually len(lons) or test with np.arange(0,1)
            newrow = [datelist[i], lats[latnum], lons[lonnum], np.squeeze(tempM[0,:,:])[latnum][lonnum]-273.15]
            data_TS.loc[len(data_TS.index)] = newrow #fixed to properly append new rows
            
data_TS.to_csv('data_TS_'+str(datetime.today().strftime("%y%m%d"))+'_final', index=False)


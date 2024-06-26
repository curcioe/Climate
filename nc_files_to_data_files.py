import pandas as pd
import os
import xarray as xr

thisdir = 'C:/Users/Owner/Documents/Postdoc/Climate Project/ERA5data/'

filelist = []
for r, d, f in os.walk(thisdir):
    for file in f:
        if file.endswith(".nc"):
            filelist.append(file)
    

datasets = []
for i in range(0,len(filelist)):
    dataFile = os.path.join(thisdir, filelist[i])
    ds = xr.open_dataset(dataFile)
    datasets.append(ds)

data_TS = xr.concat(datasets, dim='dataFile').to_dataframe().reset_index()
data_TS = data_TS.drop(['dataFile'],axis=1)
data_TS = data_TS.drop(['realization'],axis=1)
data_TS.rename(columns={'time': 'date'}, inplace=True)
data_TS.rename(columns={'t2m': 'temp'}, inplace=True)
data_TS['temp'] = data_TS['temp']-273.15
data_TS['month'] = pd.to_datetime(data_TS['date']).dt.month
data_TS['day'] = pd.to_datetime(data_TS['date']).dt.day

data_TS.to_csv('C:/Users/Owner/Documents/Postdoc/Climate Project/ERA5data/era5dataTS_raw.dat', sep='\t', index=False)

data_TS['unique_id'] = data_TS['lat'].astype(str) + "_" + data_TS['lon'].astype(str)
pivot_data_TS = pd.pivot_table(data_TS, values='temp', index='unique_id', columns='date')
pivot_data_TS.to_csv('C:/Users/Owner/Documents/Postdoc/Climate Project/ERA5data/era5pivot_raw.dat', sep='\t', index=True)
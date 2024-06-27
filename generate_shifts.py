import numpy as np
import pandas as pd
import math

pivot_data_TS = pd.read_csv('C:/Users/Owner/Documents/Postdoc/Climate Project/ERA5data/era5pivot_raw.dat', delimiter='\t', index_col=[0])
latloncombos = np.genfromtxt('C:/Users/Owner/Documents/Postdoc/Climate Project/ERA5data/latloncombos.dat', delimiter='\t', dtype=str)

min_euc_dist = []
shift_vals = []
index_vals = []
scale_vals_abs = []
scale_vals_log = []

for latloncombo in latloncombos:
    firstTS = pivot_data_TS.loc[latloncombo[0]]
    secondTS = pivot_data_TS.loc[latloncombo[1]]
    s1 = max(firstTS) - min(firstTS)
    s2 = max(secondTS) - min(secondTS)
    scale_vals_abs.append(abs(s1-s2))
    scale_vals_log.append(math.log(max(s1,s2)/min(s1,s2)))
    shift_euc_dist = []
    for j in np.arange(0,364):
        shift_euc_dist.append(np.linalg.norm((np.roll(firstTS,j).T) - (secondTS.T)))
    min_euc_dist.append(min(shift_euc_dist))
    index_vals.append(latloncombos.index(latloncombo))
    shift_vals.append(shift_euc_dist.index(min(shift_euc_dist)))

df = pd.DataFrame(data={"shifts": shift_vals, "indexes": index_vals, "minEucDist": min_euc_dist})
df.to_csv("shift_index_eucdist_vals.dat", sep='\t',index=False)
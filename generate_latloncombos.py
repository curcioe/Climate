import pandas as pd
import numpy as np
from itertools import combinations

pivot_data_TS = pd.read_csv('C:/Users/Owner/Documents/Postdoc/Climate Project/ERA5data/era5pivot_raw.dat', delimiter='\t', index_col=[0])

def rSubset(arr, r): 
    # return list of all subsets of length r
    # to deal with duplicate subsets use
    # set(list(combinations(arr, r)))
    return list(combinations(arr, r))

arr = pivot_data_TS.index
r = 2
latloncombos = np.array(rSubset(arr, r))
np.savetxt('C:/Users/Owner/Documents/Postdoc/Climate Project/ERA5data/latloncombos.dat', latloncombos,delimiter ='\t', fmt='%s')
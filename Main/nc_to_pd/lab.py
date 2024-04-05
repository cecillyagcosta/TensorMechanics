import netCDF4 as nc
import numpy as np
import pandas as pd

#simple call and load file

sample_path = r"C:/Users/cecil/Desktop/Neural/TensorMechanics/Sample/APHRO_MA_TAVE_025deg_V1808.1961.nc/sample_1961.nc"
data = nc.Dataset(sample_path)

#show variables of the dataset

print(data.variables.keys())

lon = data.variables['lon']
lat = data.variables['lat']
time = data.variables['time']
tave = data.variables['tave']
rstn = data.variables['rstn']

list = [lon, lat, time, tave, rstn]

for each in list:
    print(each)
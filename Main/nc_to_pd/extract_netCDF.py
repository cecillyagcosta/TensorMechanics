import os
import matplotlib.pyplot as plt 
import netCDF4 as nc

data = nc.Dataset("C:/Users/cecil/OneDrive/Documents/GitHub/TensorMechanics/Sample/Arquivos_Ceci/VL_0-0-0.nc")
print(data.variables.keys())
print(data['vertint_liq'][:])


import pandas as pd 
import numpy as np 
#import tensorflow as tf
import matplotlib.pyplot as plt 
import netCDF4 as nc

path = "C:/Users/cecil/Documents/GitHub/TensorMechanics/Sample/LWP.nc"
data = nc.Dataset(path)
LWP = data['LWP'][128,:,:] * 1000
plt.imshow(LWP, vmin=1, vmax=400)
plt.colorbar()
plt.show()

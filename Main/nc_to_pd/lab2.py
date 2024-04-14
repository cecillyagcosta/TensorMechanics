import pandas as pd 
import numpy as np 
#import tensorflow as tf
import matplotlib.pyplot as plt 
import netCDF4 as nc

path = "C:/Users/cecil/Desktop/Neural/TensorMechanics/Sample/LWP.nc"
data = nc.Dataset(path)

x_coords = np.array(data.variables['x_coords'][:])
y_coords = np.array(data.variables['y_coords'][:])
time_coords = np.array(data.variables['t_coords'][:])
rs_time = time_coords.resize(x_coords.shape)

df = pd.DataFrame.from_dict({'X':x_coords[:], 'Y':y_coords[:]})
clouds_perXY = pd.DataFrame(df['X'].groupby(df['Y']))
time_df = pd.DataFrame(time_coords)
time_df.columns = ['Time']
finalframe = pd.merge(clouds_perXY, time_df, left_index=True, right_index=True)
finalframe.columns = ['Y', 'X', 'Time']

#Graphic Buildup
#Clouds in coords 0,0 and time 0

x0 = finalframe['X'][0]
y0 = finalframe['Y'][0]
z0 = finalframe['Time'][0]

#Clouds in coords 1,1 and time 1

x1 = finalframe['X'][1]
y1 = finalframe['Y'][1]
z1 = finalframe['Time'][1]

fig, ax = plt.subplots()
sc1 = ax.scatter(x0, y0, c=z0, cmap='viridis', label='Cloud 0')
sc2 = ax.scatter(x1, y1, c=z1, cmap='viridis', label='Cloud 1')
plt.colorbar(sc1, ax=ax, label='Z')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
plt.show()
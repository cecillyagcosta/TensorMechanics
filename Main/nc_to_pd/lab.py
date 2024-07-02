import pandas as pd 
import numpy as np 
#import tensorflow as tf
import matplotlib.pyplot as plt 
import netCDF4 as nc

path = "C:/Users/cecil/OneDrive/Documents/GitHub/TensorMechanics/Sample/lwp.NC"
data = nc.Dataset(path)
LWP = data['LWP'][127,:,:] * 1000
#plt.imshow(LWP, vmin=1, vmax=400)
#plt.colorbar()
#plt.show()
#print(data.variables.keys())

#time has to be treated before use

time = data['t_coords'][:]/3600 + 6 #size 217
imgs = []
for i in range(len(time)):
    dataslice = data[i,:,:]
    plt.imshow(dataslice)
    plt.colorbar()
    plt.title(f'{time[i]:.2f}')
    nome_img = f'time_{i:04}.png'
    plt.savefig(nome_img)
    plt.close()
    imgs.append(nome_img)
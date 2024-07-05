from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

vis = ['vertint_liq','vertint_driz','vertint_cloud','vertint_rain']
path= "C:/Users/cecil/OneDrive/Documents/GitHub/TensorMechanics/Sample/lwp.NC"
ds = Dataset(path)
pasta = '0-4-0/'

tempo = ds['t_coords'][:]/3600 + 6
ts_vl = []
ts_vd = []
ts_vc = []
ts_vr = []

for indice, nome in enumerate(vis):
    vi = ds[nome]
    vi = vi[:]*1000 # g/m2
    maximo = -np.inf
    for t in range(len(tempo)):
        aux = np.max(np.array(vi[t,:,:]))
        if aux > maximo: maximo = aux

    imgs = []

    for i in range(len(tempo)):
        ds_slice = vi[i,:,:]
        plt.figure(figsize=(10, 5))
        plt.imshow(ds_slice, aspect='auto', origin='lower', cmap='jet', vmin=0, vmax=maximo)
        plt.colorbar(label='g/m^2')
        plt.title(f'{tempo[i]:.2f}')
        nome_img = f'tempo_{i:04}.png'
        plt.savefig(nome_img)
        plt.close()
        imgs.append(nome_img)
    

    with imageio.get_writer(f'../plots/{pasta}{nome}.gif', mode='I') as writer:
         for imagem in imgs:
            aux = imageio.imread(imagem)
            writer.append_data(aux)

    for imagem in imgs:
        os.remove(imagem)

    for t in range(len(tempo)):
        m = np.array(vi[t,:,:])
        aux = m[m>0.6]
        if indice==0:
            if len(aux)==0: ts_vl.append(0)
            else: ts_vl.append(np.mean(aux))
        if indice==1:
            if len(aux)==0: ts_vd.append(0)
            else: ts_vd.append(np.mean(aux))
        if indice==2:
            if len(aux)==0: ts_vc.append(0)
            else: ts_vc.append(np.mean(aux))
        if indice==3:
            if len(aux)==0: ts_vr.append(0)
            else: ts_vr.append(np.mean(aux))

plt.plot(tempo,ts_vl,label='vi_liq')
plt.plot(tempo,ts_vd,label='vi_driz')
plt.plot(tempo,ts_vc,label='vi_cloud')
plt.plot(tempo,ts_vr,label='vi_rain')
plt.ylabel('g/m^2')
plt.legend()
plt.grid(True)
plt.savefig('../plots/'+pasta+'vertint_medio.png')
plt.close()

ds.close()

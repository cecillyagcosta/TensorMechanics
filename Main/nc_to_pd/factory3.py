import os
import matplotlib.pyplot as plt 
import netCDF4 as nc

path = "C:/Users/cecil/OneDrive/Documents/GitHub/TensorMechanics/Main/Dataset/vertint_sample/CM3_0-0-0.nc"
outdir = "D:/CNNData/LWPimages/CM/CM3_0-0-0"

data = nc.Dataset(path)
time = data['cloud_m3'].shape[0]

def save_cloud_m3_image(path, output_dir, index):
    # Verificar se o diretório de saída existe, caso contrário, criar
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Carregar os dados
    data = nc.Dataset(path)
    cloud_m3 = data['cloud_m3'][index, :, :] * 1000
    
    # Criar a figura sem margens, eixos e barra de cor
    fig, ax = plt.subplots()
    ax.imshow(cloud_m3, vmin=1, vmax=600)
    ax.axis('off')   
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Gerar o caminho do arquivo de saída
    output_path = os.path.join(output_dir, f'cloud_m3_image_{index}.png')
    # Salvar a imagem com alta resolução
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=1200)
    plt.close()
    print(f"Imagem salva em: {output_path}")

for i in range(time):
    save_cloud_m3_image(path, outdir, i)

    
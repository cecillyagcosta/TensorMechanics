import os
import matplotlib.pyplot as plt 
import netCDF4 as nc

path = "C:/Users/cecil/OneDrive/Documents/GitHub/TensorMechanics/Sample/lwp.NC"
outdir = "C:/Users/cecil/OneDrive/Documents/GitHub/TensorMechanics/Main/Dataset/Data/sample_v2"

data = nc.Dataset(path)
time = data['LWP'].shape[0]

def save_LWP_image(path, output_dir, index):
    # Verificar se o diretório de saída existe, caso contrário, criar
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Carregar os dados
    data = nc.Dataset(path)
    LWP = data['LWP'][index, :, :] * 1000
    
    # Criar a figura sem margens, eixos e barra de cor
    fig, ax = plt.subplots()
    ax.imshow(LWP, vmin=1, vmax=400)
    ax.axis('off')  # Desativar os eixos
    
    # Remover as margens
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Gerar o caminho do arquivo de saída
    output_path = os.path.join(output_dir, f'LWP_image_{index}.png')
    
    # Salvar a imagem com alta resolução (300 DPI)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    print(f"Imagem salva em: {output_path}")


for i in range(time):
    save_LWP_image(path, outdir, i)

    
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Defina o diretório onde as imagens estão localizadas
input_dir = r'C:/Users/cecil/OneDrive/Documents/Dataset_CloudOrgs/VL0/round_class/cloud_classifier/Organizado/vertint_liq_image_105/mornas/figura_345_size_300x277_morno_from_vertint_liq_image_105'
output_dir = r'C:/Users/cecil/OneDrive/Documents/Dataset_CloudOrgs/VL0/round_class/cloud_classifier/Organizado/output/'

# Certifique-se de que a pasta de saída existe
os.makedirs(output_dir, exist_ok=True)

# Percorra os arquivos na pasta de entrada
for filename in os.listdir(input_dir):
    if filename.startswith("figura_") and filename.endswith(".png"):
        figura_path = os.path.join(input_dir, filename)
        
        # Aqui estamos assumindo que a imagem original termina com 'highlighted.png'
        highlighted_path = os.path.join(input_dir, filename.replace('.png', '_highlighted.png'))

        # Debug: Verificando se os caminhos dos arquivos estão corretos
        print(f"Tentando carregar a figura: {figura_path}")
        print(f"Tentando carregar a imagem original: {highlighted_path}")

        # Verifique se os arquivos existem antes de carregá-los
        if not os.path.isfile(figura_path):
            print(f"Figura não encontrada: {figura_path}.")
            continue

        if not os.path.isfile(highlighted_path):
            print(f"Imagem original não encontrada: {highlighted_path}.")
            continue

        # Carregar as imagens
        figura_image = cv2.imread(figura_path)
        original_image = cv2.imread(highlighted_path)

        # Verifique se as imagens foram carregadas corretamente
        if figura_image is None:
            print(f"Erro ao carregar a figura: {figura_path}.")
            continue

        if original_image is None:
            print(f"Erro ao carregar a imagem original: {highlighted_path}.")
            continue

        # Redimensionar a figura para o tamanho da imagem original
        figura_resized = cv2.resize(figura_image, (original_image.shape[1], original_image.shape[0]))

        # Usar Matplotlib para plotar as imagens lado a lado
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title('Imagem Original')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(figura_resized, cv2.COLOR_BGR2RGB))
        plt.title('Figura')
        plt.axis('off')

        # Gerar o nome do arquivo de saída
        output_filename = f'combined_{os.path.basename(figura_path)}'
        output_path = os.path.join(output_dir, output_filename)

        # Salvar a figura combinada
        plt.tight_layout()
        plt.savefig(output_path, dpi=1200)
        plt.close()

        print(f"Imagem combinada salva: {output_path}")

print("Processo finalizado.")
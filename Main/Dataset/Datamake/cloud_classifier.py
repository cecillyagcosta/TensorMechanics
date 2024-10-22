import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import io, color, measure, morphology

# Função para classificar com base na intensidade (calor)
def classify_by_intensity(region, gray_image):
    minr, minc, maxr, maxc = region.bbox
    region_image = gray_image[minr:maxr, minc:maxc]
    avg_intensity = np.mean(region_image)
    
    if avg_intensity < 33.3 / 400:
        return 'ignorar'  # Não salva
    elif avg_intensity < 100 / 400:
        return 'frio'
    elif avg_intensity < 200 / 400:
        return 'morno'
    else:
        return 'quente'

# Função para verificar se o tamanho é próximo de 15x15
def is_size_too_small(height, width, min_size=50, avoid_size=(11, 11), tolerance=1):
    if height * width < min_size:
        return True
    if abs(height - avoid_size[0]) <= tolerance and abs(width - avoid_size[1]) <= tolerance:
        return True
    return False

# Caminho da pasta que contém as imagens
input_dir =  "C:/Users/cecil/OneDrive/Documents/Dataset_CloudOrgs/VL0/Source"
output_dir = "C:/Users/cecil/OneDrive/Documents/Dataset_CloudOrgs/VL0/round_class/cloud_classifier/run_2"
os.makedirs(output_dir, exist_ok=True)

# Extensões de arquivo permitidas
allowed_extensions = ['.png', '.jpg', '.jpeg', '.tiff']

# Percorrer todas as imagens no diretório
for image_file in os.listdir(input_dir):
    if any(image_file.endswith(ext) for ext in allowed_extensions):
        image_path = os.path.join(input_dir, image_file)
        print(f"Processando: {image_file}")

        # Carregar a imagem
        image = io.imread(image_path)
        print(f"Imagem carregada: {image.shape}")

        # Remover o canal alfa, se presente
        if image.shape[2] == 4:
            image = image[:, :, :3]
            print("Canal alfa removido.")

        # Converter para escala de cinza
        gray_image = color.rgb2gray(image)

        # Definir o limiar
        threshold_value = 33.3 / 400
        binary = gray_image > threshold_value

        # Remover pequenos objetos
        cleaned = morphology.remove_small_objects(binary, min_size=35)
        print("Objetos pequenos removidos.")

        # Encontrar contornos das figuras
        labels = measure.label(cleaned)
        regions = measure.regionprops(labels)

        # Verificar quantas regiões foram encontradas
        print(f"Número de regiões encontradas: {len(regions)}")

        if len(regions) == 0:
            print("Nenhuma região encontrada.")
            continue

        # Processar cada região e classificar
        for k, region in enumerate(regions):
            intensity_class = classify_by_intensity(region, gray_image)
            if intensity_class == 'ignorar':
                continue  # Não salva figuras abaixo de 33.3

            # Verificar o tamanho da figura
            minr, minc, maxr, maxc = region.bbox
            height, width = maxr - minr, maxc - minc

            # Verificar se o tamanho é próximo de 11x12 ou muito pequeno
            if is_size_too_small(height, width):
                continue  # Ignora figuras pequenas e próximas a 11x12

            # Criar uma pasta para a classificação da região com o tamanho e a classificação
            figure_name = f'figura_{k}_size_{height}x{width}_{intensity_class}'
            figure_dir = os.path.join(output_dir, figure_name)
            os.makedirs(figure_dir, exist_ok=True)

            # Recortar a imagem da região
            cropped_image = image[minr:maxr, minc:maxc]
            save_path_cropped = os.path.join(figure_dir, f'{figure_name}.png')
            plt.imsave(save_path_cropped, cropped_image, dpi=1200)
            print(f"Imagem cortada salva em {save_path_cropped}")

            # Demarcar a posição na imagem original
            highlighted_image = image.copy()
            plt.figure()
            plt.imshow(highlighted_image)
            rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                 edgecolor='red', facecolor='none')
            plt.gca().add_patch(rect)
            plt.axis('off')

            # Salvar a imagem com a demarcação
            highlighted_save_path = os.path.join(figure_dir, f'{figure_name}_highlighted.png')
            plt.savefig(highlighted_save_path, dpi=1200, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Imagem destacada salva em {highlighted_save_path}")

print("Processamento concluído!")

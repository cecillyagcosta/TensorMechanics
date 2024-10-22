import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import io, color, measure, morphology

# Função para classificar com base na área da região
def classify_by_area(region):
    area = region.area
    if area < 1000:
        return 'pequeno'
    elif area < 5000:
        return 'medio'
    else:
        return 'grande'

# Função para classificar com base na intensidade (calor)
def classify_by_intensity(region, gray_image):
    minr, minc, maxr, maxc = region.bbox
    region_image = gray_image[minr:maxr, minc:maxc]
    avg_intensity = np.mean(region_image)
    
    if avg_intensity < 33.3 / 400:
        return 'muito_frio'
    elif avg_intensity < 100 / 400:
        return 'frio'
    elif avg_intensity < 200 / 400:
        return 'morno'
    else:
        return 'quente'

# Caminho da pasta que contém as imagens
input_dir =  "C:/Users/cecil/OneDrive/Documents/Dataset_CloudOrgs/VL0/Source"
output_dir = "C:/Users/cecil/OneDrive/Documents/Dataset_CloudOrgs/VL0/Classification/run_1"
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

        # Remover o canal alfa, se presente
        if image.shape[2] == 4:
            image = image[:, :, :3]

        # Converter para escala de cinza
        gray_image = color.rgb2gray(image)

        # Definir o limiar
        threshold_value = 33.3 / 400
        binary = gray_image > threshold_value

        # Remover pequenos objetos
        cleaned = morphology.remove_small_objects(binary, min_size=50)

        # Encontrar contornos das figuras
        labels = measure.label(cleaned)
        regions = measure.regionprops(labels)

        # Processar cada região
        for i, region in enumerate(regions):
            # Classificação por área
            size_class = classify_by_area(region)
            # Classificação por intensidade
            heat_class = classify_by_intensity(region, gray_image)
            
            # Pegar as coordenadas da bounding box
            minr, minc, maxr, maxc = region.bbox
            area = region.area

            # Criar pasta para a classe (área e calor combinados)
            class_dir = os.path.join(output_dir, f"{size_class}_{heat_class}")
            os.makedirs(class_dir, exist_ok=True)
            
            # Salvar a região como imagem
            cropped_image = image[minr:maxr, minc:maxc]

            # Adicionar posição e tamanho no nome do arquivo
            save_path = os.path.join(
                class_dir, 
                f'{os.path.splitext(image_file)[0]}_image_{i}_pos({minr},{minc})_size({maxr-minr}x{maxc-minc})_area({area}).png'
            )
            
            fig, ax = plt.subplots(figsize=(cropped_image.shape[1] / 100, cropped_image.shape[0] / 100), dpi=1200)
            ax.imshow(cropped_image)
            ax.axis('off')  # Desativa os eixos
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            print(f"Imagem {i} salva em: /n '{save_path}' /n ** Source: '{image_path}' **")

print("Processamento concluído!")

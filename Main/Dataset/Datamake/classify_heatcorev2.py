import os
import numpy as np
from skimage import io, color, measure, morphology

imdir = "C:/Users/cecil/OneDrive/Documents/GitHub/TensorMechanics/Main/Dataset/Data/vertint_liq_0.0.0_SAMPLE_1200DPI"
sampleoutdir = "C:/Users/cecil/OneDrive/Documents/GitHub/TensorMechanics/Main/Dataset/Data/vertint_liq_0.0.0_SAMPLE_1200DPI_output_v1_HEATCOREV3"

def run(indir, outdir, threshold_value, dpi=300):
    os.makedirs(outdir, exist_ok=True)

    # Função para salvar as informações sobre os núcleos
    def save_nuclei_info(nuclei_info, file_path):
        with open(file_path, 'w') as f:
            for info in nuclei_info:
                f.write(f"Image: {info['image']}, Position: {info['position']}, Size: {info['size']}, Heat Level: {info['heat_level']}\n")

    # Função para determinar o nível de calor de um núcleo
    def classify_heat_level(cropped_image, high_threshold=230, low_threshold=200):
        # Converter para escala de cinza
        gray_image = color.rgb2gray(cropped_image)

        # Contagem de pixels próximos de 255
        hot_pixels = np.sum(gray_image > high_threshold / 255.0)
        warm_pixels = np.sum((gray_image > low_threshold / 255.0) & (gray_image <= high_threshold / 255.0))
        total_pixels = gray_image.size

        # Percentual de pixels quentes
        hot_percent = (hot_pixels / total_pixels) * 100
        warm_percent = (warm_pixels / total_pixels) * 100

        # Classificação do calor com base nos percentuais
        if hot_percent > 20:  # Mais de 20% de pixels quentes
            return "very_hot"
        elif hot_percent > 10 or warm_percent > 20:  # Entre 10-20% pixels quentes ou 20% moderadamente quentes
            return "hot"
        elif warm_percent > 10:  # Entre 10-20% pixels mornos
            return "warm"
        else:
            return "cold"

    # Função para processar uma única imagem
    def process_image(image_path, outdir, threshold_value):
        # Carregar a imagem
        image = io.imread(image_path)

        # Remover o canal alfa, se presente
        if image.shape[2] == 4:
            image = image[:, :, :3]

        # Converter para escala de cinza
        gray_image = color.rgb2gray(image)

        # Aplicar o limiar para identificar núcleos quentes
        hot_nuclei = gray_image > threshold_value
        
        # Limpeza de objetos pequenos
        cleaned = morphology.remove_small_objects(hot_nuclei, min_size=50)

        # Identificar e classificar regiões
        labels = measure.label(cleaned)
        regions = measure.regionprops(labels)

        # Armazenar informações dos núcleos
        nuclei_info = []
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        for i, region in enumerate(regions):
            minr, minc, maxr, maxc = region.bbox
            cropped_image = image[minr:maxr, minc:maxc]

            # Classificar o núcleo pelo nível de calor
            heat_level = classify_heat_level(cropped_image)

            # Salvar a imagem classificada
            class_outdir = os.path.join(outdir, heat_level)
            os.makedirs(class_outdir, exist_ok=True)
            save_path = os.path.join(class_outdir, f'{base_name}_nucleus_{i}.png')
            io.imsave(save_path, cropped_image)

            # Armazenar posição e informações do núcleo
            nuclei_info.append({
                'image': base_name,
                'position': (minr, minc),
                'size': region.area,
                'heat_level': heat_level
            })

        # Salvar as informações de posição e tamanho dos núcleos
        info_file_path = os.path.join(outdir, f"{base_name}_nuclei_info.txt")
        save_nuclei_info(nuclei_info, info_file_path)

    # Iterar sobre todas as imagens na pasta de entrada
    for filename in os.listdir(indir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(indir, filename)
            process_image(image_path, outdir, threshold_value / 400)

run(imdir, sampleoutdir, 33.3, dpi=1200)

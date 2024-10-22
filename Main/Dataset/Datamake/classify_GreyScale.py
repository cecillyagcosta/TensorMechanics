import os
import numpy as np
from skimage import io, color, measure, morphology
from PIL import Image

imdir = "C:/Users/cecil/OneDrive/Documents/GitHub/TensorMechanics/Main/Dataset/Data/vertint_liq_0.0.0_SAMPLE_1200DPI_output_v1"
sampleoutdir = "C:/Users/cecil/OneDrive/Documents/GitHub/TensorMechanics/Main/Dataset/Data/vertint_liq_0.0.0_SAMPLE_1200DPI_output_v1_GREY_CLASSIFY_OUTPUT"

def run(indir, outdir, threshold_value, dpi=300):
    os.makedirs(outdir, exist_ok=True)

    # Função para salvar as imagens em alta qualidade
    def save_high_res_image(image_array, save_path, dpi):
        pil_image = Image.fromarray(image_array)
        pil_image.save(save_path, dpi=(dpi, dpi))

    # Função para calcular o nível de calor com base na intensidade em escala de cinza
    def classify_by_heat(gray_image):
        avg_intensity = np.mean(gray_image)  # Média da intensidade dos pixels em escala de cinza

        # Definir limiares para a classificação de calor
        if avg_intensity < 0.3:  # Mais escura
            return "low_heat"
        elif avg_intensity < 0.7:  # Intermediária
            return "moderate_heat"
        else:  # Mais clara
            return "high_heat"

    # Função para classificar a imagem pela complexidade (número de objetos)
    def classify_by_complexity(cleaned_image):
        labels = measure.label(cleaned_image)
        regions = measure.regionprops(labels)
        num_objects = len(regions)

        # Classificar a imagem como simples ou complexa com base no número de objetos
        if num_objects < 5:
            return "simple"
        else:
            return "complex"

    # Função para processar uma única imagem
    def process_image(image_path, outdir, threshold_value):
        # Carregar a imagem
        image = io.imread(image_path)

        # Verificar se a imagem tem pelo menos 3 canais (RGB)
        if image.shape[2] < 3:
            raise ValueError(f"A imagem {image_path} não tem canais RGB suficientes.")
        
        # Remover o canal alfa, se presente
        if image.shape[2] == 4:
            image = image[:, :, :3]

        # Converter para escala de cinza
        gray_image = color.rgb2gray(image)

        # Aplicar o limiar
        binary = gray_image > threshold_value

        # Remover pequenos objetos
        cleaned = morphology.remove_small_objects(binary, min_size=50)

        # Encontrar contornos das figuras
        labels = measure.label(cleaned)
        regions = measure.regionprops(labels)

        # Salvar as regiões como imagens individuais
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        for i, region in enumerate(regions):
            minr, minc, maxr, maxc = region.bbox
            cropped_image = image[minr:maxr, minc:maxc]
            cropped_gray = gray_image[minr:maxr, minc:maxc]

            # Classificar a imagem pelo nível de "calor"
            heat_class = classify_by_heat(cropped_gray)

            # Classificar a imagem pela complexidade
            complexity_class = classify_by_complexity(cleaned)

            # Combinar as duas classificações (calor e complexidade)
            combined_class = f"{heat_class}_{complexity_class}"

            # Criar pasta para a combinação das classificações
            class_outdir = os.path.join(outdir, combined_class)
            os.makedirs(class_outdir, exist_ok=True)

            # Salvar a imagem classificada
            save_path = os.path.join(class_outdir, f'{base_name}_figura_{i}.png')
            save_high_res_image(cropped_image, save_path, dpi)

    # Iterar sobre todas as imagens na pasta de entrada
    for filename in os.listdir(indir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(indir, filename)
            process_image(image_path, outdir, threshold_value / 400)

run(imdir, sampleoutdir, 33.3, dpi=300)

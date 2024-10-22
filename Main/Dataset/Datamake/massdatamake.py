import os
import numpy as np
from skimage import io, color, measure, morphology
from PIL import Image
import logging

# Configurando o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

imdir = "C:/Users/cecil/OneDrive/Documents/GitHub/TensorMechanics/Main/Dataset/Data/vertint_liq_0.0.0_SAMPLE_1200DPI"
sampleoutdir = "C:/Users/cecil/OneDrive/Documents/GitHub/TensorMechanics/Main/Dataset/Data/vertint_liq_0.0.0_SAMPLE_1200DPI_output_v1"

def run(indir, outdir, threshold_value, dpi=300, block_size=1024):
    os.makedirs(outdir, exist_ok=True)

    # Função para salvar as imagens em alta qualidade
    def save_high_res_image(image_array, save_path, dpi):
        pil_image = Image.fromarray(image_array)
        pil_image.save(save_path, dpi=(dpi, dpi))

    # Processamento por blocos para otimizar o uso de memória
    def process_image_in_blocks(image, block_size):
        # Processar a imagem por blocos para evitar estouro de memória
        h, w = image.shape[:2]
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                yield image[i:i+block_size, j:j+block_size], i, j

    # Função para processar uma única imagem
    def process_image(image_path, outdir, threshold_value):
        logging.info(f"Processando a imagem: {image_path}")
        
        # Carregar a imagem
        image = io.imread(image_path)

        # Verificar se a imagem tem pelo menos 3 canais (RGB)
        if image.shape[2] < 3:
            logging.error(f"A imagem {image_path} não tem canais RGB suficientes.")
            raise ValueError(f"A imagem {image_path} não tem canais RGB suficientes.")
        
        # Remover o canal alfa, se presente
        if image.shape[2] == 4:
            image = image[:, :, :3]

        # Converter para escala de cinza em blocos
        gray_image = np.empty_like(image[:, :, 0], dtype=np.float32)  # Usando float32

        for block, i, j in process_image_in_blocks(image, block_size):
            gray_image[i:i+block_size, j:j+block_size] = color.rgb2gray(block).astype(np.float32)

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
            
            # Salvar usando a nova função com alta resolução
            save_path = os.path.join(outdir, f'{base_name}_figura_{i}.png')
            save_high_res_image(cropped_image, save_path, dpi)
            logging.info(f"Imagem salva em alta resolução: {save_path}")

    # Iterar sobre todas as imagens na pasta de entrada
    for filename in os.listdir(indir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(indir, filename)
            process_image(image_path, outdir, threshold_value / 400)

logging.info("Iniciando processamento...")
run(imdir, sampleoutdir, 33.3, dpi=1200)
logging.info("Processamento concluído.")

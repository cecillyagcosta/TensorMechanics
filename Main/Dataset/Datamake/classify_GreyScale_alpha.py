import os
import numpy as np
from skimage import io, color
from PIL import Image
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

imdir = "C:/Users/cecil/OneDrive/Documents/GitHub/TensorMechanics/Main/Dataset/Data/vertint_liq_0.0.0_SAMPLE_1200DPI_output_v1"
sampleoutdir = "C:/Users/cecil/OneDrive/Documents/GitHub/TensorMechanics/Main/Dataset/Data/vertint_liq_0.0.0_SAMPLE_1200DPI_output_v1__ONLYGREY_CLASSIFY_OUTPUT"

def run(indir, outdir, threshold_value, dpi=300, block_size=1024):
    os.makedirs(outdir, exist_ok=True)

    # Função para salvar imagens com alta resolução
    def save_high_res_image(image_array, save_path, dpi):
        pil_image = Image.fromarray(image_array)
        pil_image.save(save_path, dpi=(dpi, dpi))

    # Processamento por blocos para otimizar o uso de memória
    def process_image_in_blocks(image, block_size):
        h, w = image.shape[:2]
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                yield image[i:i+block_size, j:j+block_size], i, j

    # Função para calcular o "calor" da imagem
    def classify_by_heat(gray_image, threshold_heat=230):
        # Contar quantos pixels estão acima do valor definido como "quente"
        heat_pixels = np.sum(gray_image >= threshold_heat / 255.0)
        total_pixels = gray_image.size
        heat_ratio = heat_pixels / total_pixels

        # Classificar baseado na proporção de pixels "quentes"
        if heat_ratio > 0.5:
            return "high_heat"
        elif heat_ratio > 0.2:
            return "medium_heat"
        else:
            return "low_heat"

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

        # Classificar a imagem pelo calor
        heat_class = classify_by_heat(gray_image)

        # Criar diretório para a classificação
        class_outdir = os.path.join(outdir, heat_class)
        os.makedirs(class_outdir, exist_ok=True)

        # Salvar a imagem classificada
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(class_outdir, f'{base_name}.png')
        save_high_res_image(image, save_path, dpi)
        logging.info(f"Imagem classificada e salva: {save_path}")

    # Iterar sobre todas as imagens na pasta de entrada
    for filename in os.listdir(indir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(indir, filename)
            process_image(image_path, outdir, threshold_value / 400)

logging.info("Iniciando o processamento...")
run(imdir, sampleoutdir, 33.3, dpi=1200)
logging.info("Processamento concluído.")

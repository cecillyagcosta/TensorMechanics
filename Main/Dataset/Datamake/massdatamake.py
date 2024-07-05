import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, measure, morphology

imdir = "C:/Users/cecil/OneDrive/Documents/GitHub/TensorMechanics/Main/Dataset/Data/sample_v2"
sampleoutdir = "C:/Users/cecil/OneDrive/Documents/GitHub/TensorMechanics/Main/Dataset/Data/sample_v3"

def run(indir, outdir, thrhold_value):
    os.makedirs(outdir, exist_ok=True)

    # Função para processar uma única imagem
    def process_image(image_path, outdir, threshold_value=thrhold_value/400):
        # Carregar a imagem
        image = io.imread(image_path)

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
            save_path = os.path.join(outdir, f'{base_name}_figura_{i}.png')
            io.imsave(save_path, cropped_image)

    # Iterar sobre todas as imagens na pasta de entrada
    for filename in os.listdir(indir):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            image_path = os.path.join(indir, filename)
            process_image(image_path, outdir)


run(imdir, sampleoutdir, 33.3)
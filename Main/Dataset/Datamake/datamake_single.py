import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, measure, morphology

# Carregar a imagem
image_path = "C:/Users/cecil/OneDrive/Documents/GitHub/TensorMechanics/Main/Dataset/Data/subjects/t128/t128.png"

# "C:/Users/cecil/OneDrive/Documents/GitHub/TensorMechanics/Main/Dataset/Data/test/Figure_2.png"
# "C:/Users/cecil/OneDrive/Documents/GitHub/TensorMechanics/Main/Dataset/Data/subjects/t128/t128.png"

image = io.imread(image_path)

# Remover o canal alfa, se presente
if image.shape[2] == 4:
    image = image[:, :, :3]

# Converter para escala de cinza
gray_image = color.rgb2gray(image)

# Definir o limiar para identificar regiões com valores acima de 100
threshold_value = 33.3 / 400  # Normalizar para a escala de 0 a 1

# Aplicar o limiar
binary = gray_image > threshold_value

# Remover pequenos objetos
cleaned = morphology.remove_small_objects(binary, min_size=50)

# Encontrar contornos das figuras
labels = measure.label(cleaned)
regions = measure.regionprops(labels)

# Plotar as regiões identificadas
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')

for region in regions:
    minr, minc, maxr, maxc = region.bbox
    rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                         edgecolor='red', facecolor='none')
    ax.add_patch(rect)

plt.show()

# Salvar as regiões como imagens individuais
saved_images_threshold = []
for i, region in enumerate(regions):
    minr, minc, maxr, maxc = region.bbox
    cropped_image = image[minr:maxr, minc:maxc]
    save_path = f'C:/Users/cecil/OneDrive/Documents/GitHub/TensorMechanics/Main/Dataset/Data/sample_v1/image_{i}.png'
    io.imsave(save_path, cropped_image)
    saved_images_threshold.append(save_path)

saved_images_threshold

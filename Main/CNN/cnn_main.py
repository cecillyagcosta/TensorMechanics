import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from scipy.spatial.distance import cdist
from skimage import io, color, measure, morphology

# Função para detectar nuvens
def detect_clouds(image):
    gray_image = color.rgb2gray(image)
    threshold_value = 33.3 / 400
    binary = gray_image > threshold_value
    cleaned = morphology.remove_small_objects(binary, min_size=10)
    labels = measure.label(cleaned)
    regions = measure.regionprops(labels)

    # Extrair centros e raios das nuvens
    cloud_centers = []
    cloud_radii = []
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        center = [(minr + maxr) / 2, (minc + maxc) / 2]  # Centro da nuvem
        radius = np.sqrt((maxr - minr) * (maxc - minc)) / 2  # Raio aproximado
        cloud_centers.append(center)
        cloud_radii.append(radius)

    return np.array(cloud_centers), np.array(cloud_radii)

# Função para calcular o COP
def calculate_cop(image):
    # 1. Detectar nuvens
    cloud_centers, cloud_radii = detect_clouds(image)

    # 2. Calcular o COP
    num_clouds = len(cloud_centers)
    if num_clouds < 2:
        return 0.0  # Não há pares de nuvens
    
    distances = cdist(cloud_centers, cloud_centers)
    cop_value = 0.0
    count = 0
    for i in range(num_clouds):
        for j in range(i + 1, num_clouds):
            if distances[i, j] > 0:
                v_ij = (cloud_radii[i] + cloud_radii[j]) / distances[i, j]
                cop_value += v_ij
                count += 1

    if count > 0:
        cop_value /= count
    return cop_value

# 1. Carregar as imagens e calcular o COP
def load_images_and_calculate_cop(image_dir, target_size=(128, 128)):
    images = []
    cop_values = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):  # Verificar o formato
            img = Image.open(os.path.join(image_dir, filename))
            img = img.resize(target_size)  # Redimensionar
            img = img.convert('RGB')  # Converter para RGB (remove o canal alfa)
            img = np.array(img) / 255.0  # Normalizar para [0, 1]
            images.append(img)
            cop_values.append(calculate_cop(img))  # Calcular o COP
    return np.array(images), np.array(cop_values)

image_dir = 'D:/CNNData/LWPimages/VL/VL_0-2-0'
images, cop_values = load_images_and_calculate_cop(image_dir)
print(f"Total de imagens carregadas: {len(images)}")
print(f"Formato das imagens: {images.shape}")  # Deve ser (num_imagens, 128, 128, 3)

# 2. Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(images, cop_values, test_size=0.2, random_state=42)

# 3. Callback personalizado para exibir o COP médio a cada época
class COPCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Calcular o COP médio para o conjunto de treino e teste
        train_cop = np.mean(y_train)
        test_cop = np.mean(y_test)
        print(f"\nEpoch {epoch + 1}:")
        print(f"  Train COP: {train_cop:.4f}, Test COP: {test_cop:.4f}")
        print(f"  Train Loss: {logs['loss']:.4f}, Val Loss: {logs['val_loss']:.4f}")
        print(f"  Train MAE: {logs['mae']:.4f}, Val MAE: {logs['val_mae']:.4f}")

# Treino da CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1)  # Saída única para prever o COP
])

# Compilar o modelo usando Adam
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

# Treinar o modelo com o callback personalizado
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[COPCallback()]  # Adiciona o callback
)

# 5. Avaliar o modelo
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# 6. Exibir o COP final
final_train_cop = np.mean(y_train)
final_test_cop = np.mean(y_test)
print(f"\nFinal Train COP: {final_train_cop:.4f}")
print(f"Final Test COP: {final_test_cop:.4f}")
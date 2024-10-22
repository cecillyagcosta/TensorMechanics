import os
import shutil

# Caminho da pasta principal que contém todas as figuras
input_dir = "C:/Users/cecil/OneDrive/Documents/Dataset_CloudOrgs/VL0/round_class/cloud_classifier/run_3"
output_dir = "C:/Users/cecil/OneDrive/Documents/Dataset_CloudOrgs/VL0/round_class/cloud_classifier/Organizado"

# Certifique-se que o diretório de saída existe
os.makedirs(output_dir, exist_ok=True)

# Função para organizar pastas
def organize_folders_by_keywords(input_dir, output_dir):
    # Percorre todas as pastas no diretório de entrada
    for folder_name in os.listdir(input_dir):
        if os.path.isdir(os.path.join(input_dir, folder_name)):
            # Verificar se o nome da pasta segue o padrão "figura_xxx_size_..."
            if "from_" in folder_name:
                # Extrair o nome da imagem original
                image_id = folder_name.split("from_")[-1]
                classification = None

                # Determinar a classificação: "frio", "morno" ou "quente"
                if "frio" in folder_name:
                    classification = "frias"
                elif "morno" in folder_name:
                    classification = "mornas"
                elif "quente" in folder_name:
                    classification = "quentes"

                if classification:
                    # Criar a pasta da imagem original dentro do diretório de saída
                    image_folder = os.path.join(output_dir, image_id)
                    os.makedirs(image_folder, exist_ok=True)

                    # Criar a subpasta para a classificação (frio, morno, quente)
                    class_folder = os.path.join(image_folder, classification)
                    os.makedirs(class_folder, exist_ok=True)

                    # Mover a pasta atual para a subpasta correta
                    source_path = os.path.join(input_dir, folder_name)
                    destination_path = os.path.join(class_folder, folder_name)
                    shutil.move(source_path, destination_path)
                    print(f"Movido: {folder_name} para {destination_path}")

# Executa a organização
organize_folders_by_keywords(input_dir, output_dir)

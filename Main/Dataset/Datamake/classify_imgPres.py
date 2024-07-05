import os
from PIL import Image
import shutil
import time
from pathlib import Path

def move_file(src, dst):
    try:
        shutil.move(src, dst)
        return True
    except Exception as e:
        print(f"Erro ao mover arquivo: {e}")
        return False

def create_folders_for_resolution(input_dir, output_dir):
    # Verificar se o diretório de saída existe, caso contrário, criar
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Lista para armazenar os caminhos dos arquivos a serem movidos
    files_to_move = []

    # Percorrer todas as imagens no diretório de entrada
    for filename in os.listdir(input_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            filepath = os.path.join(input_dir, filename)
            
            # Verificar se o arquivo está acessível
            if Path(filepath).is_file():
                files_to_move.append(filepath)
            else:
                print(f"Arquivo '{filename}' não está acessível ou não existe.")

    # Função para mover arquivos verificando acessibilidade
    def move_files_with_check(filepaths):
        for filepath in filepaths:
            try:
                # Verificar novamente se o arquivo está acessível
                if Path(filepath).is_file():
                    # Abrir a imagem e obter suas dimensões
                    with Image.open(filepath) as img:
                        width, height = img.size
                        
                        # Criar uma pasta com base na resolução da imagem
                        resolution_folder = os.path.join(output_dir, f"{width}x{height}")
                        if not os.path.exists(resolution_folder):
                            os.makedirs(resolution_folder)
                        
                        # Mover a imagem para a pasta correspondente
                        new_filepath = os.path.join(resolution_folder, os.path.basename(filepath))
                        move_file(filepath, new_filepath)
                        print(f"Imagem '{filename}' movida para '{resolution_folder}'")
                else:
                    print(f"Arquivo '{filename}' não está acessível ou não existe.")
            
            except Exception as e:
                print(f"Erro ao processar '{filename}': {e}")

    # Chamar a função de movimentação de arquivos com verificação
    move_files_with_check(files_to_move)

# Exemplo de uso:
input_directory = "C:/Users/cecil/OneDrive/Documents/GitHub/TensorMechanics/Main/Dataset/Data/sample_v3"
output_directory = "C:/Users/cecil/OneDrive/Documents/GitHub/TensorMechanics/Main/Dataset/Data/Organized_v3"

create_folders_for_resolution(input_directory, output_directory)

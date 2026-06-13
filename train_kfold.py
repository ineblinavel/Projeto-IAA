import os
import zipfile
import urllib.request
import json
import shutil
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Configurações gerais
DATASET_URL = "https://github.com/ineblinavel/Projeto-IAA/releases/download/1.0.0/dataset.zip"
ZIP_PATH = "dataset_kfold.zip"
EXTRACT_DIR = "dataset_kfold"
YOLO_DATA_DIR = "yolo_dataset_kfold"
NUM_FOLDS = 5
EPOCHS = 50
BATCH_SIZE = 16
MODEL_SIZE = "yolov8s"  # Opções: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x

def download_dataset():
    if os.path.exists(EXTRACT_DIR):
        print("Dataset já extraído. Pulando download.")
        return
    if not os.path.exists(ZIP_PATH):
        print(f"Baixando dataset de: {DATASET_URL}...")
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
        print("Download concluído!")
    
    print("Descompactando dataset...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(".")
    print("Dataset extraído com sucesso!")
    os.remove(ZIP_PATH)

def coco_to_yolo(coco_json_path, images_src_dir, labels_dest_dir):
    """Converte anotações no formato COCO para o formato YOLO."""
    os.makedirs(labels_dest_dir, exist_ok=True)
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
        
    # Mapear IDs de imagem para nomes de arquivos e dimensões
    images = {img['id']: img for img in coco_data['images']}
    
    # Agrupar anotações por imagem
    annotations_by_img = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_img:
            annotations_by_img[img_id] = []
        annotations_by_img[img_id].append(ann)
        
    # Gerar os arquivos .txt para cada imagem
    for img_id, img_info in images.items():
        file_name = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # O nome do arquivo txt deve coincidir com o da imagem (trocando a extensão)
        base_name = os.path.splitext(file_name)[0]
        txt_path = os.path.join(labels_dest_dir, f"{base_name}.txt")
        
        anns = annotations_by_img.get(img_id, [])
        yolo_lines = []
        
        for ann in anns:
            # COCO bbox: [x_min, y_min, width, height]
            bbox = ann['bbox']
            x_min, y_min, w, h = bbox
            
            # Converter para centro x, centro y, largura e altura (normalizados)
            x_center = (x_min + w / 2.0) / img_width
            y_center = (y_min + h / 2.0) / img_height
            w_norm = w / img_width
            h_norm = h / img_height
            
            # Garantir limites [0, 1]
            x_center = clip(x_center, 0.0, 1.0)
            y_center = clip(y_center, 0.0, 1.0)
            w_norm = clip(w_norm, 0.0, 1.0)
            h_norm = clip(h_norm, 0.0, 1.0)
            
            # A classe de árvores no dataset original é 0 (ou mapeada como 0)
            class_id = 0 
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            
        with open(txt_path, 'w') as f_txt:
            f_txt.write("\n".join(yolo_lines))

def clip(val, min_val, max_val):
    return max(min_val, min(val, max_val))

def prepare_yolo_dataset(num_folds):
    """Cria a estrutura de pastas do YOLO e converte os dados para cada fold."""
    if os.path.exists(YOLO_DATA_DIR):
        print("Estrutura YOLO já preparada. Pulando conversão.")
        return
        
    print("Preparando e convertendo dataset para o formato YOLO...")
    for k in range(num_folds):
        fold_src = os.path.join(EXTRACT_DIR, f"fold_{k}")
        fold_dest = os.path.join(YOLO_DATA_DIR, f"fold_{k}")
        
        for split in ['train', 'valid', 'test']:
            # Pastas de destino
            dest_images = os.path.join(fold_dest, split, 'images')
            dest_labels = os.path.join(fold_dest, split, 'labels')
            os.makedirs(dest_images, exist_ok=True)
            os.makedirs(dest_labels, exist_ok=True)
            
            # Mover imagens
            src_images_dir = os.path.join(fold_src, split)
            for file in os.listdir(src_images_dir):
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    shutil.copy(os.path.join(src_images_dir, file), os.path.join(dest_images, file))
            
            # Converter anotações do arquivo JSON correspondente
            coco_json = os.path.join(fold_src, split, f"_annotations.coco.json")
            if os.path.exists(coco_json):
                coco_to_yolo(coco_json, src_images_dir, dest_labels)

def create_yaml_config(fold_idx):
    """Gera o arquivo de configuração .yaml requerido pelo YOLOv8."""
    fold_dir = os.path.abspath(os.path.join(YOLO_DATA_DIR, f"fold_{fold_idx}"))
    yaml_content = f"""
path: {fold_dir}
train: train/images
val: valid/images
test: test/images

names:
  0: tree
"""
    yaml_path = os.path.join(YOLO_DATA_DIR, f"fold_{fold_idx}.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content.strip())
    return yaml_path

def train(folds, epochs, batch_size, model_size):
    download_dataset()
    prepare_yolo_dataset(folds)
    
    results_summary = []
    
    # Rodar o treinamento para cada um dos folds especificados
    for k in range(folds):
        print(f"\n==========================================")
        print(f"       INICIANDO TREINAMENTO - FOLD {k+1}/{folds}")
        print(f"==========================================")
        
        yaml_config = create_yaml_config(k)
        
        # Carregar modelo pré-treinado da Ultralytics
        model = YOLO(f"{model_size}.pt")
        
        # Executar treino
        # Nota: Usamos augment=True para habilitar os data augmentations padrão do YOLOv8
        results = model.train(
            data=yaml_config,
            epochs=epochs,
            batch=batch_size,
            imgsz=512,
            augment=True,
            val=True,
            save=True,
            project="tree_detection_runs",
            name=f"yolov8_fold_{k}",
            device=0, # Usa a GPU do Colab
        )
        
        # Rodar avaliação no conjunto de teste independente do fold
        print(f"Avaliando Fold {k+1} no conjunto de teste...")
        eval_results = model.val(split='test')
        map50 = eval_results.results_dict['metrics/mAP50(B)']
        results_summary.append(map50)
        
        print(f"mAP@50 do Fold {k+1} no Teste: {map50:.4f}")
        
        # No último fold do loop, exporta o melhor modelo para ONNX
        if k == folds - 1:
            print(f"Exportando modelo final do Fold {k+1} para o formato ONNX...")
            model.export(format="onnx", imgsz=512, optimize=True)
            
    print("\n" + "="*50)
    print("   RESUMO DA VALIDAÇÃO CRUZADA (YOLOv8s)   ")
    print("="*50)
    for idx, r in enumerate(results_summary):
        print(f"Fold {idx}: mAP@50 = {r:.4f}")
    print("-"*50)
    print(f"Média mAP@50: {np.mean(results_summary):.4f}")
    print(f"Desvio Padrão mAP@50: {np.std(results_summary):.4f}")
    print("="*50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5, help="Número de folds para treinar (1-5)")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas por fold")
    parser.add_argument("--batch", type=int, default=16, help="Tamanho do batch")
    parser.add_argument("--model", type=str, default="yolov8s", help="Modelo YOLOv8 (yolov8n, yolov8s, yolov8m)")
    args = parser.parse_args()
    
    train(folds=args.folds, epochs=args.epochs, batch_size=args.batch, model_size=args.model)


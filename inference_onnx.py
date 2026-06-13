import os
import cv2
import numpy as np
import onnxruntime as ort
import time

def nms(boxes, scores, iou_threshold):
    """Algoritmo clássico de Non-Maximum Suppression (NMS)."""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
        
    return keep

class YOLOv8ONNX:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Iniciar a sessão ONNX Runtime na CPU
        # Nota: Usamos CPU como padrão para permitir que rode em qualquer máquina
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # Obter informações dos inputs do modelo
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape  # Ex: [1, 3, 512, 512]
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
        # Obter classes a partir dos metadados
        self.classes = {}
        try:
            metadata = self.session.get_modelmeta().custom_metadata_map
            if 'names' in metadata:
                import ast
                self.classes = ast.literal_eval(metadata['names'])
                # Garantir chaves como inteiros
                self.classes = {int(k): v for k, v in self.classes.items()}
        except Exception:
            pass

    def preprocess(self, img):
        """Prepara a imagem de entrada para o formato do modelo."""
        self.img_height, self.img_width = img.shape[:2]
        
        # Redimensionar e converter cor BGR -> RGB
        img_resized = cv2.resize(img, (self.input_width, self.input_height))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalizar e reordenar dimensões (HWC -> CHW)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_transposed = img_normalized.transpose(2, 0, 1)
        
        # Adicionar dimensão do batch (1, 3, H, W)
        input_tensor = np.expand_dims(img_transposed, axis=0)
        return input_tensor

    def postprocess(self, outputs):
        """Processa as saídas do modelo para gerar as caixas finais."""
        # O formato da saída do YOLOv8 é [1, 4 + num_classes, num_candidates] (Ex: [1, 5, 5376])
        output = np.squeeze(outputs[0])
        if len(output.shape) == 3:
            output = output[0]
        output = output.T  # -> [num_candidates, 4 + num_classes]
        
        # YOLOv8: caixa de saída é [x_center, y_center, width, height, class_scores...]
        boxes_raw = output[:, :4]
        class_scores = output[:, 4:]
        
        scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
        
        # Filtrar pelo threshold de confiança
        mask = scores > self.conf_threshold
        boxes_filtered = boxes_raw[mask]
        scores_filtered = scores[mask]
        class_ids_filtered = class_ids[mask]
        
        if len(boxes_filtered) == 0:
            return [], [], []
            
        # Converter caixas de [x_center, y_center, w, h] para [x1, y1, x2, y2]
        boxes_converted = np.zeros_like(boxes_filtered)
        boxes_converted[:, 0] = boxes_filtered[:, 0] - boxes_filtered[:, 2] / 2.0  # x1
        boxes_converted[:, 1] = boxes_filtered[:, 1] - boxes_filtered[:, 3] / 2.0  # y1
        boxes_converted[:, 2] = boxes_filtered[:, 0] + boxes_filtered[:, 2] / 2.0  # x2
        boxes_converted[:, 3] = boxes_filtered[:, 1] + boxes_filtered[:, 3] / 2.0  # y2
        
        # Reescalar coordenadas para as dimensões originais da imagem
        scale_x = self.img_width / self.input_width
        scale_y = self.img_height / self.input_height
        
        boxes_converted[:, [0, 2]] *= scale_x
        boxes_converted[:, [1, 3]] *= scale_y
        
        # Aplicar NMS
        keep_indices = nms(boxes_converted, scores_filtered, self.iou_threshold)
        
        final_boxes = boxes_converted[keep_indices].astype(np.int32)
        final_scores = scores_filtered[keep_indices]
        final_class_ids = class_ids_filtered[keep_indices]
        
        # Preencher self.classes dinamicamente se necessário
        num_classes = class_scores.shape[1]
        for i in range(num_classes):
            if i not in self.classes:
                if num_classes == 1:
                    self.classes[i] = "Tree"
                else:
                    self.classes[i] = f"Class {i}"
        
        return final_boxes, final_scores, final_class_ids

    def detect(self, img):
        """Executa o pipeline completo: preprocess, run, postprocess."""
        input_tensor = self.preprocess(img)
        
        # Rodar inferência no ONNX Runtime
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Processar resultados
        boxes, scores, class_ids = self.postprocess(outputs)
        return boxes, scores, class_ids

def draw_detections(img, boxes, scores, class_ids=None, classes=None):
    """Desenha as detecções e exibe informações de contagem na imagem."""
    img_draw = img.copy()
    count = len(boxes)
    
    # Se class_ids ou classes não forem passados, cria defaults (classe única)
    if class_ids is None or len(class_ids) == 0:
        class_ids = [0] * len(boxes)
    if classes is None:
        classes = {0: "Tree"}
        
    # Cores por classe
    import random
    random.seed(42)
    colors = {}
    for cid in set(class_ids):
        name = classes.get(cid, '').lower()
        if name in ['tree', 'árvore', 'arvore']:
            colors[cid] = (0, 200, 0)  # Verde para árvores
        else:
            # Gerar cor RGB/BGR pseudo-aleatória agradável
            colors[cid] = (random.randint(50, 230), random.randint(50, 230), random.randint(50, 230))
            
    # Desenhar caixas e scores
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        color = colors.get(class_id, (0, 200, 0))
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        
        # Label com score
        class_name = classes.get(class_id, f"Class {class_id}")
        label = f"{class_name}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Fundo do label
        cv2.rectangle(img_draw, (x1, y1 - label_h - 4), (x1 + label_w + 4, y1), color, -1)
        cv2.putText(img_draw, label, (x1 + 2, y1 - 2), font, font_scale, (255, 255, 255), thickness)
        
    # Texto com contagem total de objetos detectados por classe
    y_offset = 35
    unique_ids, counts = np.unique(class_ids, return_counts=True)
    if len(class_ids) > 0:
        for cid, cnt in zip(unique_ids, counts):
            class_name = classes.get(cid, f"Class {cid}")
            text = f"Total {class_name}s: {cnt}"
            # Sombra para contraste
            cv2.putText(img_draw, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(img_draw, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 25
    else:
        cv2.putText(img_draw, "Total: 0", (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    return img_draw

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Caminho para o arquivo .onnx do modelo")
    parser.add_argument("--image", type=str, required=True, help="Caminho para a imagem de entrada")
    parser.add_argument("--output", type=str, default="output.jpg", help="Caminho para salvar a imagem com as detecções")
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Erro: Modelo não encontrado em {args.model}")
        exit(1)
        
    if not os.path.exists(args.image):
        print(f"Erro: Imagem não encontrada em {args.image}")
        exit(1)
        
    print(f"Carregando modelo ONNX: {args.model}")
    detector = YOLOv8ONNX(args.model)
    
    print(f"Carregando imagem: {args.image}")
    img = cv2.imread(args.image)
    
    # Medir tempo de processamento
    t_start = time.time()
    boxes, scores, class_ids = detector.detect(img)
    t_end = time.time()
    
    print(f"Detecção concluída em {(t_end - t_start)*1000:.1f}ms")
    print(f"Número de objetos encontrados: {len(boxes)}")
    
    # Desenhar caixas
    img_result = draw_detections(img, boxes, scores, class_ids, detector.classes)
    
    # Salvar resultado
    cv2.imwrite(args.output, img_result)
    print(f"Resultado salvo em: {args.output}")

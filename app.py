import streamlit as st
import cv2
import numpy as np
import os
import time
from PIL import Image
import io
from inference_onnx import YOLOv8ONNX, draw_detections

# Configuração da página Streamlit
st.set_page_config(
    page_title="Urban Canopy AI - Detector de Árvores",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS customizado (Premium Green Accents - Responsivo Light/Dark Mode)
st.markdown("""
<style>
    /* Estilização dos Botões com tema verde */
    .stButton>button {
        background-color: #2e6f40 !important;
        color: white !important;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
    }
    .stButton>button:hover {
        background-color: #1a4d2e !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    
    /* Cards de métricas adaptáveis ao tema Light/Dark usando variáveis do Streamlit */
    .metric-card {
        background-color: rgba(46, 111, 64, 0.05) !important;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #2e6f40;
        margin-bottom: 20px;
        border-top: 1px solid rgba(46, 111, 64, 0.1);
        border-right: 1px solid rgba(46, 111, 64, 0.1);
        border-bottom: 1px solid rgba(46, 111, 64, 0.1);
    }
    
    /* Destaque verde para o valor da métrica */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2e6f40 !important;
        line-height: 1.2;
    }
    
    /* Label da métrica (usa a cor padrão do tema para máximo contraste) */
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

def get_onnx_models():
    """Busca por arquivos .onnx no diretório atual e subdiretórios."""
    models = []
    # Buscar no diretório atual e pasta checkpoints
    paths_to_search = [".", "checkpoints", "YOLO-MS/mmyolo/checkpoints"]
    for path in paths_to_search:
        if os.path.exists(path):
            for file in os.listdir(path):
                if file.endswith(".onnx"):
                    models.append(os.path.join(path, file))
    return list(set(models))

def calculate_canopy_coverage(boxes, img_shape):
    """Calcula a porcentagem exata da área coberta pelas copas usando máscara binária."""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for box in boxes:
        x1, y1, x2, y2 = box
        # Garantir coordenadas dentro dos limites da imagem
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        mask[y1:y2, x1:x2] = 1
        
    covered_pixels = np.sum(mask)
    total_pixels = h * w
    percentage = (covered_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    return percentage

# --- INTERFACE ---

# Cabeçalho principal
st.title("🌳 Urban Canopy AI")
st.subheader("Detecção Avançada e Análise de Cobertura Arbórea Urbana")
st.markdown("Uma aplicação de Inteligência Artificial para identificar copas de árvores individuais e monitorar áreas verdes urbanas usando modelos YOLO otimizados para produção (ONNX).")

# Sidebar
st.sidebar.header("⚙️ Configurações do Modelo")

# Opção de origem do modelo
model_source = st.sidebar.radio(
    "Origem do Modelo",
    ["Modelo Padrão (Árvores)", "Upload de Modelo Customizado (.onnx)"]
)

model_path = None
is_custom_model = False

if model_source == "Modelo Padrão (Árvores)":
    model_path = "best.onnx"
    if not os.path.exists(model_path):
        import urllib.request
        with st.spinner("Baixando pesos padrão do modelo (42MB)... Por favor, aguarde."):
            try:
                url = "https://github.com/ineblinavel/Projeto-IAA/releases/download/1.0.0/best.onnx"
                urllib.request.urlretrieve(url, model_path)
                st.success("Download dos pesos concluído com sucesso!")
            except Exception as e:
                st.sidebar.error(f"Erro ao baixar pesos automaticamente: {e}")
else:
    is_custom_model = True
    uploaded_model = st.sidebar.file_uploader("Selecione o arquivo de pesos (.onnx)", type=["onnx"])
    if uploaded_model is not None:
        os.makedirs("checkpoints", exist_ok=True)
        model_path = os.path.join("checkpoints", "custom_model.onnx")
        with open(model_path, "wb") as f:
            f.write(uploaded_model.getbuffer())
        st.sidebar.success("Modelo customizado carregado com sucesso!")
    else:
        st.sidebar.info("Aguardando upload do arquivo de pesos .onnx...")

# Sliders de limites
conf_threshold = st.sidebar.slider("Limiar de Confiança (Confidence)", 0.05, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("Limiar de Supressão (IoU)", 0.1, 1.0, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🧠 Como Funciona?
Este aplicativo carrega um modelo **YOLOv8** convertido para o formato **ONNX**. 
Toda a inferência é feita usando o **ONNX Runtime** otimizado para **CPU**, eliminando a necessidade de uma GPU para realizar análises em tempo real.
""")

# Carregar detector
detector = None
if model_path and os.path.exists(model_path):
    try:
        detector = YOLOv8ONNX(model_path, conf_threshold, iou_threshold)
        st.sidebar.success(f"Modelo carregado com sucesso!")
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar modelo: {e}")
else:
    st.sidebar.warning("Aguardando carregamento do modelo...")

# Área de Upload de Imagem
uploaded_file = st.file_uploader("Selecione uma imagem aérea ou de drone (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

# Exemplo de imagem demo caso exista
demo_image_path = "YOLO-MS/mmyolo/demo/demo.jpg"
use_demo = st.checkbox("Usar imagem de demonstração (Exemplo do projeto)") if os.path.exists(demo_image_path) else False

input_img = None
if uploaded_file is not None:
    # Ler imagem enviada
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    input_img = cv2.imdecode(file_bytes, 1)
elif use_demo:
    input_img = cv2.imread(demo_image_path)

# Fluxo de Detecção
if input_img is not None:
    # Mostrar imagem original antes de processar
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📷 Imagem Original")
        st.image(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    with col2:
        st.markdown("### 🔍 Detecção do Modelo")
        
        if detector is not None:
            # Executar inferência
            t_start = time.time()
            boxes, scores, class_ids = detector.detect(input_img)
            t_end = time.time()
            
            inference_time = (t_end - t_start) * 1000
            
            # Identificar se existe classe de árvore para o cálculo de cobertura
            tree_class_ids = []
            for cid, name in detector.classes.items():
                if name.lower() in ['tree', 'árvore', 'arvore']:
                    tree_class_ids.append(cid)
            
            if tree_class_ids and len(class_ids) > 0:
                mask_tree = np.isin(class_ids, tree_class_ids)
                tree_boxes = boxes[mask_tree]
                canopy_percentage = calculate_canopy_coverage(tree_boxes, input_img.shape)
                coverage_label = "Cobertura de Copa Estimada (Árvores)"
            else:
                canopy_percentage = calculate_canopy_coverage(boxes, input_img.shape)
                coverage_label = "Cobertura Total Estimada (Geral)"
            
            # Desenhar na imagem com suporte multi-classe
            img_result = draw_detections(input_img, boxes, scores, class_ids, detector.classes)
            
            # Exibir imagem resultante
            st.image(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Converter imagem resultante para download
            is_success, buffer = cv2.imencode(".jpg", img_result)
            io_buf = io.BytesIO(buffer)
            
            # Botão de download
            st.download_button(
                label="📥 Baixar Imagem Processada",
                data=io_buf,
                file_name="deteccao_arvores.jpg",
                mime="image/jpeg"
            )
        else:
            st.info("Por favor, garanta que um modelo ONNX válido esteja carregado no painel lateral.")
            
    # Se detectou com sucesso, mostra métricas abaixo das imagens
    if detector is not None and input_img is not None:
        st.markdown("---")
        st.markdown("### 📊 Relatório Estatístico")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total de Detecções</div>
                <div class="metric-value">{len(boxes)}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{coverage_label}</div>
                <div class="metric-value">{canopy_percentage:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Tempo de Processamento (CPU)</div>
                <div class="metric-value">{inference_time:.1f} ms</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Detalhamento por classe se houver múltiplos tipos detectados
        if len(class_ids) > 0:
            st.markdown("#### 📋 Detalhamento por Classe")
            unique_ids, counts = np.unique(class_ids, return_counts=True)
            
            cols = st.columns(min(len(unique_ids), 5))
            for idx, (cid, cnt) in enumerate(zip(unique_ids, counts)):
                col_idx = idx % 5
                class_name = detector.classes.get(cid, f"Class {cid}").capitalize()
                with cols[col_idx]:
                    st.metric(label=class_name, value=cnt)
            
else:
    # Tela inicial sem imagem carregada
    st.info("Aguardando upload de imagem aérea para iniciar a análise.")
    
    # Adicionar seção informativa sobre os dados e relevância
    st.markdown("---")
    st.markdown("""
    ### 🌿 Sobre a Cobertura Arbórea Urbana
    A cobertura arbórea urbana refere-se à proporção de áreas municipais cobertas por folhas e ramos de árvores quando visualizadas de cima.
    
    O monitoramento da cobertura vegetal é vital para:
    * **Redução das Ilhas de Calor**: Árvores interceptam a radiação solar e resfriam o ar por evapotranspiração.
    * **Gestão de Águas Pluviais**: Reduzem o escoamento superficial de chuvas fortes, prevenindo enchentes.
    * **Sequestro de Carbono**: Absorvem dióxido de carbono e melhoram a qualidade do ar urbano.
    * **Saúde Humana**: O contato visual e físico com áreas verdes reduz níveis de estresse e melhora o bem-estar psicológico.
    """)

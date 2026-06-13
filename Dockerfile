# Utilizar imagem oficial Python slim para um container leve
FROM python:3.10-slim

# Definir diretório de trabalho no container
WORKDIR /app

# Instalar dependências básicas do sistema necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements.txt primeiro para cachear a instalação dos pacotes
COPY requirements.txt .

# Instalar as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo o código fonte para o container
COPY . .

# Expor a porta padrão utilizada pelo Streamlit
EXPOSE 8501

# Definir as variáveis de ambiente necessárias para o Streamlit rodar corretamente em containers
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Comando para executar o aplicativo Streamlit
CMD ["streamlit", "run", "app.py"]

FROM python:3.13

WORKDIR /app

# 1. Instalando dependências do sistema
RUN apt-get update && apt-get install -y \
    libmagic-dev \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    libglx-mesa0 \
    libegl1 \
    libx11-6 \
    libxcb1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-por \    
    && rm -rf /var/lib/apt/lists/*

# 2. Copiar apenas o requirements primeiro (otimiza o cache do Docker)
COPY requirements.txt .

# 3. Instalar dependências Python
RUN pip install -r requirements.txt

# 4. Copiar o restante do seu código (seu script .py, etc)
COPY . .

CMD ["/bin/bash"]
FROM python:3.10-slim

WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copie des dépendances
COPY requirements.frontend.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.frontend.txt

# Copie du code de l'application
COPY frontend.py .

# Exposition du port
EXPOSE 8501

# Commande de démarrage avec l'URL de l'API configurée
CMD ["sh", "-c", "streamlit run frontend.py --server.port=$PORT --server.address=0.0.0.0 --browser.serverAddress=0.0.0.0 --server.fileWatcherType none"]

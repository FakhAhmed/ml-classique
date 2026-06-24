# 1. L'OS de base : Un Linux minimaliste avec Python 3.11 pré-installé
FROM python:3.11-slim

# 2. Le dossier de travail à l'intérieur de la boîte
WORKDIR /app

# 3. On copie UNIQUEMENT notre fichier propre de dépendances d'abord
COPY requirements.txt .

# 4. On installe les outils (le cache est désactivé pour gagner de la place)
RUN pip install --no-cache-dir -r requirements.txt

# 5. On copie tout le reste du code (le dossier api et le dossier src)
COPY . .

# 6. La commande de démarrage obligatoire pour Google Cloud Run (Port 8080)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
# ================================================================
# Dockerfile – BANK_FRAUD_DETECTION
# Auteur : Papa Malick NDIAYE | Master DSGL – UADB
#
# Ce fichier décrit comment construire l'image Docker du projet.
# Une image c'est comme une "photo" de l'environnement complet :
# Python, bibliothèques, code — tout est dedans.
# ================================================================

# ── Étape 1 : image de base ──────────────────────────────────────
# On part d'une image Python 3.10 officielle version slim
# "slim" = version allégée sans outils inutiles (image plus petite)
FROM python:3.10-slim

# ── Étape 2 : métadonnées ────────────────────────────────────────
LABEL maintainer="Papa Malick NDIAYE <njaymika@gmail.com>"
LABEL description="Bank Fraud Detection – ML Pipeline + Flask API"
LABEL version="1.0"

# ── Étape 3 : dossier de travail dans le conteneur ───────────────
# Tous les fichiers du projet seront dans /app à l'intérieur du conteneur
WORKDIR /app

# ── Étape 4 : variables d'environnement ─────────────────────────
# PYTHONDONTWRITEBYTECODE : évite de créer des fichiers .pyc inutiles
# PYTHONUNBUFFERED : affiche les logs Python en temps réel
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# ── Étape 5 : installation des dépendances ───────────────────────
# On copie d'abord UNIQUEMENT requirements.txt
# Pourquoi ? Car Docker met en cache chaque étape.
# Si on ne change que le code (pas les dépendances),
# Docker ne réinstalle pas tout → build beaucoup plus rapide.
COPY requirements.txt .

# Mise à jour pip puis installation des dépendances
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Étape 6 : copie du code source ──────────────────────────────
# Maintenant on copie tout le reste du projet
# Le .dockerignore empêche de copier les fichiers inutiles
COPY . .

# ── Étape 7 : création des dossiers nécessaires ─────────────────
# On s'assure que les dossiers models/ et metrics/ existent
RUN mkdir -p models metrics data

# ── Étape 8 : entraînement du modèle ────────────────────────────
# On lance la pipeline ML pour entraîner et sauvegarder le modèle
# Cette étape tourne UNE SEULE FOIS lors du build de l'image
# NB : le fichier CSV doit être dans data/ avant le build
RUN if [ -f "data/creditcarddata.csv" ]; then \
        python main.py; \
    else \
        echo "⚠ CSV non trouvé – placez creditcarddata.csv dans data/ avant le build"; \
    fi

# ── Étape 9 : port exposé ────────────────────────────────────────
# On indique que le conteneur écoute sur le port 5000
# (c'est informatif, le vrai mapping se fait dans docker-compose)
EXPOSE 5000

# ── Étape 10 : commande de démarrage ────────────────────────────
# Commande lancée automatiquement quand le conteneur démarre
CMD ["python", "app/app.py"]

# ================================================================
# Makefile – BANK_FRAUD_DETECTION
# Auteur : Papa Malick NDIAYE | Master DSGL – UADB
# ================================================================

.PHONY: install run serve test clean docker-build docker-up docker-down docker-logs docker-clean help

help:
	@echo ""
	@echo "  BANK_FRAUD_DETECTION – Commandes disponibles"
	@echo "  ─────────────────────────────────────────────"
	@echo "  ── Local ──────────────────────────────────"
	@echo "  make install      →  Installer les dépendances"
	@echo "  make run          →  Lancer la pipeline ML complète"
	@echo "  make serve        →  Démarrer l'API Flask (port 5000)"
	@echo "  make test         →  Lancer les tests unitaires"
	@echo "  make clean        →  Supprimer les fichiers générés"
	@echo ""
	@echo "  ── Docker ─────────────────────────────────"
	@echo "  make docker-build →  Construire l'image Docker"
	@echo "  make docker-up    →  Lancer avec docker-compose"
	@echo "  make docker-down  →  Arrêter les conteneurs"
	@echo "  make docker-logs  →  Voir les logs en temps réel"
	@echo "  make docker-clean →  Supprimer images et volumes"
	@echo ""

# ── Commandes locales ───────────────────────────────────────────

install:
	pip install -r requirements.txt

run:
	python main.py

serve:
	python app/app.py

test:
	pytest tests/ -v --tb=short

clean:
	rm -f models/best_model.pkl models/scaler.pkl
	rm -f metrics/*.json metrics/*.png
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	@echo "Nettoyage terminé."

# ── Commandes Docker ────────────────────────────────────────────

docker-build:
	docker-compose build

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-clean:
	docker-compose down --rmi all --volumes
	@echo "Images et volumes Docker supprimés."

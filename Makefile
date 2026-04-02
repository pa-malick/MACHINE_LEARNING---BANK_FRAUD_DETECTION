# ================================================================
# Makefile – BANK_FRAUD_DETECTION
# Auteur : Papa Malick NDIAYE | Master DSGL – UADB
# ================================================================

.PHONY: install run serve test clean help

help:
	@echo ""
	@echo "  BANK_FRAUD_DETECTION – Commandes disponibles"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make install   →  Installer les dépendances"
	@echo "  make run       →  Lancer la pipeline ML complète"
	@echo "  make serve     →  Démarrer l'API Flask (port 5000)"
	@echo "  make test      →  Lancer les tests unitaires"
	@echo "  make clean     →  Supprimer les fichiers générés"
	@echo ""

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
	rm -f metrics/results.json metrics/*.png
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	@echo "Nettoyage terminé."

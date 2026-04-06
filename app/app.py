# ================================================================
# app.py  –  API Flask + interface web de prédiction
#
# Auteur : Papa Malick NDIAYE
# Master Data Science & Génie Logiciel – UADB
#
# Endpoints :
#   GET  /            → interface HTML
#   POST /predict     → prédiction JSON
#   GET  /metrics     → métriques des modèles
#   GET  /health      → statut de l'API
# ================================================================

import sys
import os
import numpy as np
from flask import Flask, request, jsonify, render_template

# Accès aux modules src depuis le dossier app/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from utils import charger_modele, charger_metriques

app = Flask(__name__)

# Chargement du modèle et du scaler au démarrage du serveur
modele, scaler = charger_modele(
    chemin_modele=os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl"),
    chemin_scaler=os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl"),
)

# Chargement des métriques
metriques = charger_metriques(
    os.path.join(os.path.dirname(__file__), "..", "metrics", "results.json")
)

# Noms des 13 features du dataset (dans l'ordre du CSV, hors PotentialFraud)
FEATURE_NAMES = [
    "CustomerID", "Age", "Income", "AccountBalance",
    "NumTransactions", "NumLatePayments", "CreditScore",
    "LoanAmount", "LoanDuration", "NumCreditCards",
    "HasLoan", "HasMortgage", "TransactionFrequency"
]


# ── Routes ───────────────────────────────────────────────────────

@app.route("/")
def index():
    """Page principale de l'interface de prédiction."""
    return render_template("index.html", metriques=metriques,
                           features=FEATURE_NAMES)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Reçoit un JSON  { "features": [val1, ..., val13] }
    Retourne        { "prediction": 0|1, "probabilite": float, "label": str }
    """
    try:
        data = request.get_json(force=True)

        if not data or "features" not in data:
            return jsonify({"erreur": "Champ 'features' manquant dans la requête."}), 400

        features = np.array(data["features"], dtype=float).reshape(1, -1)

        if features.shape[1] != len(FEATURE_NAMES):
            return jsonify({
                "erreur": f"Attendu {len(FEATURE_NAMES)} features, reçu {features.shape[1]}."
            }), 400

        # Normalisation avec le scaler entraîné
        features_norm = scaler.transform(features)

        prediction   = int(modele.predict(features_norm)[0])
        probabilite  = float(modele.predict_proba(features_norm)[0][1])
        label        = "🚨 Fraude détectée" if prediction == 1 else "✅ Transaction normale"

        return jsonify({
            "prediction" : prediction,
            "probabilite": round(probabilite * 100, 2),
            "label"      : label,
        })

    except ValueError as e:
        return jsonify({"erreur": f"Valeur invalide : {e}"}), 400
    except Exception as e:
        return jsonify({"erreur": str(e)}), 500


@app.route("/metrics")
def get_metrics():
    """Retourne les métriques de tous les modèles au format JSON."""
    return jsonify(metriques)


@app.route("/health")
def health():
    """Endpoint de monitoring – vérifie que l'API est opérationnelle."""
    return jsonify({"status": "ok", "message": "API opérationnelle ✔"})


# ── Lancement ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚀  API Flask démarrée sur  http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)

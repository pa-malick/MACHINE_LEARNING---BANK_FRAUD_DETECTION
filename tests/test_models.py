# ================================================================
# test_models.py  –  Tests unitaires : modèles ML
# Auteur : Papa Malick NDIAYE | Master DSGL – UADB
# ================================================================

import sys, os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from src.models import get_modeles, entrainer_modeles


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def donnees():
    """Données synthétiques simples pour les tests."""
    np.random.seed(0)
    X_train = np.random.rand(200, 13)
    y_train = np.random.randint(0, 2, 200)
    X_test  = np.random.rand(60, 13)
    y_test  = np.random.randint(0, 2, 60)
    return X_train, X_test, y_train, y_test


# ── get_modeles ──────────────────────────────────────────────────

class TestGetModeles:

    def test_nombre_modeles(self):
        assert len(get_modeles()) == 5

    def test_noms_attendus(self):
        noms = set(get_modeles().keys())
        attendus = {
            "Régression Logistique", "Random Forest",
            "Gradient Boosting", "SVM", "KNN"
        }
        assert noms == attendus

    def test_interface_sklearn(self):
        for nom, m in get_modeles().items():
            assert hasattr(m, "fit"),           f"{nom} : méthode fit manquante"
            assert hasattr(m, "predict"),       f"{nom} : méthode predict manquante"
            assert hasattr(m, "predict_proba"), f"{nom} : méthode predict_proba manquante"


# ── entrainer_modeles ────────────────────────────────────────────

class TestEntrainerModeles:

    def test_tous_entraines(self, donnees):
        X_tr, _, y_tr, _ = donnees
        entraines = entrainer_modeles(get_modeles(), X_tr, y_tr)
        assert len(entraines) == 5

    def test_predictions_valides(self, donnees):
        X_tr, X_te, y_tr, _ = donnees
        entraines = entrainer_modeles(get_modeles(), X_tr, y_tr)
        for nom, m in entraines.items():
            preds = m.predict(X_te)
            assert len(preds) == len(X_te),         f"{nom} : mauvaise taille"
            assert set(preds).issubset({0, 1}),     f"{nom} : labels invalides"

    def test_probabilites_entre_0_et_1(self, donnees):
        X_tr, X_te, y_tr, _ = donnees
        entraines = entrainer_modeles(get_modeles(), X_tr, y_tr)
        for nom, m in entraines.items():
            probas = m.predict_proba(X_te)
            assert probas.shape == (len(X_te), 2),  f"{nom} : forme incorrecte"
            assert (probas >= 0).all(),              f"{nom} : proba < 0"
            assert (probas <= 1).all(),              f"{nom} : proba > 1"

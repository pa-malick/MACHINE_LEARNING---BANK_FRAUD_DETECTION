# ================================================================
# test_preprocessing.py  –  Tests unitaires : preprocessing
# Auteur : Papa Malick NDIAYE | Master DSGL – UADB
# ================================================================

import sys, os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from src.preprocessing import (verifier_qualite, preparer_features,
                            diviser_et_normaliser)


# ── Fixture partagée ─────────────────────────────────────────────

@pytest.fixture
def df_base():
    """Dataset synthétique minimal (100 observations, 5 variables)."""
    np.random.seed(42)
    return pd.DataFrame({
        "Age"            : np.random.randint(20, 70, 100).astype(float),
        "Income"         : np.random.uniform(500, 10000, 100),
        "AccountBalance" : np.random.uniform(0, 50000, 100),
        "CreditScore"    : np.random.randint(300, 850, 100).astype(float),
        "PotentialFraud" : np.random.randint(0, 2, 100),
    })


# ── verifier_qualite ─────────────────────────────────────────────

class TestVerifierQualite:

    def test_supprime_doublons(self, df_base):
        df = pd.concat([df_base, df_base.iloc[:5]], ignore_index=True)
        assert df.shape[0] == 105
        df_clean = verifier_qualite(df)
        assert df_clean.shape[0] == 100

    def test_corrige_valeurs_manquantes(self, df_base):
        df_base.loc[0, "Age"]    = np.nan
        df_base.loc[3, "Income"] = np.nan
        df_clean = verifier_qualite(df_base)
        assert df_clean.isnull().sum().sum() == 0

    def test_retourne_dataframe(self, df_base):
        assert isinstance(verifier_qualite(df_base), pd.DataFrame)

    def test_variable_cible_non_clampee(self, df_base):
        df_clean = verifier_qualite(df_base)
        # PotentialFraud doit toujours valoir 0 ou 1
        assert set(df_clean["PotentialFraud"].unique()).issubset({0, 1})


# ── preparer_features ────────────────────────────────────────────

class TestPreparerFeatures:

    def test_cible_absente_de_X(self, df_base):
        X, y = preparer_features(df_base)
        assert "PotentialFraud" not in X.columns

    def test_longueurs_coherentes(self, df_base):
        X, y = preparer_features(df_base)
        assert len(X) == len(y) == len(df_base)

    def test_nom_cible_correct(self, df_base):
        _, y = preparer_features(df_base)
        assert y.name == "PotentialFraud"


# ── diviser_et_normaliser ────────────────────────────────────────

class TestDiviserNormaliser:

    def test_proportion_test_approx(self, df_base):
        X, y = preparer_features(df_base)
        X_tr, X_te, _, _, _ = diviser_et_normaliser(X, y, taille_test=0.30)
        total = len(X_tr) + len(X_te)
        assert abs(len(X_te) / total - 0.30) < 0.05

    def test_somme_egale_dataset(self, df_base):
        X, y = preparer_features(df_base)
        X_tr, X_te, y_tr, y_te, _ = diviser_et_normaliser(X, y)
        assert len(X_tr) + len(X_te) == len(X)
        assert len(y_tr) + len(y_te) == len(y)

    def test_scaler_retourne(self, df_base):
        from sklearn.preprocessing import StandardScaler
        X, y = preparer_features(df_base)
        _, _, _, _, scaler = diviser_et_normaliser(X, y)
        assert isinstance(scaler, StandardScaler)

    def test_normalisation_moyenne_proche_zero(self, df_base):
        X, y = preparer_features(df_base)
        X_tr, _, _, _, _ = diviser_et_normaliser(X, y)
        assert abs(X_tr.mean()) < 0.5

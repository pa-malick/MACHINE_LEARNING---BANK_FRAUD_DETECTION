# ================================================================
# preprocessing.py  –  Nettoyage, split et normalisation
#
# Auteur : Papa Malick NDIAYE
# Master Data Science & Génie Logiciel – UADB
# ================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


# ── 1. Qualité des données ───────────────────────────────────────

def verifier_qualite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vérifie et corrige trois problèmes classiques :
      1. Valeurs manquantes  →  remplacement par la médiane
      2. Doublons            →  suppression
      3. Valeurs aberrantes  →  winsorisation par IQR

    Paramètres
    ----------
    df : pd.DataFrame  –  dataset brut

    Retourne
    --------
    df : pd.DataFrame  –  dataset nettoyé
    """
    print("\n┌─ PRÉPARATION DES DONNÉES ─────────────────────────────┐")

    # --- Valeurs manquantes ---
    nb_manq = df.isnull().sum().sum()
    print(f"\n  [1] Valeurs manquantes : {nb_manq}")
    if nb_manq > 0:
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                med = df[col].median()
                df[col].fillna(med, inplace=True)
                print(f"      → '{col}' : remplacé par médiane = {med:.2f}")
        print("      ✔ Corrigé")
    else:
        print("      ✔ Aucune valeur manquante")

    # --- Doublons ---
    nb_dup = df.duplicated().sum()
    print(f"\n  [2] Doublons : {nb_dup}")
    if nb_dup > 0:
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"      ✔ {nb_dup} doublon(s) supprimé(s)")
    else:
        print("      ✔ Aucun doublon")

    # --- Valeurs aberrantes (IQR) ---
    print("\n  [3] Valeurs aberrantes (méthode IQR) :")
    cols_num = [c for c in df.select_dtypes(include=[np.number]).columns
                if c != "PotentialFraud"]
    total_ab = 0
    for col in cols_num:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        borne_inf = Q1 - 1.5 * IQR
        borne_sup = Q3 + 1.5 * IQR
        nb_ab = ((df[col] < borne_inf) | (df[col] > borne_sup)).sum()
        if nb_ab > 0:
            df[col] = df[col].clip(borne_inf, borne_sup)
            total_ab += nb_ab
            print(f"      → '{col}' : {nb_ab} valeur(s) clampée(s)")

    if total_ab == 0:
        print("      ✔ Aucune valeur aberrante")
    else:
        print(f"      ✔ Total corrigé : {total_ab} valeur(s)")

    print(f"\n  → Taille finale : {df.shape[0]} × {df.shape[1]}")
    print("└───────────────────────────────────────────────────────┘\n")
    return df


# ── 2. Séparation features / cible ──────────────────────────────

def preparer_features(df: pd.DataFrame, cible: str = "PotentialFraud"):
    """
    Sépare les variables explicatives de la variable cible.

    Retourne
    --------
    X : pd.DataFrame  –  features
    y : pd.Series     –  cible
    """
    X = df.drop(columns=[cible])
    y = df[cible]
    print(f"[✔] Features : {X.shape[1]} variables  |  Cible : '{cible}'")
    return X, y


# ── 3. Split + normalisation ────────────────────────────────────

def diviser_et_normaliser(X, y, taille_test: float = 0.30):
    """
    Divise le dataset (70 % train / 30 % test) et applique
    une normalisation StandardScaler (fit sur le train uniquement).

    Retourne
    --------
    X_train, X_test, y_train, y_test, scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=taille_test,
        random_state=42,
        stratify=y          # conserve la proportion des classes
    )
    print(f"[✔] Split  →  train : {len(X_train)}  |  test : {len(X_test)}")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    print("[✔] Normalisation StandardScaler appliquée")

    return X_train, X_test, y_train, y_test, scaler


# ── 4. Rééquilibrage SMOTE ───────────────────────────────────────

def traiter_desequilibre(X_train, y_train):
    """
    Applique SMOTE sur les données d'entraînement pour corriger
    le déséquilibre de classes (sur-échantillonnage de la minorité).

    Note : SMOTE ne doit jamais être appliqué sur les données de test.

    Retourne
    --------
    X_res, y_res : données rééquilibrées
    """
    print("\n[SMOTE] Rééquilibrage des classes :")
    print(f"  Avant  →  {dict(pd.Series(y_train).value_counts())}")

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"  Après  →  {dict(pd.Series(y_res).value_counts())}")
    print("[✔] SMOTE appliqué avec succès\n")
    return X_res, y_res

# ================================================================
# data_loader.py  –  Chargement et premier aperçu du dataset
#
# Auteur : Papa Malick NDIAYE
# Master Data Science & Génie Logiciel – UADB
# ================================================================

import os
import pandas as pd


def charger_donnees(chemin: str) -> pd.DataFrame:
    """
    Charge le fichier CSV dans un DataFrame pandas.

    Paramètres
    ----------
    chemin : str
        Chemin absolu ou relatif vers le fichier CSV.

    Retourne
    --------
    df : pd.DataFrame
        Le jeu de données brut.
    """
    if not os.path.exists(chemin):
        raise FileNotFoundError(
            f"Fichier introuvable : '{chemin}'\n"
            "→ Vérifiez que 'creditcarddata.csv' est bien dans le dossier data/"
        )

    df = pd.read_csv(chemin)
    print(f"[✔] Données chargées  :  {df.shape[0]} observations  ×  {df.shape[1]} variables")
    return df


def apercu_donnees(df: pd.DataFrame) -> None:
    """
    Affiche un résumé complet du dataset :
    premières lignes, types, statistiques, distribution de la cible.

    Paramètres
    ----------
    df : pd.DataFrame
    """
    sep = "─" * 55

    print(f"\n{sep}")
    print("  APERÇU DU DATASET")
    print(sep)

    print("\n── Premières lignes ──")
    print(df.head(5).to_string())

    print("\n── Infos générales ──")
    print(df.info())

    print("\n── Statistiques descriptives ──")
    print(df.describe().round(2).to_string())

    print("\n── Distribution de la variable cible : PotentialFraud ──")
    counts = df["PotentialFraud"].value_counts()
    pcts   = df["PotentialFraud"].value_counts(normalize=True).mul(100).round(2)
    for label, cnt in counts.items():
        tag = "Fraude" if label == 1 else "Normal"
        print(f"  {tag} ({label})  :  {cnt} observations  ({pcts[label]} %)")
    print()

# ================================================================
# main.py  –  Pipeline complète : du CSV au modèle déployable
#
# Auteur : Papa Malick NDIAYE
# Master Data Science & Génie Logiciel – UADB
#
# Usage :
#   python main.py
#   python main.py --data /chemin/vers/creditcarddata.csv
# ================================================================

import sys
import os
import argparse

# Rendre le dossier src importable peu importe d'où on lance le script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data_loader   import charger_donnees, apercu_donnees
from src.preprocessing import (verifier_qualite, preparer_features,
                            diviser_et_normaliser, traiter_desequilibre)
from src.models        import get_modeles, entrainer_modeles
from src.evaluation    import (evaluer_modele, comparer_modeles,
                            sauvegarder_metriques)
from src.utils         import sauvegarder_modele


CHEMIN_PAR_DEFAUT = "data/creditcarddata.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline ML – Détection de fraude bancaire"
    )
    parser.add_argument(
        "--data", type=str, default=CHEMIN_PAR_DEFAUT,
        help="Chemin vers le fichier CSV (défaut : data/creditcarddata.csv)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("   BANK FRAUD DETECTION  –  Pipeline Machine Learning")
    print("   Papa Malick NDIAYE | Master DSGL | UADB")
    print("=" * 60)

    # ── Étape 1 : Chargement ────────────────────────────────────
    print("\n[ÉTAPE 1 / 6]  Chargement des données")
    df = charger_donnees(args.data)
    apercu_donnees(df)

    # ── Étape 2 : Préparation ───────────────────────────────────
    print("\n[ÉTAPE 2 / 6]  Préparation des données")
    df = verifier_qualite(df)
    X, y = preparer_features(df, cible="PotentialFraud")
    X_train, X_test, y_train, y_test, scaler = diviser_et_normaliser(X, y)
    X_train, y_train = traiter_desequilibre(X_train, y_train)

    # ── Étape 3 : Entraînement ──────────────────────────────────
    print("\n[ÉTAPE 3 / 6]  Entraînement des modèles")
    modeles = get_modeles()
    modeles_entraines = entrainer_modeles(modeles, X_train, y_train)

    # ── Étape 4 : Évaluation ────────────────────────────────────
    print("\n[ÉTAPE 4 / 6]  Évaluation sur les données de test")
    print("┌─ RÉSULTATS ───────────────────────────────────────────┐")
    resultats = {}
    for nom, modele in modeles_entraines.items():
        resultats[nom] = evaluer_modele(nom, modele, X_test, y_test)
    print("└───────────────────────────────────────────────────────┘")

    # ── Étape 5 : Comparaison ───────────────────────────────────
    print("\n[ÉTAPE 5 / 6]  Comparaison des modèles")
    meilleur_nom = comparer_modeles(resultats)

    # ── Étape 6 : Sauvegarde ────────────────────────────────────
    print("\n[ÉTAPE 6 / 6]  Sauvegarde")
    meilleur_modele = modeles_entraines[meilleur_nom]
    sauvegarder_modele(meilleur_modele, scaler)
    sauvegarder_metriques(resultats, meilleur_nom)

    print("\n" + "=" * 60)
    print(f"Pipeline terminée avec succès !")
    print(f"Meilleur modèle  : {meilleur_nom}")
    print(f"Métriques        : metrics/results.json")
    print(f"Graphiques       : metrics/")
    print(f"Modèle           : models/best_model.pkl")
    print("=" * 60)
    print("\n  → Lancez maintenant l'API :  python app/app.py\n")


if __name__ == "__main__":
    main()

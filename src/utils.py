# ================================================================
# utils.py  –  Sauvegarde et chargement du modèle
#
# Auteur : Papa Malick NDIAYE
# Master Data Science & Génie Logiciel – UADB
# ================================================================

import os
import json
import joblib


def sauvegarder_modele(modele, scaler,
                       chemin_modele : str = "models/best_model.pkl",
                       chemin_scaler : str = "models/scaler.pkl") -> None:
    """
    Sérialise le modèle et le scaler avec joblib.

    Paramètres
    ----------
    modele        : modèle sklearn entraîné
    scaler        : StandardScaler ajusté sur X_train
    chemin_modele : chemin de sauvegarde du modèle
    chemin_scaler : chemin de sauvegarde du scaler
    """
    os.makedirs("models", exist_ok=True)
    joblib.dump(modele, chemin_modele)
    joblib.dump(scaler, chemin_scaler)
    print(f"[✔] Modèle sauvegardé  : {chemin_modele}")
    print(f"[✔] Scaler sauvegardé  : {chemin_scaler}")


def charger_modele(chemin_modele : str = "models/best_model.pkl",
                   chemin_scaler : str = "models/scaler.pkl"):
    """
    Charge le modèle et le scaler depuis le disque.

    Retourne
    --------
    (modele, scaler)
    """
    for chemin in (chemin_modele, chemin_scaler):
        if not os.path.exists(chemin):
            raise FileNotFoundError(
                f"Fichier introuvable : '{chemin}'\n"
                "→ Lancez d'abord :  python main.py"
            )

    modele = joblib.load(chemin_modele)
    scaler = joblib.load(chemin_scaler)
    print(f"[✔] Modèle chargé : {chemin_modele}")
    return modele, scaler


def charger_metriques(chemin: str = "metrics/results.json") -> dict:
    """
    Charge le fichier JSON des métriques.

    Retourne
    --------
    dict  –  contenu du fichier, ou dict vide si absent
    """
    if not os.path.exists(chemin):
        return {}
    with open(chemin, "r", encoding="utf-8") as f:
        return json.load(f)

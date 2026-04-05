# ================================================================
# evaluation.py  –  Évaluation, comparaison et visualisation
#
# Auteur : Papa Malick NDIAYE
# Master Data Science & Génie Logiciel – UADB
# ================================================================

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")          # pas besoin d'affichage graphique
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)


# ── 1. Évaluation d'un modèle ───────────────────────────────────

def evaluer_modele(nom: str, modele, X_test, y_test) -> dict:
    """
    Évalue un modèle et affiche :
      - les métriques principales
      - le rapport de classification complet
      - la matrice de confusion (sauvegardée en PNG)

    Retourne
    --------
    dict  –  { accuracy, precision, recall, f1_score }
    """
    y_pred = modele.predict(X_test)

    metriques = {
        "accuracy"  : round(float(accuracy_score(y_test, y_pred)), 4),
        "precision" : round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall"    : round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score"  : round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
    }

    print(f"\n  ── {nom} ──")
    print(f"     Accuracy  : {metriques['accuracy']  * 100:.2f} %")
    print(f"     Précision : {metriques['precision'] * 100:.2f} %")
    print(f"     Rappel    : {metriques['recall']    * 100:.2f} %")
    print(f"     F1-Score  : {metriques['f1_score']  * 100:.2f} %")
    print("\n  Rapport complet :")
    print(classification_report(y_test, y_pred,
                                target_names=["Normal", "Fraude"],
                                zero_division=0))

    _tracer_matrice_confusion(nom, y_test, y_pred)
    return metriques


# ── 2. Matrice de confusion ──────────────────────────────────────

def _tracer_matrice_confusion(nom: str, y_test, y_pred) -> None:
    """
    Trace la matrice de confusion et l'enregistre dans metrics/.
    Affiche également une analyse textuelle des cases.
    """
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Analyse métier de chaque case
    print("  Analyse de la matrice de confusion :")
    print(f"    TN = {tn:4d}  → clients normaux correctement identifiés")
    print(f"    FP = {fp:4d}  → clients normaux signalés à tort (fausse alarme)")
    print(f"    FN = {fn:4d}  → fraudes non détectées  ⚠ risque métier élevé")
    print(f"    TP = {tp:4d}  → fraudes détectées avec succès")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Fraude"],
        yticklabels=["Normal", "Fraude"],
        linewidths=0.5, ax=ax
    )
    ax.set_title(f"Matrice de confusion\n{nom}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Prédit", fontsize=9)
    ax.set_ylabel("Réel",   fontsize=9)

    os.makedirs("metrics", exist_ok=True)
    fichier = nom.lower().replace(" ", "_").replace("é", "e").replace("è", "e")
    chemin  = f"metrics/cm_{fichier}.png"
    plt.tight_layout()
    plt.savefig(chemin, dpi=150)
    plt.close()
    print(f"    → Sauvegardée : {chemin}\n")


# ── 3. Comparaison globale ───────────────────────────────────────

def comparer_modeles(resultats: dict) -> str:
    """
    Affiche un tableau récapitulatif, génère un graphique comparatif
    et retourne le nom du meilleur modèle (critère : F1-score).

    On utilise le F1-score comme critère principal car il équilibre
    précision et rappel, ce qui est crucial quand les classes sont
    déséquilibrées (beaucoup plus de transactions normales que frauduleuses).
    """
    print("┌─ COMPARAISON DES MODÈLES ─────────────────────────────┐")
    print(f"  {'Modèle':<28} {'Accuracy':>9} {'Précision':>10} {'Rappel':>8} {'F1':>8}")
    print("  " + "─" * 60)

    for nom, m in resultats.items():
        print(
            f"  {nom:<28}"
            f"  {m['accuracy']  * 100:>7.2f}%"
            f"  {m['precision'] * 100:>8.2f}%"
            f"  {m['recall']    * 100:>6.2f}%"
            f"  {m['f1_score']  * 100:>6.2f}%"
        )

    meilleur = max(resultats, key=lambda k: resultats[k]["f1_score"])
    print(f"\n  🏆  Meilleur modèle (F1) : {meilleur}  "
          f"→  {resultats[meilleur]['f1_score']*100:.2f}%")
    print("└───────────────────────────────────────────────────────┘\n")

    _tracer_comparaison(resultats, meilleur)
    return meilleur


def _tracer_comparaison(resultats: dict, meilleur: str) -> None:
    """
    Génère un graphique en barres groupées comparant les 4 métriques
    pour tous les modèles.
    """
    noms = list(resultats.keys())
    acc  = [resultats[n]["accuracy"]  * 100 for n in noms]
    prec = [resultats[n]["precision"] * 100 for n in noms]
    rec  = [resultats[n]["recall"]    * 100 for n in noms]
    f1   = [resultats[n]["f1_score"]  * 100 for n in noms]

    x = np.arange(len(noms))
    w = 0.18

    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - 1.5*w, acc,  w, label="Accuracy",  color="#3b82f6", edgecolor="white")
    b2 = ax.bar(x - 0.5*w, prec, w, label="Précision", color="#10b981", edgecolor="white")
    b3 = ax.bar(x + 0.5*w, rec,  w, label="Rappel",    color="#f59e0b", edgecolor="white")
    b4 = ax.bar(x + 1.5*w, f1,   w, label="F1-Score",  color="#ef4444", edgecolor="white")

    # Valeurs au-dessus des barres
    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(noms, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score (%)", fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_title("Comparaison des modèles – Détection de fraude bancaire",
                 fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Surligner le meilleur modèle
    idx = noms.index(meilleur)
    ax.axvspan(idx - 0.45, idx + 0.45, alpha=0.06, color="#ef4444",
               label=f"Meilleur : {meilleur}")

    plt.tight_layout()
    plt.savefig("metrics/comparaison_modeles.png", dpi=150)
    plt.close()
    print("[✔] Graphique sauvegardé : metrics/comparaison_modeles.png")


# ── 4. Sauvegarde des métriques JSON ────────────────────────────

def sauvegarder_metriques(resultats: dict, meilleur: str) -> None:
    """
    Exporte toutes les métriques dans metrics/results.json.
    Ce fichier est lu par l'API Flask et l'interface web.
    """
    os.makedirs("metrics", exist_ok=True)
    payload = {"meilleur_modele": meilleur, "resultats": resultats}
    with open("metrics/results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    print("[✔] Métriques exportées : metrics/results.json")


# ── 5. Autre visualisation ───────────────────────────────────────

def tracer_importance_features(modele, feature_names, nom="Random Forest"):
    """
    Affiche les variables les plus importantes pour la prédiction.
    Uniquement pour les modèles qui ont feature_importances_
    (Random Forest, Gradient Boosting, Decision Tree)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # récupère l'importance de chaque variable
    importances = modele.feature_importances_
    
    # trie du plus important au moins important
    indices = np.argsort(importances)[::-1]
    noms_tries = [feature_names[i] for i in indices]
    vals_tries = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(noms_tries, vals_tries, color="#3b82f6", edgecolor="white")
    ax.set_title(f"Importance des variables – {nom}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Variables")
    ax.set_ylabel("Importance")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    os.makedirs("metrics", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"metrics/feature_importance_{nom}.png", dpi=150)
    plt.close()
    print(f"[✔] Feature importance sauvegardée : metrics/feature_importance_{nom}.png")
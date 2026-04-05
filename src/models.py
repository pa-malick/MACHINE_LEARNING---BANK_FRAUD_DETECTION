# ================================================================
# models.py  –  Définition et entraînement des modèles ML
#
# Auteur : Papa Malick NDIAYE
# Master Data Science & Génie Logiciel – UADB
#
# Modèles retenus pour la détection de fraude :
#   1. Régression Logistique  – référence linéaire, interprétable
#   2. Random Forest          – robuste, gère bien les non-linéarités
#   3. Gradient Boosting      – souvent le meilleur sur données tabulaires
#   4. SVM (RBF)              – efficace sur petits/moyens datasets
#   5. KNN                    – simple, utile comme baseline non-paramétrique
# ================================================================

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm           import SVC
from sklearn.neighbors     import KNeighborsClassifier


def get_modeles() -> dict:
    """
    Instancie tous les modèles avec des hyperparamètres de départ
    raisonnables pour une première exécution.

    Retourne
    --------
    dict  –  { nom_modele : instance_sklearn }
    """
    return {
        "Régression Logistique": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced"    # compense le déséquilibre résiduel
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,    # nombre d'arbres augmenté pour plus de stabilité
            max_depth=15,        #  arbres plus profonds pour capturer les interactions
            min_samples_leaf=1,  # chaque feuille peut avoir 1 seul exemple
            random_state=42,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.8,             # légère régularisation
            random_state=42
        ),

        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,          # nécessaire pour predict_proba()
            random_state=42,
            class_weight="balanced"
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=7,
            metric="euclidean",
            n_jobs=-1
        ),
    }


def entrainer_modeles(modeles: dict, X_train, y_train) -> dict:
    """
    Entraîne chaque modèle sur les données d'entraînement.

    Paramètres
    ----------
    modeles  : dict   –  modèles non entraînés
    X_train  : array  –  features d'entraînement
    y_train  : array  –  labels d'entraînement

    Retourne
    --------
    dict  –  modèles entraînés
    """
    print("┌─ ENTRAÎNEMENT ────────────────────────────────────────┐")
    entraines = {}
    for nom, modele in modeles.items():
        print(f"  → {nom} ...", end=" ", flush=True)
        modele.fit(X_train, y_train)
        entraines[nom] = modele
        print("✔")
    print("└───────────────────────────────────────────────────────┘\n")
    return entraines

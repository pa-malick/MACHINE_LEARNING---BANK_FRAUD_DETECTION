🌐 **Démo live :** https://machine-learning-bank-fraud-detection.onrender.com/
<div align="center">

# 🏦 BANK FRAUD DETECTION

### Détection automatique de fraude bancaire par Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen?style=flat-square)]()

**Papa Malick NDIAYE** · Master Data Science & Génie Logiciel · Université Alioune Diop de Bambey

</div>

---

## 📌 Contexte & Problématique

La fraude bancaire représente des milliards de dollars de pertes chaque année pour les institutions financières. Les systèmes de détection manuelle sont lents, coûteux et peu scalables.

Ce projet propose une **solution IA complète** : de la préparation des données jusqu'au déploiement d'une API Flask, en passant par la comparaison de 5 modèles de Machine Learning, pour identifier automatiquement les transactions frauduleuses **avant qu'elles causent des dommages**.

> *"Minimiser les faux négatifs (fraudes non détectées) est la priorité métier absolue."*

---

## ✨ Fonctionnalités

- ✅ Pipeline ML complète et modulaire (preprocessing → modélisation → évaluation)
- ✅ 5 modèles comparés : Régression Logistique, Random Forest, Gradient Boosting, SVM, KNN
- ✅ Gestion du déséquilibre de classe par **SMOTE**
- ✅ Matrices de confusion commentées + graphique comparatif
- ✅ API Flask avec endpoint de prédiction en temps réel
- ✅ Interface web responsive
- ✅ Métriques exportées en JSON (versionnage léger)
- ✅ Tests unitaires pytest
- ✅ Makefile pour automatiser toutes les commandes

---

## 🗂️ Structure du projet

```
BANK_FRAUD_DETECTION/
│
├── data/
│   └── creditcarddata.csv          ← dataset brut (à placer ici)
│
├── src/
│   ├── data_loader.py              ← chargement & aperçu
│   ├── preprocessing.py            ← nettoyage, split, SMOTE
│   ├── models.py                   ← définition des 5 modèles
│   ├── evaluation.py               ← métriques, matrices, graphiques
│   └── utils.py                    ← save/load modèle & métriques
│
├── app/
│   ├── app.py                      ← serveur Flask
│   ├── templates/index.html        ← interface web
│   └── static/
│       ├── style.css
│       └── script.js
│
├── models/
│   ├── best_model.pkl              ← meilleur modèle (généré)
│   └── scaler.pkl                  ← scaler (généré)
│
├── metrics/
│   ├── results.json                ← métriques JSON (généré)
│   ├── comparaison_modeles.png     ← graphique (généré)
│   └── cm_*.png                    ← matrices de confusion (générées)
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_models.py
│
├── main.py                         ← pipeline complète
├── requirements.txt
├── Makefile
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md
```

---

## 🚀 Installation & Utilisation

```
### 1. Cloner le projet

```bash
git clone https://github.com/pa-malick/MACHINE_LEARNING---BANK_FRAUD_DETECTION.git
cd BANK_FRAUD_DETECTION
```

### 2. Installer les dépendances

```bash
make install
# ou
pip install -r requirements.txt
```

### 3. Placer le dataset

```bash
# Copier votre fichier CSV dans le dossier data/
cp /chemin/vers/creditcarddata.csv data/
```

### 4. Lancer la pipeline ML

```bash
make run
# ou
python main.py
```

### 5. Démarrer l'API Flask

```bash
make serve
# ou
python app/app.py
```

## 🐳 Lancer avec Docker
```bash
# 1. Placer le dataset
cp /chemin/vers/creditcarddata.csv data/

# 2. Construire et lancer
docker-compose up --build


Accédez à **http://localhost:5000** dans votre navigateur.

### 6. Lancer les tests

```bash
make test
```

---

## 📊 Modèles implémentés

| Modèle | Type | Avantage principal |
|--------|------|--------------------|
| Régression Logistique | Linéaire | Interprétable, rapide, bonne baseline |
| Random Forest | Ensemble (bagging) | Robuste, gère les non-linéarités |
| Gradient Boosting | Ensemble (boosting) | Souvent le plus performant sur données tabulaires |
| SVM (RBF) | Noyau | Efficace sur datasets de taille moyenne |
| KNN | Instance-based | Simple, utile comme référence non-paramétrique |

> Le meilleur modèle est sélectionné selon le **F1-score**, métrique la plus adaptée aux problèmes de classification déséquilibrée.

---

## 🖥️ Interface Web

Une fois l'API démarrée, l'interface permet de :
- Visualiser les performances du meilleur modèle
- Saisir les caractéristiques d'un client et obtenir une prédiction en temps réel
- Comparer tous les modèles dans un tableau récapitulatif

---

## 🛠️ Technologies

| Catégorie | Outils |
|-----------|--------|
| Langage | Python 3.10+ |
| ML | Scikit-learn, Imbalanced-learn (SMOTE) |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| API | Flask |
| Persistance | Joblib, JSON |
| Tests | Pytest |
| Automatisation | Makefile |

---

## 📈 Résultats

Après exécution de la pipeline, les métriques complètes sont disponibles dans `metrics/results.json`. Les graphiques de comparaison et matrices de confusion sont sauvegardés dans `metrics/`.

---

## 💡 Améliorations possibles

- [ ] Ajouter une recherche d'hyperparamètres (GridSearchCV)
- [ ] Intégrer MLflow pour un vrai versionnage des expériences
- [ ] Déployer sur Render ou Railway (gratuit)
- [ ] Ajouter SHAP pour l'explicabilité des prédictions
- [ ] Créer un notebook Jupyter d'analyse exploratoire

---

## 👨‍💻 Auteur

**Papa Malick NDIAYE**
Master Data Science & Génie Logiciel — Université Alioune Diop de Bambey (UADB)
📧 njaymika@gmail.com
🔗 [LinkedIn](www.linkedin.com/in/papa-malick-ndiaye-b58b22309)
🐙 [GitHub](https://github.com/pa-malick)

---

<div align="center">
<sub>Projet réalisé dans le cadre du Master DSGL – UADB © 2026</sub>
</div>

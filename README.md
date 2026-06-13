# Diabetes Prediction Dashboard

Dashboard interactif de prédiction du diabète, construit avec Streamlit. Projet 1 d'une roadmap d'apprentissage en IA médicale.

## Aperçu

Ce projet couvre l'ensemble du pipeline ML sur un problème de classification médicale : exploration des données, imputation des valeurs manquantes, entraînement de plusieurs modèles, évaluation avec des métriques cliniques, explicabilité via SHAP, et interface de prédiction personnalisée.

**Dataset** : [PIMA Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) — 768 patientes, 8 variables cliniques, classification binaire (diabétique / non diabétique).

## Fonctionnalités

- **EDA interactive** — distributions, boxplots, répartition des classes, matrice de corrélation
- **Comparaison de deux stratégies d'imputation** — Mean imputation vs MICE (Iterative Imputer)
- **4 modèles comparables** — KNN, SVM, Random Forest, XGBoost
- **Métriques cliniques** — AUC-ROC, précision, rappel, F1, matrice de confusion, courbe ROC
- **Explicabilité SHAP** — feature importance pour chaque modèle et chaque stratégie d'imputation
- **Prédiction personnalisée** — saisie manuelle des données patient, probabilité de risque en temps réel

## Résultats

Meilleurs AUC-ROC obtenus sur le jeu de test (20%, stratifié) :

| Modèle | Mean imputation | MICE imputation |
| ------ | :-------------: | :-------------: |
| KNN    |      0.792      |      0.786      |
| SVM    |      0.813      |      0.803      |
| RF     |      0.814      |    **0.818**    |
| XGB    |    **0.818**    |      0.816      |

Le Random Forest avec MICE et le XGBoost avec MEAN atteignent la meilleure AUC (0.818). Le SVM offre le meilleur rappel (0.76) sur les cas diabétiques, ce qui est souvent prioritaire en contexte médical.

## Stack technique

- **ML** : scikit-learn, XGBoost
- **Explicabilité** : SHAP
- **Interface** : Streamlit
- **Data** : pandas, NumPy

## Structure du projet

```
├── Notebook.ipynb      # EDA, preprocessing, entraînement
├── backend.py          # Chargement des données, génération des graphiques SHAP
├── dashboard.py        # Interface Streamlit
├── data/
│   ├── mean_train      # Features train (mean imputation, standardisées)
│   ├── mean_test       # Features test  (mean imputation, standardisées)
│   ├── mice_train      # Features train (MICE imputation, standardisées)
│   └── mice_test       # Features test  (MICE imputation, standardisées)
└── models/
    ├── results_Mean    # Modèles entraînés + métriques (mean)
    ├── results_Mice    # Modèles entraînés + métriques (MICE)
    ├── scaler_mean.pkl # StandardScaler fitted sur train (mean)
    └── scaler_mice.pkl # StandardScaler fitted sur train (MICE)
```

## Installation

```bash
git clone https://github.com/CocoBZH29/diabete-dashboard.git
cd diabete-dashboard

python -m venv .venv
source .venv/bin/activate  # Windows : .venv\Scripts\activate

pip install streamlit scikit-learn xgboost shap matplotlib pandas numpy joblib kagglehub
```

## Utilisation

**1. Générer les données et entraîner les modèles**

Exécuter toutes les cellules de `Notebook.ipynb`. Cela génère les fichiers dans `data/` et `models/`.

**2. Lancer le dashboard**

```bash
streamlit run dashboard.py
```

## Points clés d'implémentation

- **Pas de data leakage** : le `StandardScaler` est fitté uniquement sur le train set, puis appliqué en transform sur le test set et les prédictions utilisateur.
- **Gestion du déséquilibre de classes** : `class_weight='balanced'` pour SVM, poids calculés manuellement pour RF, `scale_pos_weight` pour XGBoost.
- **SHAP adapté par modèle** : `TreeExplainer` pour RF et XGB (rapide), `KernelExplainer` pour KNN et SVM (échantillonné sur 50 points).

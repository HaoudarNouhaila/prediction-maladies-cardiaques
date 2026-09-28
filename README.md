# Prédiction des maladies cardiaques

Projet de Machine Learning : prédire la présence d'une maladie cardiaque à partir de
mesures cliniques, en comparant cinq algorithmes de classification.

L'objectif métier oriente tout le projet : **mieux vaut détecter un maximum de malades**
(recall élevé sur la classe 1) que maximiser l'accuracy globale, car un faux négatif — un patient
malade déclaré sain — est bien plus grave qu'un faux positif.

## Données

`data/heart.csv` — jeu de données *Heart Disease* (UCI / Cleveland), 14 variables cliniques.

- 1 025 lignes brutes → **302 après suppression des doublons**
- Aucune valeur manquante
- Cible `target` : `1` = malade (54,3 %), `0` = non malade (45,7 %) → classes quasi équilibrées

| Variable | Description |
|---|---|
| `age`, `sex` | âge, sexe |
| `cp` | type de douleur thoracique |
| `trestbps` | tension artérielle au repos |
| `chol` | cholestérol sérique |
| `fbs` | glycémie à jeun > 120 mg/dl |
| `restecg` | résultat de l'ECG au repos |
| `thalach` | fréquence cardiaque maximale atteinte |
| `exang` | angine induite par l'effort |
| `oldpeak`, `slope` | dépression ST et pente du segment ST |
| `ca` | nombre de vaisseaux principaux colorés |
| `thal` | thalassémie |
| `target` | **cible** — présence d'une maladie cardiaque |

## Démarche

1. **Exploration** — `info`, `describe`, contrôle des valeurs manquantes et des doublons.
2. **Nettoyage** — suppression des 723 lignes dupliquées.
3. **Analyse visuelle** — distributions des variables numériques, boxplots par classe,
   matrice de corrélation. `thalach` (fréquence cardiaque max) ressort comme très discriminante :
   les patients malades ont une fréquence maximale nettement plus basse.
4. **Préparation** — split train/test 67/33 (`random_state=42`), puis `StandardScaler`
   sur les 5 variables numériques (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`).
5. **Modélisation** — entraînement et comparaison de 5 classifieurs sur accuracy,
   rapport de classification et matrice de confusion.

## Résultats

| Modèle | Train acc. | Test acc. | Recall classe 1 | Précision classe 1 |
|---|---|---|---|---|
| Logistic Regression | 86,63 % | **83,00 %** | 0,86 | 0,81 |
| **SVM** (RBF) | 88,61 % | 81,00 % | **0,90** | 0,76 |
| Random Forest (1000 arbres) | 100,00 % | 81,00 % | 0,88 | 0,77 |
| Decision Tree | 100,00 % | 73,00 % | 0,69 | 0,74 |
| KNN | 85,64 % | 72,00 % | 0,80 | 0,68 |

**Modèle retenu : SVM.** La régression logistique a la meilleure accuracy globale, mais le SVM
détecte **90 % des patients malades** — le critère qui compte ici — au prix de quelques faux positifs
supplémentaires. Random Forest et Decision Tree atteignent 100 % en entraînement pour 81 % et 73 %
en test : **surapprentissage** net, généralisation peu fiable.

## Pistes d'amélioration

- Optimisation des hyperparamètres du SVM (`kernel`, `C`, `gamma`) par `GridSearchCV`
- Validation croisée plutôt qu'un unique split — l'échantillon de test ne fait que 100 lignes,
  les écarts entre modèles restent dans la marge d'erreur
- Sélection de variables et méthodes d'ensemble pour renforcer la détection de la classe 1

## Installation

```bash
pip install -r requirements.txt
jupyter notebook prediction_des_maladies_cardiaques.ipynb
```

# Classificateur Arbre de Décision pour les Données de Criminalité

Ce projet implémente un classificateur Arbre de Décision pour analyser un jeu de données contenant des informations sur les crimes. Le modèle utilise le type de crime, la région et la gravité pour prédire le moment de la journée où les crimes sont les plus susceptibles de se produire.

## Structure du Projet

- `Crime_clasification_dataset.csv` : Le jeu de données original de criminalité avec des champs tels que `Type_de_crime`, `Région`, `Gravité`, et `Période_de_la_journée`.
- `decision_tree.py` : Script Python qui encode les données catégorielles, entraîne un modèle d’arbre de décision et visualise l'arbre de décision.

## Caractéristiques

Les caractéristiques suivantes sont utilisées pour entraîner le classificateur d’arbre de décision :
- **Type_de_crime** : Le type de crime (par exemple, "Vol", "Fraude").
- **Région** : La région où le crime a eu lieu (par exemple, "Île-de-France", "PACA").
- **Gravité** : La gravité du crime (par exemple, "Élevée", "Faible").

La variable cible est :
- **Période_de_la_journée** : Le moment de la journée (par exemple, "Matin", "Après-midi", "Soir", "Nuit").

## Prise en Main

### Prérequis

Assurez-vous d'avoir Python installé avec les bibliothèques suivantes :
- `pandas`
- `scikit-learn`
- `matplotlib`

Vous pouvez installer les packages requis avec :
```bash
pip install pandas scikit-learn matplotlib
```

### Utilisation

1. Clonez le dépôt sur votre machine locale.
2. Ajoutez votre jeu de données de criminalité sous le nom `Crime_clasification_dataset.csv` ou modifiez le script pour utiliser vos propres données.
3. Exécutez le script :
   ```bash
   python decision_tree.py
   ```

### Vue d’Ensemble du Code

Le script effectue les étapes suivantes :
1. **Encodage des Données** : Les caractéristiques catégorielles sont encodées en valeurs numériques pour être compatibles avec le classificateur d'arbre de décision de scikit-learn.
2. **Entraînement du Modèle** : Un modèle de Classificateur Arbre de Décision est entraîné sur les caractéristiques.
3. **Évaluation du Modèle** : La précision du modèle est calculée et affichée.
4. **Visualisation de l'Arbre de Décision** : L’arbre de décision est visualisé, montrant les séparations et l’importance des caractéristiques.

### Exemple de Code

Voici un extrait du code principal utilisé dans `decision_tree.py` :

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# Chargement du jeu de données
df = pd.read_csv('Crime_clasification_dataset.csv')

# Encodage des caractéristiques catégorielles
df['Type_de_crime'] = df['Type_de_crime'].astype('category').cat.codes
df['Région'] = df['Région'].astype('category').cat.codes
df['Gravité'] = df['Gravité'].astype('category').cat.codes
df['Période_de_la_journée'] = df['Période_de_la_journée'].astype('category').cat.codes

# Définir les caractéristiques et la cible
X = df[['Type_de_crime', 'Région', 'Gravité']]
y = df['Période_de_la_journée']

# Diviser les données en ensembles d’entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialiser et entraîner le modèle
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Évaluer la précision du modèle
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy:.2f}")

# Visualiser l’arbre de décision
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=['Type_de_crime', 'Région', 'Gravité'], 
          class_names=['Matin', 'Après-midi', 'Soir', 'Nuit'], filled=True)
plt.title(f"Classificateur Arbre de Décision
Précision : {accuracy:.2f}")
plt.show()
```

## Résultats

L’arbre de décision fournit des indications sur la façon dont le type de crime, la région et la gravité influencent le moment de la journée où le crime est le plus susceptible de se produire. La précision du modèle est affichée lors de l'exécution, ainsi qu'une visualisation de l'arbre.
![myplot]('TP1/myplot.png')

## Licence

Ce projet est sous licence MIT.

## Remerciements

- Source du jeu de données (si applicable)
- Bibliothèques utilisées : [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/), [matplotlib](https://matplotlib.org/)

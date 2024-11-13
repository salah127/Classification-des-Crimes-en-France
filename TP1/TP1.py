import pandas as test
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df_expanded = test.read_csv('Crime_classification_dataset.csv')

# Encoding categorical features
df_encoded = df_expanded.copy()
df_encoded['Type_de_crime'] = df_encoded['Type_de_crime'].astype('category').cat.codes
df_encoded['Région'] = df_encoded['Région'].astype('category').cat.codes
df_encoded['Gravité'] = df_encoded['Gravité'].astype('category').cat.codes
df_encoded['Période_de_la_journée'] = df_encoded['Période_de_la_journée'].astype('category').cat.codes

# Define features (X) and target (y)
X = df_encoded[['Type_de_crime', 'Région', 'Gravité']]
y = df_encoded['Période_de_la_journée']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict on test set and calculate accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=['Type_de_crime', 'Région', 'Gravité'], 
          class_names=['Matin', 'Après-midi', 'Soir', 'Nuit'], filled=True)
plt.title(f"Decision Tree Classifier\nAccuracy: {accuracy:.2f}")
plt.show()

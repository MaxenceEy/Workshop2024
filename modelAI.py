import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# 1. Charger le dataset nettoyé
dataset_path = 'datasets/datasets_cleaned/dataset_cleaned.csv'
df = pd.read_csv(dataset_path)

# 2. Prétraitement des données : Encodage de la colonne 'type'
df['label'] = df['type'].map({'HS': 1, 'CN': 0})  # 1 pour hate speech, 0 pour non hate speech

# 3. Vectorisation des textes
vectorizer = CountVectorizer(tokenizer=lambda x: eval(x))  # Utiliser la colonne filtered_tokens
X = vectorizer.fit_transform(df['filtered_tokens'])
y = df['label']

# 4. Diviser le dataset en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Choisir et entraîner le modèle
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. Faire des prédictions
y_pred = model.predict(X_test)

# 7. Évaluer le modèle
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred, target_names=['Non Hate Speech', 'Hate Speech']))

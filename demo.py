import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer  # Utilisation de TF-IDF pour SVM
from sklearn.svm import SVC  # Importer le modèle SVM
from sklearn.metrics import classification_report, accuracy_score

# Exemple de liste de stop words personnalisée
# custom_stop_words = {'the', 'and', 'is', 'in', 'to', 'with', 'on', 'Im', 'a'}  # Ajoutez les mots que vous souhaitez exclure

# Fonction pour nettoyer un texte
def clean_text(text):
    # Convertir en minuscules
    text = text.lower()
    
    # Supprimer la ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Supprimer les chiffres
    text = re.sub(r'\d+', '', text)
    
    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
# Fonction pour tokeniser un texte nettoyé
def tokenize_text(cleaned_text):
    # Utiliser une regex pour trouver tous les mots
    return re.findall(r'\b\w+\b', cleaned_text)

# Fonction pour supprimer les stop words
#def remove_stop_words(tokens):
#    return [word for word in tokens if word not in custom_stop_words]

# Charger le dataset CSV
dataset_path = 'datasets/datasets_sources/DIAlOCONAN.csv' 
df = pd.read_csv(dataset_path)
# Nettoyer chaque texte
df['cleaned_text'] = df['text'].apply(clean_text)
# Tokeniser les textes nettoyés
df['tokens'] = df['cleaned_text'].apply(tokenize_text)

# Supprimer les stop words pour chaque texte tokenisé
# df['filtered_tokens'] = df['tokens'].apply(remove_stop_words)

# Convertir les tokens filtrés en chaînes de caractères
df['text'] = df['tokens'].apply(lambda x: ' '.join(x))

# Supprimer les colonnes dialogue_id, turn_id et source si elles existent
columns_to_drop = ['dialogue_id', 'turn_id', 'source']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# Sauvegarder le dataset avec les colonnes nettoyées et tokenisées dans un nouveau fichier CSV
df.to_csv('datasets/datasets_cleaned/dataset_cleaned.csv', index=False)

# Prétraitement des données : Encodage de la colonne 'type'
df['label'] = df['type'].map({'HS': 1, 'CN': 0})  # 1 pour hate speech, 0 pour non hate speech

# Vectorisation des textes avec TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Diviser le dataset en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choisir et entraîner le modèle SVM
model = SVC(kernel='linear')  # Le kernel linéaire est souvent utilisé pour les tâches de classification de texte
model.fit(X_train, y_train)

# Faire des prédictions
y_pred = model.predict(X_test)
# Évaluer le modèle
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred, target_names=['Non Hate Speech', 'Hate Speech']))
# Fonction pour tester une nouvelle phrase
def test_phrase(phrase):
    cleaned_phrase = clean_text(phrase)  # Nettoyage de la phrase
    tokens = tokenize_text(cleaned_phrase)  # Tokenisation
    #filtered_tokens = remove_stop_words(tokens)  # Suppression des stop words
    txt = ' '.join(tokens)  # Conversion en chaîne de caractères

    # Vectoriser la phrase
    vectorized_phrase = vectorizer.transform([txt])  # Transforme la phrase en vecteur

    # Prédiction
    prediction = model.predict(vectorized_phrase)
    return 'Hate Speech' if prediction[0] == 1 else 'Non Hate Speech'
# Exemple d'utilisation de la fonction test_phrase
user_input = input("Entrez une phrase à tester :")  # Interaction avec l'utilisateur
result = test_phrase(user_input)
print(f"Résultat : {result}")
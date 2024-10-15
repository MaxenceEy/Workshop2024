import pandas as pd
import re
import string

# Exemple de liste de stop words personnalisée
custom_stop_words = {'the', 'and', 'is', 'in', 'to', 'with'}  # Ajoutez les mots que vous souhaitez exclure

# Fonction pour nettoyer un texte
def clean_text(text):
    # Convertir en minuscules
    text = text.lower()
    
    # Supprimer la ponctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Supprimer les chiffres (s'il y a lieu)
    text = re.sub(r'\d+', '', text)
    
    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Fonction pour tokeniser un texte nettoyé avec des expressions régulières
def tokenize_text(cleaned_text):
    # Utiliser une regex pour trouver tous les mots
    return re.findall(r'\b\w+\b', cleaned_text)  # Renvoie une liste de mots

# Fonction pour supprimer les stop words
def remove_stop_words(tokens):
    return [word for word in tokens if word not in custom_stop_words]

# 1. Charger le dataset CSV
dataset_path = 'datasets/datasets_sources/DIAlOCONAN.csv'  # Assurez-vous que le chemin est correct
df = pd.read_csv(dataset_path)

# 2. Nettoyer chaque texte
df['cleaned_text'] = df['text'].apply(clean_text)

# 3. Tokeniser les textes nettoyés
df['tokens'] = df['cleaned_text'].apply(tokenize_text)

# 4. Supprimer les stop words pour chaque texte tokenisé
df['filtered_tokens'] = df['tokens'].apply(remove_stop_words)

# Supprimer les colonnes dialogue_id, turn_id et source
df = df.drop(columns=['dialogue_id', 'turn_id', 'source'])

# Afficher les premiers résultats pour vérifier
print(df[['text', 'cleaned_text', 'tokens', 'filtered_tokens']].head(10))  # Affiche les 10 premiers résultats

# Sauvegarder le dataset avec les colonnes nettoyées et tokenisées dans un nouveau fichier CSV
df.to_csv('datasets/datasets_cleaned/dataset_cleaned.csv', index=False)

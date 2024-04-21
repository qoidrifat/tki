import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import re
import sys
import json
import pickle
import math
import sqlite3

# Load CSV data from local directory
csv_file = r'C:\xampp\htdocs\TKI\UTS\fix\indexing\MPL Indonesia Season 13 - BoxMatch.csv'
df = pd.read_csv(csv_file)

# Load stopwords from local directory
stopword_file = r'C:\xampp\htdocs\TKI\UTS\fix\indexing\stopword.txt'
with open(stopword_file, 'r') as f:
    stop_words = f.read().splitlines()  # Convert to list

# Remove trailing spaces from column names
df.columns = df.columns.str.strip()

# Check if 'team' or 'role' is in the column names (case-insensitive)
candidate_columns = [col for col in df.columns if 'team' in col.lower() or 'role' in col.lower()]
text_data = None
for col in candidate_columns:
    if 'team' in col.lower():  # Prefer 'team' over 'role'
        text_data = df[col]
        break

if text_data is None:
    print("No column containing team names found in the CSV data.")
    sys.exit(1)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)
tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)

# Get feature names
feature_names = tfidf_vectorizer.get_feature_names_out()

# Convert TF-IDF matrix to DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# Save TF-IDF DataFrame to a file without extension
tfidf_df.to_pickle('indexdb')
print("TF-IDF data processed successfully and saved to 'indexdb'")
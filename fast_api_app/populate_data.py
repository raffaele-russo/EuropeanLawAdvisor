"""This script creates an index and populates with legal data"""
import os
import zipfile
import joblib
import gdown
import pandas as pd
from search import Search

# Download and extract dataset
FILE_ID = '1BZgvyxU5opZfBBzfKJfIW6eOWNZv5AES'
URL = f'https://drive.google.com/uc?id={FILE_ID}'
FILENAME = 'dataset'
ZIP_FILENAME = FILENAME + ".zip"
gdown.download(URL, ZIP_FILENAME, quiet=False)

with zipfile.ZipFile(ZIP_FILENAME,"r") as zip_ref:
    zip_ref.extractall()

os.remove(ZIP_FILENAME)

# Setup Elastic Search
es = Search()
es.create_index()

# Load data
df = pd.read_csv(FILENAME + ".csv")
documents = df.to_dict(orient="records")

# Fit the TF-IDF vectorizer on the text data
print("Fitting TF-IDF on the dataset...")
documents_text = df['Text'].tolist()
es.fit_tfidf(documents_text)

# Save the fitted TF-IDF model
joblib.dump(es.vectorizer, 'tfidf_vectorizer.pkl')

print("Inserting documents...")
es.insert_documents(documents = documents, batch_size=500)
print("Dataset succesfully loaded.")

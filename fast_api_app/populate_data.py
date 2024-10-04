"""This script creates an index and populates with legal data"""
import os
import zipfile
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

# Insert data into Elastic Search
df = pd.read_csv(FILENAME + ".csv")
es = Search()
es.create_index()
documents = df.to_dict(orient="records")
es.insert_documents(documents = documents, batch_size=500)

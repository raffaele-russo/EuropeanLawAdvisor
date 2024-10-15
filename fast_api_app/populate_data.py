"""This script creates an index and populates with legal data"""
import os
import zipfile
import joblib
import gdown
import pandas as pd
from search import Search
from config import Config
import logging
from app_logging import setup_logging

# Setup logging 
logger = logging.getLogger(__name__)
setup_logging(logger)

# Paths and URLs
URL_DATASET = Config.URL_DATASET
FILENAME = Config.DATASET_FILENAME
ZIP_FILENAME = FILENAME + ".zip"
CSV_FILENAME = FILENAME + ".csv"
VECTORIZER_PATH = Config.VECTORIZER_MODEL_PATH

# Download and extract dataset only if the CSV file doesn't exist
if not os.path.exists(CSV_FILENAME):
    logger.info(f"{CSV_FILENAME} not found. Downloading and extracting dataset...")
    
    # Download dataset
    gdown.download(URL_DATASET, ZIP_FILENAME, quiet=False)

    # Extract dataset
    with zipfile.ZipFile(ZIP_FILENAME, "r") as zip_ref:
        zip_ref.extractall()

    # Remove the zip file after extraction
    os.remove(ZIP_FILENAME)
else:
    logger.info(f"{CSV_FILENAME} already exists, skipping download and extraction.")

# Setup Elasticsearch
es = Search()

# Check if the vectorizer model exists before proceeding
if not os.path.exists(VECTORIZER_PATH):
    logger.info(f"Fitting TF-IDF vectorizer as {VECTORIZER_PATH} does not exist...")

    # Load dataset for TF-IDF fitting
    df = pd.read_csv(CSV_FILENAME)
    documents_text = df[Config.QUERY_TEXT_FIELD].tolist()

    # Fit the TF-IDF vectorizer
    es.fit_tfidf(documents_text)

    # Save the fitted TF-IDF model
    joblib.dump(es.vectorizer, VECTORIZER_PATH)
    logger.info(f"TF-IDF model saved to {VECTORIZER_PATH}")
else:
    logger.info(f"{VECTORIZER_PATH} already exists, skipping TF-IDF fitting.")
    es.vectorizer = joblib.load(VECTORIZER_PATH)

logger.info(f"Creating Elasticsearch index...")
es.create_index()

# Load dataset and insert documents
logger.info(f"Inserting documents into Elasticsearch index...")
df = pd.read_csv(CSV_FILENAME)
documents = df.to_dict(orient="records")
es.insert_documents(documents=documents)
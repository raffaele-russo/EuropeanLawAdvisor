"""This script creates an index and populates with legal data"""
import logging
import os
import zipfile
import joblib
import gdown
import pandas as pd
from search import Search
from config import Config
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
    logger.info("%s not found. Downloading and extracting dataset...",CSV_FILENAME)

    # Download dataset
    gdown.download(URL_DATASET, ZIP_FILENAME, quiet=False)

    # Extract dataset
    with zipfile.ZipFile(ZIP_FILENAME, "r") as zip_ref:
        zip_ref.extractall()

    # Remove the zip file after extraction
    os.remove(ZIP_FILENAME)
else:
    logger.info("%s already exists, skipping download and extraction.",CSV_FILENAME)

# Setup Elasticsearch
es = Search()

# Check if the vectorizer model exists before proceeding
model_path = os.path.join("fast_api_app",VECTORIZER_PATH)
if not os.path.exists(model_path):
    logger.info("Fitting TF-IDF vectorizer as %s does not exist...",model_path)

    # Load dataset for TF-IDF fitting
    df = pd.read_csv(CSV_FILENAME)
    documents_text = df[Config.QUERY_TEXT_FIELD].tolist()

    # Fit the TF-IDF vectorizer
    es.fit_tfidf(documents_text)

    # Save the fitted TF-IDF model
    joblib.dump(es.vectorizer, model_path)
    logger.info("TF-IDF model saved to %s",model_path)
else:
    logger.info("%s already exists, skipping TF-IDF fitting.",model_path)
    es.vectorizer = joblib.load(model_path)

logger.info("Creating Elasticsearch index...")
es.create_index()

# Load dataset and insert documents
logger.info("Inserting documents into Elasticsearch index...")
df = pd.read_csv(CSV_FILENAME)
documents = df.to_dict(orient="records")
es.insert_documents(documents=documents)

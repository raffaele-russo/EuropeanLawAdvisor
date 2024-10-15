"""Module handling all the app configuration"""
import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

class Config:
    """Application Configuration"""
    # ElasticSearch Configuration
    ELASTICSEARCH_USERNAME = os.getenv("ELASTICSEARCH_USERNAME", "elastic")
    ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD")
    ELASTICSEARCH_HOST = os.getenv("ELASTIC_SEARCH_URL", "localhost")
    ELASTICSEARCH_PORT = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
    
    # Index Names
    INDEX_NAME = os.getenv("INDEX_NAME", "law_docs")
    
    # Embedding Model
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Query Parameters
    ELASTIC_QUERY_SIZE = int(os.getenv("ELASTIC_QUERY_SIZE", "10"))
    KNN_CANDIDATES = int(os.getenv("KNN_CANDIDATES", "50"))
    KNN_SEARCH_K = int(os.getenv("KNN_SEARCH_K", "20"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "500"))
    QUERY_TEXT_FIELD = os.getenv("QUERY_TEXT_FIELD", "Text")


    # Rag parameters
    RAG_PROMPT_PATH = os.environ.get("RAG_PROMPT", "rlm/rag-prompt")
    LLM_MODEL = os.environ.get("LLM_MODEL", "phi3")

    # Vectorizer
    VECTORIZER_MODEL_PATH = os.environ.get("VECTORIZER_MODEL_PATH", "tfidf_vectorizer.pkl")

    # Dataset info
    URL_DATASET = os.environ.get("URL_DATASET", "https://drive.google.com/uc?id=1BZgvyxU5opZfBBzfKJfIW6eOWNZv5AES")
    DATASET_FILENAME = os.environ.get("DATASET_FILENAME","dataset")
    # Log Level
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

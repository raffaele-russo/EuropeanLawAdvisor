"""Module handling all the app configuration"""
import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

class Config:
    """Application Configuration"""
    # ElasticSearch Configuration
    # pylint: disable=too-few-public-methods
    ELASTICSEARCH_USERNAME = os.getenv("ELASTICSEARCH_USERNAME", "elastic")
    ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD")
    ELASTICSEARCH_HOST = os.getenv("ELASTIC_SEARCH_URL", "localhost")
    ELASTICSEARCH_PORT = int(os.getenv("ELASTICSEARCH_PORT", "9200"))

    # Index info
    INDEX_NAME = os.getenv("INDEX_NAME", "law_docs")
    RAG_INDEX_NAME = os.getenv("RAG_INDEX_NAME","law_chunks")
    INDEX_NAME_MAPPING = {
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                    },
                    "sparse_embedding": {  
                        "type": "sparse_vector"
                    },
                }
            }

    # Embedding Model
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Query Parameters
    ELASTIC_QUERY_SIZE = int(os.getenv("ELASTIC_QUERY_SIZE", "10"))
    KNN_CANDIDATES = int(os.getenv("KNN_CANDIDATES", "50"))
    KNN_SEARCH_K = int(os.getenv("KNN_SEARCH_K", "20"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
    QUERY_TEXT_FIELD = os.getenv("QUERY_TEXT_FIELD", "Text")

    # Rag parameters
    RAG_PROMPT_PATH = os.getenv("RAG_PROMPT", "rlm/rag-prompt")
    RAG_CUSTOM_PROMPT_PATH = os.getenv("RAG_CUSTOM_PROMPT_PATH")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME","gpt-3.5-turbo-instruct")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE","512"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP","16"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS","4096"))

    # Vectorizer
    VECTORIZER_MODEL_PATH = os.getenv("VECTORIZER_MODEL_PATH", "tfidf_vectorizer.pkl")
    VECTORIZER_MAX_FEATS = int(os.getenv("VECTORIZER_MAX_FEATS","5000"))

    # Dataset info
    URL_DATASET = os.getenv("URL_DATASET",
    "https://drive.google.com/uc?id=1BZgvyxU5opZfBBzfKJfIW6eOWNZv5AES")
    DATASET_FILENAME = os.getenv("DATASET_FILENAME","dataset")

    # Log Level
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    # Retriever info
    K_DOCS_TO_RETRIEVE = int(os.getenv("K_DOCS_TO_RETRIEVE","10"))
    K_CHUNKS_TO_RETRIEVE = int(os.getenv("K_CHUNKS_TO_RETRIEVE","3"))

    # Define constants for query types
    QUERY_TYPES = {
        'RAG': 'rag',
        'MULTI_MATCH': 'multi_match',
        'KNN': 'knn',
        'HYBRID': 'hybrid',
        'TF_IDF': 'tf-idf'
    }

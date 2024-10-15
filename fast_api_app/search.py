"""Module responsible for the Elastic Search client that will perform the full search"""
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_elasticsearch import ElasticsearchRetriever
from models.text_embedding import TextEmbedding 
from config import Config
import logging
from app_logging import setup_logging
from tqdm import tqdm

# Setup logger
logger = logging.getLogger(__name__)
setup_logging(logger)

class Search:
    """Elastic Search Client""" 
    def __init__(self):
        """Init client connection to Elastic Search"""
        try:
            self.es = Elasticsearch(
                    [{"scheme": "http", "host": Config.ELASTICSEARCH_HOST,
                    "port": Config.ELASTICSEARCH_PORT}],
                    basic_auth=(Config.ELASTICSEARCH_USERNAME, Config.ELASTICSEARCH_PASSWORD),
                    timeout=60
                    )
            logger.info("Connected to Elasticsearch!")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise

        self.embedding = TextEmbedding(Config.EMBEDDING_MODEL)
        self.vectorizer = None

        self.retriever = ElasticsearchRetriever(
                            es_client = self.es,
                            index_name = Config.INDEX_NAME,
                            body_func = self.vector_query,
                            content_field=Config.QUERY_TEXT_FIELD,
                        )

    def create_index(self):
        """Create a new index after previously deleting an index with the same name"""
        logger.info("Creating index")
        self.es.indices.delete(index=Config.INDEX_NAME,ignore_unavailable = True)
        self.es.indices.create(
            index=Config.INDEX_NAME,
            mappings={
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                    },
                    "sparse_embedding": {  
                        "type": "sparse_vector"
                    },
                }
            })
   
    def insert_document(self, document):
        """Insert a document in the specified index"""
        return self.es.index(index=Config.INDEX_NAME, document={
            **document,
            "embedding" : self.get_embedding(document[Config.QUERY_TEXT_FIELD]),
            "sparse_embedding": self.get_tfidf_scores(document[Config.QUERY_TEXT_FIELD])
        })

    def insert_documents(self, documents):
        """Insert a list of documents in the specified index in batches."""
        operations = []
        for i, document in tqdm(enumerate(documents), total=len(documents), desc="Inserting Documents"):
            operations.append({"index": {"_index": Config.INDEX_NAME}})
            operations.append({
                **document,
                "embedding" : self.get_embedding(document[Config.QUERY_TEXT_FIELD]),
                "sparse_embedding": self.get_tfidf_scores(document[Config.QUERY_TEXT_FIELD])
                })

            # Once we reach the batch size, send the bulk request
            if (i + 1) % Config.BATCH_SIZE == 0:
                response = self.es.bulk(operations=operations)
                operations = []

        # Insert any remaining documents
        if operations:
            response = self.es.bulk(operations=operations)

    def search(self, **query_args):
        """Search query in the specified index"""
        return self.es.search(index=Config.INDEX_NAME, **query_args)

    def multi_match_search(self, query, query_fields,
    size : int = Config.ELASTIC_QUERY_SIZE, from_ : int = 0):
        """Perform multi match query in the specified index"""
        return self.es.search(
                            index = Config.INDEX_NAME,
                            query={
                                "multi_match": {
                                    "query" : query,
                                    "fields" : query_fields,   
                                }
                            }, size=size, from_= from_
            )

    def knn_search(self, query, size : int = Config.ELASTIC_QUERY_SIZE,
                   from_ : int = 0):
        """Perform knn query in the specified index"""
        return self.es.search(
                    index= Config.INDEX_NAME,
                    knn={
                        "field": "embedding",
                        "query_vector": self.get_embedding(query),
                        "num_candidates": Config.KNN_CANDIDATES,
                        "k": Config.KNN_SEARCH_K,
                    },
                    size=size,
                    from_=from_
                )

    def hybrid_search(self, query, query_fields,
    size : int = Config.ELASTIC_QUERY_SIZE, from_ : int = 0):
        """Perform multi match query in the specified index"""
        return self.es.search(
                            index = Config.INDEX_NAME,
                            query={
                                "multi_match": {
                                    "query" : query,
                                    "fields" : query_fields,   
                                }
                            },
                            knn={
                                "field": "embedding",
                                "query_vector": self.get_embedding(query),
                                "num_candidates": Config.KNN_CANDIDATES,
                                "k": Config.KNN_SEARCH_K,
                            },
                            rank={
                                "rrf" : {
                                    "rank_window_size" : size + from_
                                }
                            },
                            size=size,
                            from_=from_
            )

    def semantic_search(self, query, size : int = Config.ELASTIC_QUERY_SIZE,
                   from_ : int = 0):
        """Perform semantic search query in the specified index"""
        tfidf_score = self.get_tfidf_scores(query)
        return self.es.search(
                    index= Config.INDEX_NAME,
                    query={
                            "sparse_vector": {
                                "field" : "sparse_embedding",
                                "query_vector": tfidf_score
                            }
                        },
                    size=size,
                    from_=from_
                )

    def retrieve_document(self, id_document : int):
        """Retrieve the document given the id"""
        return self.es.get(index=Config.INDEX_NAME, id=id_document)

    def get_index_fields(self) -> list[str]:
        """Retrieve the index fields as a list"""
        mapping = self.es.indices.get_mapping(index=Config.INDEX_NAME)
        return list(mapping[Config.INDEX_NAME]["mappings"]["properties"])

    def get_text_index_fields(self) -> list[str]:
        """Retrieve the text index fields as a list"""
        index_name = Config.INDEX_NAME
        mapping = self.es.indices.get_mapping(index=index_name)
        properties = mapping[index_name]["mappings"]["properties"]
        searchable_fields_types = ["text"]
        searchable_fields = []

        for field, field_info in properties.items():
            field_type = field_info.get("type")
            if field_type in searchable_fields_types:
                searchable_fields.append(field)

        return searchable_fields

    def get_embedding(self, text : str):
        """Get model embeddings"""
        return self.embedding.model.encode(text, show_progress_bar = False)

    def fit_tfidf(self, documents):
        """Fit the TF-IDF Vectorizer on the given documents."""
        self.vectorizer = TfidfVectorizer(max_features=5000,
                                          stop_words="english",
                                          token_pattern=r"\b[a-zA-Z]+\b")
        self.vectorizer.fit(documents)

    def get_tfidf_scores(self, document):
        """Get TF-IDF scores for a single document as a dictionary of {word: score}."""
        if self.vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf first.")

        tfidf_vector = self.vectorizer.transform([document])
        feature_names = self.vectorizer.get_feature_names_out()
        sparse_array = zip(feature_names, tfidf_vector.toarray().flatten())
        doc_tfidf = {word: float(score) for word, score in sparse_array if score > 0.0}
        return doc_tfidf
    
    def vector_query(self, search_query: str):
        """Vector query for the retriever"""
        embedding_vector = self.embedding.embed_query(search_query)
        return {
            "knn": {
                "field": "embedding",
                "query_vector": embedding_vector,
                "k": Config.KNN_SEARCH_K,
                "num_candidates": Config.KNN_CANDIDATES,
            }
        }

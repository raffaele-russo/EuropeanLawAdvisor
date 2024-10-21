"""Module responsible for the Elastic Search client that will perform the full search"""
import logging
from typing import List, Dict, Any
import joblib
from elasticsearch import Elasticsearch
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_elasticsearch import ElasticsearchRetriever, ElasticsearchStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from helper.text_embedding import TextEmbedding
from helper.config import Config
from helper.app_logging import setup_logging
from helper.exception_handler import exception_handler, SearchException

# Setup logger
logger = logging.getLogger(__name__)
setup_logging(logger)

class Search:
    """Elastic Search Client""" 
    def __init__(self):
        """Init client connection to Elastic Search"""
        self.es = self.initialize_es_client()
        self.embedding = self.initialize_text_embedding()
        self.vectorizer = None
        self.retriever = self.initialize_es_retriever()
        self.vector_store = self.initialize_vector_store()
        self.text_splitter = self.initialize_text_splitter()

    @exception_handler("Failed to connect to Elasticsearch")
    def initialize_es_client(self) -> Elasticsearch:
        """Setup connection to the Elasticsearch client.

        Returns:
            Elasticsearch: An instance of the Elasticsearch client.

        Raises:
            SearchException: If the connection to Elasticsearch fails.
        """
        es = Elasticsearch(
            [{"scheme": "http", "host": Config.ELASTICSEARCH_HOST,
            "port": Config.ELASTICSEARCH_PORT}],
            basic_auth=(Config.ELASTICSEARCH_USERNAME, Config.ELASTICSEARCH_PASSWORD),
            timeout=60
        )
        logger.info("Connected to Elasticsearch!")
        return es

    @exception_handler("Failed to initialize text embedding")
    def initialize_text_embedding(self) -> TextEmbedding:
        """Setup embedding model to compute the embeddings.

        Returns:
            TextEmbedding: An instance of the embedding model.

        Raises:
            SearchException: If the initialization of the text embedding fails.
        """
        embedding = TextEmbedding(Config.EMBEDDING_MODEL)
        logger.info("Text embedding initialized successfully.")
        return embedding

    @exception_handler("Error initializing ElasticSearch client")
    def load_weights(self) -> None:
        """Load Elasticsearch client with tf-idf weights
        Raises:
            SearchException: If the initialization of the text embedding fails.
        """
        self.vectorizer = joblib.load(Config.VECTORIZER_MODEL_PATH)
        logger.info("tf-idf weights loaded")

    @exception_handler("Failed to initialize text splitter")
    def initialize_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Setup text splitter to divide documents into chunks.

        Returns:
            RecursiveCharacterTextSplitter: An instance of the text splitter.

        Raises:
            SearchException: If the initialization of the text splitter fails.
        """
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        logger.info("Text splitter initialized successfully.")
        return text_splitter

    @exception_handler("Failed to initialize Elasticsearch retriever")
    def initialize_es_retriever(self) -> ElasticsearchRetriever:
        """Setup retriever used to find related documents based on the sparse embedding.

        Returns:
            ElasticsearchRetriever: An instance of the Elasticsearch retriever.

        Raises:
            SearchException: If the initialization of the Elasticsearch retriever fails.
        """
        retriever = ElasticsearchRetriever(
            es_client=self.es,
            index_name=Config.INDEX_NAME,
            body_func=self._vector_query,
            content_field=Config.QUERY_TEXT_FIELD,
            search_kwargs={'k': Config.K_DOCS_TO_RETRIEVE}
        )
        logger.info("Elasticsearch retriever initialized successfully.")
        return retriever

    @exception_handler("Failed to initialize vector store")
    def initialize_vector_store(self) -> ElasticsearchStore:
        """Connect to the vector store containing the chunks.

        Returns:
            ElasticsearchStore: An instance of the vector store.

        Raises:
            SearchException: If the initialization of the vector store fails.
        """
        vector_store = ElasticsearchStore(
            es_user=Config.ELASTICSEARCH_USERNAME,
            es_password=Config.ELASTICSEARCH_PASSWORD,
            es_url=f"http://{Config.ELASTICSEARCH_HOST}:{Config.ELASTICSEARCH_PORT}",
            index_name=Config.RAG_INDEX_NAME,
            embedding=self.embedding
        )
        logger.info("Vector store initialized successfully.")
        return vector_store

    @exception_handler("Failed to create index")
    def create_index(self):
        """Create a new index after previously deleting an index with the same name"""
        logger.info("Creating indexes...")
        self._create_main_index()
        self._create_rag_index()

    @exception_handler("Failed to create main index")
    def _create_main_index(self) -> None:
        """Create the main index.

        Raises:
            SearchException: If the creation of the main index fails.
        """
        logger.info("Creating main index")
        self.es.indices.delete(index=Config.INDEX_NAME, ignore_unavailable=True)
        self.es.indices.create(index=Config.INDEX_NAME, mappings=Config.INDEX_NAME_MAPPING)

    @exception_handler("Failed to create RAG index")
    def _create_rag_index(self) -> None:
        """Create the RAG index.

        Raises:
            SearchException: If the creation of the RAG index fails.
        """
        logger.info("Creating RAG index")
        self.es.indices.delete(index=Config.RAG_INDEX_NAME, ignore_unavailable=True)
        self.es.indices.create(index=Config.RAG_INDEX_NAME)

    def insert_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Insert a list of documents into the specified index in batches.

        Args:
            documents (List[Dict[str, Any]]): A list of documents to be inserted.

        Raises:
            SearchException: If an issue occurs while inserting the documents.
        """
        # Prepare metadata and content for chunk processing
        metadata, content, operations = [], [], []

        tqdm_wrapper = tqdm(enumerate(documents), total=len(documents), desc="Inserting Documents")
        for i, document in tqdm_wrapper:
            try:
                self._prepare_bulk_operations(operations, document)
                # Once we reach the batch size, send the bulk request
                if (i + 1) % Config.BATCH_SIZE == 0:
                    self._execute_bulk_insert(operations)
                    self._process_and_insert_chunks(content, metadata)

                content.append(document[Config.QUERY_TEXT_FIELD])
                metadata.append(self._extract_metadata(document))
            except SearchException as search_exc: # We try to continue the insertion
                logger.error("An issue occurred while inserting the documents: %s", search_exc)
            except Exception as e:
                raise SearchException(f"An issue occurred inserting the documents: {str(e)}") from e

        # Insert any remaining documents
        if operations:
            self._execute_bulk_insert(operations)
            self._process_and_insert_chunks(content, metadata)

    @exception_handler("An error occurred during the bulk preparation")
    def _prepare_bulk_operations(self, operations: List[Dict[str, Any]],
    document: Dict[str, Any]) -> None:
        """Prepare bulk operations by adding indexing instructions and document details.

        Args:
            operations (List[Dict[str, Any]]): The list of bulk operations to prepare.
            document (Dict[str, Any]): The document to prepare for indexing.

        Raises:
            SearchException: If an error occurs during the bulk preparation.
        """
        operations.append({"index": {"_index": Config.INDEX_NAME}})
        operations.append({
            **document,
            "embedding": self._get_embedding(document[Config.QUERY_TEXT_FIELD]),
            "sparse_embedding": self._get_tfidf_scores(document[Config.QUERY_TEXT_FIELD])
        })

    def _execute_bulk_insert(self, operations: List[Dict[str, Any]]) -> None:
        """Execute bulk insert of operations into Elasticsearch.

        Args:
            operations (List[Dict[str, Any]]): The list of operations to execute.

        Raises:
            SearchException: If an error occurs during the bulk insert.
        """
        try:
            self.es.bulk(operations=operations)
            logger.debug("Bulk insert executed successfully.")
        except Exception as e:
            logger.error("Error during bulk insert: %s", e)
            raise SearchException(f"Error during bulk insert: {str(e)}") from e
        finally:
            operations.clear()

    @exception_handler("Failed to extract metadata")
    def _extract_metadata(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant metadata from the document.

        Args:
            document (Dict[str, Any]): The document from which to extract metadata.

        Returns:
            Dict[str, Any]: A dictionary containing the extracted metadata.

        Raises:
            SearchException: If an error occurs during metadata extraction.
        """
        return {
            "CELEX_ID": document.get("CELEX_ID"),
            "Directory_code": document.get("Directory code"),
            "Citations": document.get("Citations"),
            "PublicationInfo": document.get("PublicationInfo"),
        }

    def _process_and_insert_chunks(self, content: List[str],
    metadata: List[Dict[str, Any]]) -> None:
        """Process and insert chunks of content and metadata into the index.

        Args:
            content (List[str]): The list of content chunks to insert.
            metadata (List[Dict[str, Any]]): The corresponding metadata for the chunks.

        Raises:
            SearchException: If an error occurs during chunk processing and insertion.
        """
        try:
            docs = self.text_splitter.create_documents(content, metadatas=metadata)
            logger.debug("Split %s documents into %s chunks", len(content), len(docs))

            ElasticsearchStore.from_documents(
                docs,
                es_user=Config.ELASTICSEARCH_USERNAME,
                es_password=Config.ELASTICSEARCH_PASSWORD,
                es_url=f"http://{Config.ELASTICSEARCH_HOST}:{Config.ELASTICSEARCH_PORT}",
                index_name=Config.RAG_INDEX_NAME,
                embedding=self.embedding,
            )
            logger.debug("Chunks inserted into the index %s", Config.RAG_INDEX_NAME)
        except Exception as e:
            logger.error("Failed to process and insert chunks: %s", e)
            raise SearchException(f"Failed to process and insert chunks: {str(e)}") from e
        finally:
            content.clear()
            metadata.clear()

    @exception_handler("Error executing search")
    def search(self, **query_args: Any) -> Dict[str, Any]:
        """Search query in the specified index.

        Args:
            **query_args: Additional arguments for the search query.

        Returns:
            Dict[str, Any]: The search results.

        Raises:
            SearchException: If an error occurs during the search execution.
        """
        result = self.es.search(index=Config.INDEX_NAME, **query_args)
        logger.info("Search executed successfully with args: %s", query_args)
        return result

    @exception_handler("Error during multi-match search")
    def multi_match_search(self, query: str, query_fields: List[str],
    size: int = Config.ELASTIC_QUERY_SIZE, from_: int = 0) -> Dict[str, Any]:
        """Perform a multi-match query in the specified index.

        Args:
            query (str): The query string to search for.
            query_fields (List[str]): The fields to search in.
            size (int): The number of results to return.
            from_ (int): The starting point for pagination.

        Returns:
            Dict[str, Any]: The search results.

        Raises:
            SearchException: If an error occurs during the multi-match search execution.
        """
        result = self.es.search(
            index=Config.INDEX_NAME,
            query={
                "multi_match": {
                    "query": query,
                    "fields": query_fields,
                }
            },
            size=size, from_=from_
        )
        logger.info("Multi-match search executed successfully.")
        return result

    @exception_handler("Error during KNN search")
    def knn_search(self, query: str,
    size: int = Config.ELASTIC_QUERY_SIZE, from_: int = 0) -> Dict[str, Any]:
        """Perform a KNN query in the specified index.

        Args:
            query (str): The query string to search for.
            size (int): The number of results to return.
            from_ (int): The starting point for pagination.

        Returns:
            Dict[str, Any]: The KNN search results.

        Raises:
            SearchException: If an error occurs during the KNN search execution.
        """
        result = self.es.search(
            index=Config.INDEX_NAME,
            knn={
                "field": "embedding",
                "query_vector": self._get_embedding(query),
                "num_candidates": Config.KNN_CANDIDATES,
                "k": Config.KNN_SEARCH_K,
            },
            size=size,
            from_=from_
        )
        logger.info("KNN search executed successfully.")
        return result

    @exception_handler("Error during hybrid search")
    def hybrid_search(self, query: str, query_fields: List[str],
    size: int = Config.ELASTIC_QUERY_SIZE, from_: int = 0) -> Dict[str, Any]:
        """Perform a hybrid search query in the specified index.

        Args:
            query (str): The query string to search for.
            query_fields (List[str]): The fields to search in.
            size (int): The number of results to return.
            from_ (int): The starting point for pagination.

        Returns:
            Dict[str, Any]: The hybrid search results.

        Raises:
            SearchException: If an error occurs during the hybrid search execution.
        """
        result = self.es.search(
            index=Config.INDEX_NAME,
            query={
                "multi_match": {
                    "query": query,
                    "fields": query_fields,
                }
            },
            knn={
                "field": "embedding",
                "query_vector": self._get_embedding(query),
                "num_candidates": Config.KNN_CANDIDATES,
                "k": Config.KNN_SEARCH_K,
            },
            rank={
                "rrf": {
                    "rank_window_size": size + from_
                }
            },
            size=size,
            from_=from_
        )
        logger.info("Hybrid search executed successfully.")
        return result

    @exception_handler("Error during semantic search")
    def semantic_search(self, query: str,
    size: int = Config.ELASTIC_QUERY_SIZE, from_: int = 0) -> Dict[str, Any]:
        """Perform a semantic search query in the specified index.

        Args:
            query (str): The query string to search for.
            size (int): The number of results to return.
            from_ (int): The starting point for pagination.

        Returns:
            Dict[str, Any]: The semantic search results.

        Raises:
            SearchException: If an error occurs during the semantic search execution.
        """
        tfidf_score = self._get_tfidf_scores(query)
        result = self.es.search(
            index=Config.INDEX_NAME,
            query={
                "sparse_vector": {
                    "field": "sparse_embedding",
                    "query_vector": tfidf_score
                }
            },
            size=size,
            from_=from_
        )
        logger.info("Semantic search executed successfully.")
        return result

    @exception_handler("Error retrieving document")
    def retrieve_document(self, id_document: int) -> Dict[str, Any]:
        """Retrieve the document given the id.

        Args:
            id_document (int): The ID of the document to retrieve.

        Returns:
            Dict[str, Any]: The retrieved document.

        Raises:
            SearchException: If an error occurs while retrieving the document.
        """
        result = self.es.get(index=Config.INDEX_NAME, id=id_document)
        logger.info("Document retrieved successfully: %s", id_document)
        return result

    @exception_handler("Error retrieving index fields")
    def get_index_fields(self) -> List[str]:
        """Retrieve the index fields as a list.

        Returns:
            List[str]: A list of index fields.

        Raises:
            SearchException: If an error occurs while retrieving index fields.
        """
        mapping = self.es.indices.get_mapping(index=Config.INDEX_NAME)
        fields = list(mapping[Config.INDEX_NAME]["mappings"]["properties"])
        logger.info("Index fields retrieved successfully.")
        return fields

    @exception_handler("Error retrieving text index fields")
    def get_text_index_fields(self) -> List[str]:
        """Retrieve the text index fields as a list.

        Returns:
            List[str]: A list of text index fields.

        Raises:
            SearchException: If an error occurs while retrieving text index fields.
        """
        index_name = Config.INDEX_NAME
        mapping = self.es.indices.get_mapping(index=index_name)
        properties = mapping[index_name]["mappings"]["properties"]
        searchable_fields_types = ["text"]
        searchable_fields = []

        for field, field_info in properties.items():
            field_type = field_info.get("type")
            if field_type in searchable_fields_types:
                searchable_fields.append(field)

        logger.info("Text index fields retrieved successfully.")
        return searchable_fields

    @exception_handler("Error getting embedding")
    def _get_embedding(self, text: str) -> Any:
        """Get model embeddings for the provided text.

        Args:
            text (str): The input text to generate embeddings for.

        Returns:
            Any: The embedding vector for the input text.

        Raises:
            SearchException: If an error occurs while retrieving the embedding.
        """
        embedding = self.embedding.model.encode(text, show_progress_bar=False)
        logger.debug("Embedding retrieved successfully for text.")
        return embedding

    @exception_handler("Error fitting TF-IDF vectorizer")
    def fit_tfidf(self, documents: List[str]) -> None:
        """Fit the TF-IDF Vectorizer on the given documents.

        Args:
            documents (List[str]): A list of documents to fit the vectorizer on.

        Returns:
            None: This method does not return a value.

        Raises:
            SearchException: If an error occurs while fitting the vectorizer.
        """
        self.vectorizer = TfidfVectorizer(max_features=Config.VECTORIZER_MAX_FEATS,
                                        stop_words="english",
                                        token_pattern=r"\b[a-zA-Z]+\b")
        self.vectorizer.fit(documents)
        logger.info("TF-IDF vectorizer fitted successfully.")

    @exception_handler("Error getting TF-IDF scores")
    def _get_tfidf_scores(self, document: str) -> Dict[str, float]:
        """Get TF-IDF scores for a single document as a dictionary of {word: score}.

        Args:
            document (str): The document to evaluate.

        Returns:
            Dict[str, float]: A dictionary mapping words to their TF-IDF scores.

        Raises:
            ValueError: If the TF-IDF vectorizer has not been fitted.
            SearchException: If an error occurs while retrieving TF-IDF scores.
        """
        if self.vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf first.")

        tfidf_vector = self.vectorizer.transform([document])
        feature_names = self.vectorizer.get_feature_names_out()
        sparse_array = zip(feature_names, tfidf_vector.toarray().flatten())
        doc_tfidf = {word: float(score) for word, score in sparse_array if score > 0.0}
        logger.debug("TF-IDF scores retrieved successfully for document.")
        return doc_tfidf

    @exception_handler("Error creating vector query")
    def _vector_query(self, search_query: str) -> dict:
        """Generate a vector query for the retriever.

        Args:
            search_query (str): The query string to be embedded.

        Returns:
            dict: A dictionary containing the KNN query parameters.

        Raises:
            SearchException: If an error occurs during the query creation.
        """
        embedding_vector = self.embedding.embed_query(search_query)
        logger.info("Vector query created successfully for search query.")
        return {
            "knn": {
                "field": "embedding",
                "query_vector": embedding_vector,
                "k": Config.KNN_SEARCH_K,
                "num_candidates": Config.KNN_CANDIDATES,
            }
        }

"""Module responsible for the Elastic Search client that will perform the full search"""
import logging
from typing import List, Dict, Optional, Any
from elasticsearch import Elasticsearch
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_elasticsearch import ElasticsearchRetriever, ElasticsearchStore
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from models.text_embedding import TextEmbedding
from config import Config
from app_logging import setup_logging

# Setup logger
logger = logging.getLogger(__name__)
setup_logging(logger)

class SearchException(Exception):
    """Base class for exceptions in the Search module."""
    def __init__(self, message: str) -> None:
        super().__init__(message)

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

    def initialize_es_client(self) -> Elasticsearch:
        """Setup connection to the Elasticsearch client.

        Returns:
            Elasticsearch: An instance of the Elasticsearch client.

        Raises:
            SearchException: If the connection to Elasticsearch fails.
        """
        try:
            es = Elasticsearch(
                [{"scheme": "http", "host": Config.ELASTICSEARCH_HOST, 
                "port": Config.ELASTICSEARCH_PORT}],
                basic_auth=(Config.ELASTICSEARCH_USERNAME, Config.ELASTICSEARCH_PASSWORD),
                timeout=60
            )
            logger.info("Connected to Elasticsearch!")
            return es
        except Exception as e:
            logger.error("Failed to connect to Elasticsearch: %s", e)
            raise SearchException(f"Failed to initialize Elasticsearch client: {str(e)}") from e

    def initialize_text_embedding(self) -> TextEmbedding:
        """Setup embedding model to compute the embeddings.

        Returns:
            TextEmbedding: An instance of the embedding model.

        Raises:
            SearchException: If the initialization of the text embedding fails.
        """
        try:
            embedding = TextEmbedding(Config.EMBEDDING_MODEL)
            logger.info("Text embedding initialized successfully.")
            return embedding
        except Exception as e:
            logger.error("Failed to initialize text embedding: %s", e)
            raise SearchException(f"Failed to initialize text embedding: {str(e)}") from e

    def initialize_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Setup text splitter to divide documents into chunks.

        Returns:
            RecursiveCharacterTextSplitter: An instance of the text splitter.

        Raises:
            SearchException: If the initialization of the text splitter fails.
        """
        try:
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=Config.CHUNK_SIZE, 
                chunk_overlap=Config.CHUNK_OVERLAP
            )
            logger.info("Text splitter initialized successfully.")
            return text_splitter
        except Exception as e:
            logger.error("Failed to initialize text splitter: %s", e)
            raise SearchException(f"Failed to initialize text splitter: {str(e)}") from e

    def initialize_es_retriever(self) -> ElasticsearchRetriever:
        """Setup retriever used to find related documents based on the sparse embedding.

        Returns:
            ElasticsearchRetriever: An instance of the Elasticsearch retriever.

        Raises:
            SearchException: If the initialization of the Elasticsearch retriever fails.
        """
        try:
            retriever = ElasticsearchRetriever(
                es_client=self.es,
                index_name=Config.INDEX_NAME,
                body_func=self._vector_query,
                content_field=Config.QUERY_TEXT_FIELD,
                search_kwargs={'k': Config.K_DOCS_TO_RETRIEVE}
            )
            logger.info("Elasticsearch retriever initialized successfully.")
            return retriever
        except Exception as e:
            logger.error("Failed to initialize Elasticsearch retriever: %s", e)
            raise SearchException(f"Failed to initialize Elasticsearch retriever: {str(e)}") from e

    def initialize_vector_store(self) -> ElasticsearchStore:
        """Connect to the vector store containing the chunks.

        Returns:
            ElasticsearchStore: An instance of the vector store.

        Raises:
            SearchException: If the initialization of the vector store fails.
        """
        try:
            vector_store = ElasticsearchStore(
                es_user=Config.ELASTICSEARCH_USERNAME,
                es_password=Config.ELASTICSEARCH_PASSWORD,
                es_url=f"http://{Config.ELASTICSEARCH_HOST}:{Config.ELASTICSEARCH_PORT}",
                index_name=Config.RAG_INDEX_NAME,
                embedding=self.embedding
            )
            logger.info("Vector store initialized successfully.")
            return vector_store
        except Exception as e:
            logger.error("Failed to initialize vector store: %s", e)
            raise SearchException(f"Failed to initialize vector store: {str(e)}") from e

    def _create_main_index(self) -> None:
        """Create the main index.

        Raises:
            SearchException: If the creation of the main index fails.
        """
        try:
            logger.info("Creating main index")
            self.es.indices.delete(index=Config.INDEX_NAME, ignore_unavailable=True)
            self.es.indices.create(index=Config.INDEX_NAME, mappings=Config.INDEX_NAME_MAPPING)
            logger.info("Main index created successfully.")
        except Exception as e:
            logger.error("Failed to create main index: %s", e)
            raise SearchException(f"Failed to create main index: {str(e)}") from e

    def _create_rag_index(self) -> None:
        """Create the RAG index.

        Raises:
            SearchException: If the creation of the RAG index fails.
        """
        try:
            logger.info("Creating RAG index")
            self.es.indices.delete(index=Config.RAG_INDEX_NAME, ignore_unavailable=True)
            self.es.indices.create(index=Config.RAG_INDEX_NAME)
            logger.info("RAG index created successfully.")
        except Exception as e:
            logger.error("Failed to create RAG index: %s", e) 
            raise SearchException(f"Failed to create RAG index: {str(e)}") from e

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
            except Exception as e:
                logger.error("An issue occurred while inserting the documents: %s", e)
                raise SearchException(f"An issue occurred while inserting the documents: {str(e)}") from e

        # Insert any remaining documents
        if operations:
            self._execute_bulk_insert(operations)
            self._process_and_insert_chunks(content, metadata)

    def _prepare_bulk_operations(self, operations: List[Dict[str, Any]], document: Dict[str, Any]) -> None:
        """Prepare bulk operations by adding indexing instructions and document details.

        Args:
            operations (List[Dict[str, Any]]): The list of bulk operations to prepare.
            document (Dict[str, Any]): The document to prepare for indexing.

        Raises:
            SearchException: If an error occurs during the bulk preparation.
        """
        try:
            operations.append({"index": {"_index": Config.INDEX_NAME}})
            operations.append({
                **document,
                "embedding": self._get_embedding(document[Config.QUERY_TEXT_FIELD]),
                "sparse_embedding": self._get_tfidf_scores(document[Config.QUERY_TEXT_FIELD])
            })
        except Exception as e:
            logger.error("An error occurred during the bulk preparation: %s", e)
            raise SearchException(f"An error occurred during the bulk preparation: {str(e)}") from e

    def _execute_bulk_insert(self, operations: List[Dict[str, Any]]) -> None:
        """Execute bulk insert of operations into Elasticsearch.

        Args:
            operations (List[Dict[str, Any]]): The list of operations to execute.

        Raises:
            SearchException: If an error occurs during the bulk insert.
        """
        try:
            self.es.bulk(operations=operations)
            logger.info("Bulk insert executed successfully.")
            operations.clear()
        except Exception as e:
            logger.error("Error during bulk insert: %s", e)
            raise SearchException(f"Error during bulk insert: {str(e)}") from e

    def _extract_metadata(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant metadata from the document.

        Args:
            document (Dict[str, Any]): The document from which to extract metadata.

        Returns:
            Dict[str, Any]: A dictionary containing the extracted metadata.

        Raises:
            SearchException: If an error occurs during metadata extraction.
        """
        try:
            return {
                "CELEX_ID": document.get("CELEX_ID"),
                "Directory_code": document.get("Directory code"),
                "Citations": document.get("Citations"),
                "PublicationInfo": document.get("PublicationInfo"),
            }
        except Exception as e:
            logger.error("Failed to extract metadata: %s", e)
            raise SearchException(f"Failed to extract metadata: {str(e)}") from e

    def _process_and_insert_chunks(self, content: List[str], metadata: List[Dict[str, Any]]) -> None:
        """Process and insert chunks of content and metadata into the index.

        Args:
            content (List[str]): The list of content chunks to insert.
            metadata (List[Dict[str, Any]]): The corresponding metadata for the chunks.

        Raises:
            SearchException: If an error occurs during chunk processing and insertion.
        """
        try:
            docs = self.text_splitter.create_documents(content, metadatas=metadata)
            logger.info("Split %s documents into %s chunks", len(content), len(docs))

            ElasticsearchStore.from_documents(
                docs,
                es_user=Config.ELASTICSEARCH_USERNAME,
                es_password=Config.ELASTICSEARCH_PASSWORD,
                es_url=f"http://{Config.ELASTICSEARCH_HOST}:{Config.ELASTICSEARCH_PORT}",
                index_name=Config.RAG_INDEX_NAME,
                embedding=self.embedding,
            )
            content.clear()
            metadata.clear()
            logger.info("Chunks inserted into the index %s", Config.RAG_INDEX_NAME)
        except Exception as e:
            logger.error("Failed to process and insert chunks: %s", e)
            raise SearchException(f"Failed to process and insert chunks: {str(e)}") from e

    def search(self, **query_args: Any) -> Dict[str, Any]:
        """Search query in the specified index.

        Args:
            **query_args: Additional arguments for the search query.

        Returns:
            Dict[str, Any]: The search results.

        Raises:
            SearchException: If an error occurs during the search execution.
        """
        try:
            result = self.es.search(index=Config.INDEX_NAME, **query_args)
            logger.info("Search executed successfully with args: %s", query_args)
            return result
        except Exception as e:
            logger.error(f"Error executing search: {e}")
            raise SearchException(f"Error executing search: {str(e)}") from e

    def multi_match_search(self, query: str, query_fields: List[str], size: int = Config.ELASTIC_QUERY_SIZE, from_: int = 0) -> Dict[str, Any]:
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
        try:
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
        except Exception as e:
            logger.error(f"Error during multi-match search: {e}")
            raise SearchException(f"Error during multi-match search: {str(e)}") from e

    def knn_search(self, query: str, size: int = Config.ELASTIC_QUERY_SIZE, from_: int = 0) -> Dict[str, Any]:
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
        try:
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
        except Exception as e:
            logger.error(f"Error during KNN search: {e}")
            raise SearchException(f"Error during KNN search: {str(e)}") from e

    def hybrid_search(self, query: str, query_fields: List[str], size: int = Config.ELASTIC_QUERY_SIZE, from_: int = 0) -> Dict[str, Any]:
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
        try:
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
        except Exception as e:
            logger.error(f"Error during hybrid search: {e}")
            raise SearchException(f"Error during hybrid search: {str(e)}") from e

    def semantic_search(self, query: str, size: int = Config.ELASTIC_QUERY_SIZE, from_: int = 0) -> Dict[str, Any]:
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
        try:
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
        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            raise SearchException(f"Error during semantic search: {str(e)}") from e

    def retrieve_document(self, id_document: int) -> Dict[str, Any]:
        """Retrieve the document given the id.

        Args:
            id_document (int): The ID of the document to retrieve.

        Returns:
            Dict[str, Any]: The retrieved document.

        Raises:
            SearchException: If an error occurs while retrieving the document.
        """
        try:
            result = self.es.get(index=Config.INDEX_NAME, id=id_document)
            logger.info("Document retrieved successfully: %s", id_document)
            return result
        except Exception as e:
            logger.error(f"Error retrieving document {id_document}: {e}")
            raise SearchException(f"Error retrieving document {id_document}: {str(e)}") from e

    def get_index_fields(self) -> List[str]:
        """Retrieve the index fields as a list.

        Returns:
            List[str]: A list of index fields.

        Raises:
            SearchException: If an error occurs while retrieving index fields.
        """
        try:
            mapping = self.es.indices.get_mapping(index=Config.INDEX_NAME)
            fields = list(mapping[Config.INDEX_NAME]["mappings"]["properties"])
            logger.info("Index fields retrieved successfully.")
            return fields
        except Exception as e:
            logger.error("Error retrieving index fields: %s", e)
            raise SearchException(f"Error retrieving index fields: {str(e)}") from e

    def get_text_index_fields(self) -> List[str]:
        """Retrieve the text index fields as a list.

        Returns:
            List[str]: A list of text index fields.

        Raises:
            SearchException: If an error occurs while retrieving text index fields.
        """
        try:
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
        except Exception as e:
            logger.error("Error retrieving text index fields: %s", e)
            raise SearchException(f"Error retrieving text index fields: {str(e)}") from e

    def _get_embedding(self, text: str) -> Any:
        """Get model embeddings for the provided text.

        Args:
            text (str): The input text to generate embeddings for.

        Returns:
            Any: The embedding vector for the input text.

        Raises:
            SearchException: If an error occurs while retrieving the embedding.
        """
        try:
            embedding = self.embedding.model.encode(text, show_progress_bar=False)
            logger.info("Embedding retrieved successfully for text.")
            return embedding
        except Exception as e:
            logger.error("Error getting embedding: %s", e)
            raise SearchException(f"Error getting embedding: {str(e)}") from e

    def fit_tfidf(self, documents: List[str]) -> None:
        """Fit the TF-IDF Vectorizer on the given documents.

        Args:
            documents (List[str]): A list of documents to fit the vectorizer on.

        Returns:
            None: This method does not return a value.

        Raises:
            SearchException: If an error occurs while fitting the vectorizer.
        """
        try:
            self.vectorizer = TfidfVectorizer(max_features=Config.VECTORIZER_MAX_FEATS,
                                            stop_words="english",
                                            token_pattern=r"\b[a-zA-Z]+\b")
            self.vectorizer.fit(documents)
            logger.info("TF-IDF vectorizer fitted successfully.")
        except Exception as e:
            logger.error("Error fitting TF-IDF vectorizer: %s", e)
            raise SearchException(f"Error fitting TF-IDF vectorizer: {str(e)}") from e

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
        try:
            if self.vectorizer is None:
                raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf first.")

            tfidf_vector = self.vectorizer.transform([document])
            feature_names = self.vectorizer.get_feature_names_out()
            sparse_array = zip(feature_names, tfidf_vector.toarray().flatten())
            doc_tfidf = {word: float(score) for word, score in sparse_array if score > 0.0}
            logger.info("TF-IDF scores retrieved successfully for document.")
            return doc_tfidf
        except Exception as e:
            logger.error("Error getting TF-IDF scores: %s", e)
            raise SearchException(f"Error getting TF-IDF scores: {str(e)}") from e

    def _vector_query(self, search_query: str) -> dict:
        """Generate a vector query for the retriever.

        Args:
            search_query (str): The query string to be embedded.

        Returns:
            dict: A dictionary containing the KNN query parameters.

        Raises:
            SearchException: If an error occurs during the query creation.
        """
        try:
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
        except Exception as e:
            logger.error("Error creating vector query: %s", e)
            raise SearchException(f"Error creating vector query: {str(e)}") from e
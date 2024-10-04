"""Module responsible for the Elastic Search client that will perform the full search"""
import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

load_dotenv()
ELASTIC_QUERY_SIZE = int(os.environ.get("ELASTIC_QUERY_SIZE","10"))
KNN_CANDIDATES = int(os.environ.get("KNN_CANDIDATES","50"))
KNN_SEARCH_K = int(os.environ.get("KNN_SEARCH_K","10"))

class Search:
    """Elastic Search Client"""
    def __init__(self):
        """Init client connection to Elastic Search"""
        elasticsearch_username = os.environ.get("ELASTIC_USERNAME","elastic")
        elasticsearch_password = os.environ.get("ELASTIC_PASSWORD")
        elasticsearch_port = int(os.environ.get("ELASTICSEARCH_PORT","9200"))
        self.index = os.environ.get("INDEX_NAME","law_docs")
        embedding_model = os.environ.get("EMBEDDING_MODEL","all-MiniLM-L6-v2")
        self.es = Elasticsearch(
                    [{'scheme': 'http', 'host': 'localhost',
                    'port': elasticsearch_port}],
                    basic_auth=(elasticsearch_username, elasticsearch_password)
                    )
        self.model = SentenceTransformer(embedding_model)
        print('Connected to Elasticsearch!')

    def create_index(self):
        """Create a new index after previously deleting an index with the same name"""
        self.es.indices.delete(index=self.index,ignore_unavailable = True)
        self.es.indices.create(index=self.index, mappings={
            'properties': {
                'embedding': {
                    'type': 'dense_vector',
                }
            }
        })

    def insert_document(self, document):
        """Insert a document in the specified index"""
        return self.es.index(index=self.index, document={
            **document,
            "embedding" : self.get_embedding(document["Text"])
        })

    def insert_documents(self, documents, batch_size: int = 500):
        """Insert a list of documents in the specified index in batches."""
        operations = []
        for i, document in enumerate(documents):
            operations.append({"index": {"_index": self.index}})
            operations.append({
                **document,
                "embedding" : self.get_embedding(document["Text"])
                })

            # Once we reach the batch size, send the bulk request
            if (i + 1) % batch_size == 0:
                self.es.bulk(operations=operations)
                operations = []

        # Insert any remaining documents
        if operations:
            self.es.bulk(operations=operations)

    def search(self, **query_args):
        """Search query in the specified index"""
        return self.es.search(index=self.index, **query_args)

    def multi_match_search(self, query, query_fields,
    size : int = ELASTIC_QUERY_SIZE, from_ : int = 0):
        """Perform multi match query in the specified index"""
        return self.es.search(
                            index = self.index,
                            query={
                                'multi_match': {
                                    "query" : query,
                                    "fields" : query_fields,   
                                }
                            }, size=size, from_= from_
            )

    def knn_search(self, query, size : int = ELASTIC_QUERY_SIZE,
                   from_ : int = 0):
        """Perform knn query in the specified index"""
        return self.es.search(
                    index= self.index,
                    knn={
                        'field': 'embedding',
                        'query_vector': self.get_embedding(query),
                        'num_candidates': KNN_CANDIDATES,
                        'k': KNN_SEARCH_K,
                    },
                    size=size,
                    from_=from_
                )

    def retrieve_document(self, id_document : int):
        """Retrieve the document given the id"""
        return self.es.get(index=self.index, id=id_document)

    def get_index_fields(self) -> list[str]:
        """Retrieve the index fields as a list"""
        mapping = self.es.indices.get_mapping(index=self.index)
        return list(mapping[self.index]['mappings']['properties'])

    def get_text_index_fields(self) -> list[str]:
        """Retrieve the text index fields as a list"""
        index_name = self.index
        mapping = self.es.indices.get_mapping(index=index_name)
        properties = mapping[index_name]['mappings']['properties']
        searchable_fields_types = ["text"]
        searchable_fields = []

        for field, field_info in properties.items():
            field_type = field_info.get('type')
            if field_type in searchable_fields_types:
                searchable_fields.append(field)

        return searchable_fields

    def get_embedding(self, text : str):
        """Get model embeddings"""
        return self.model.encode(text)

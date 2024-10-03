"""Module responsible for the Elastic Search client that will perform the full search"""
from pprint import pprint
import os
#import time
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()
class Search:
    """Elastic Search Client"""
    def __init__(self):
        """Init client connection to Elastic Search"""
        elasticsearch_username = os.environ.get("ELASTIC_USERNAME","elastic")
        elasticsearch_password = os.environ.get("ELASTIC_PASSWORD")
        elasticsearch_port = int(os.environ.get("ELASTICSEARCH_PORT","9200"))
        self.es = Elasticsearch([{'scheme': 'http', 'host': 'localhost',
                                  'port': elasticsearch_port}],
                                    basic_auth=(elasticsearch_username, elasticsearch_password)
                                )
        client_info = self.es.info()
        print('Connected to Elasticsearch!')
        pprint(client_info.body)

    def create_index(self, index_name : str = "law_docs"):
        """Create a new index after previously deleting an index with the same name"""
        self.es.indices.delete(index=index_name,ignore_unavailable = True)
        self.es.indices.create(index=index_name)

    def insert_document(self, document, index_name : str = "law_docs"):
        """Insert a document in the specified index"""
        return self.es.index(index=index_name, body=document)

    def insert_documents(self, documents, index_name: str = "law_docs", batch_size: int = 500):
        """Insert a list of documents in the specified index in batches."""
        operations = []
        for i, document in enumerate(documents):
            operations.append({"index": {"_index": index_name}})
            operations.append(document)

            # Once we reach the batch size, send the bulk request
            if (i + 1) % batch_size == 0:
                self.es.bulk(operations=operations)
                operations = []

        # Insert any remaining documents
        if operations:
            self.es.bulk(operations=operations)

    def search(self, index_name : str = "law_docs", **query_args):
        """Search query in the specified index"""
        return self.es.search(index=index_name, **query_args)

    def retrieve_document(self, id_document : int, index_name : str = "law_docs"):
        """Retrieve the document given the id"""
        return self.es.get(index=index_name, id=id_document)
 